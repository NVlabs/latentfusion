import csv
import logging
from pathlib import Path

import imageio
import math
import os
import random
import structlog
import torch
from pyrender import MetallicRoughnessMaterial
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import transforms

import latentfusion.three.rigid
from latentfusion import augment
from latentfusion import rendering
from latentfusion import consts
from latentfusion import three
from latentfusion import utils

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np

_package_dir = Path(os.path.dirname(os.path.realpath(__file__)))
_resources_dir = _package_dir.parent.parent.parent / 'resources'

logger = structlog.get_logger(__name__)

# This defines the reference camera coordinate frame.
# Facing the positive z direction.

# ShapeNet uses +Y as up. YCB uses +Z as up. Swap these.
OBJ_DEFAULT_POSE = torch.tensor((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))


def _load_roughness_values():
    path = _resources_dir / 'merl_blinn_phong.csv'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        glossiness = np.array([float(row[-1]) for row in reader])

    roughness = (2 / (glossiness + 2)) ** (1.0 / 4.0)
    return roughness


def set_egl_device(rel_device_id):
    abs_device_id = utils.absolute_device_id(rel_device_id)
    os.environ['EGL_DEVICE_ID'] = str(abs_device_id)


def _index_paths(dataset_dir, ext, index_name='paths.txt'):
    index_path = (dataset_dir / index_name)
    if index_path.exists():
        with open(index_path, 'r') as f:
            return [Path(dataset_dir, p.strip()) for p in f.readlines()]
    else:
        return list(dataset_dir.glob(f'**/*{ext}'))


class PyrenderDataset(IterableDataset):

    def __init__(self,
                 shape_paths,
                 num_input_views,
                 num_output_views,
                 x_bound=(-0.5, 0.5),
                 y_bound=None,
                 z_bound=(1.5, 3),
                 size_jitter=(0.5, 1.0),
                 color_noise_level=0.0,
                 depth_noise_level=0.0,
                 mask_noise_p=0.0,
                 min_lights=3,
                 max_lights=8,
                 width=640,
                 height=480,
                 device_id=0,
                 camera_angle_min=0.0,
                 camera_angle_max=math.pi / 2.0,
                 camera_angle_spread=math.pi / 12.0,
                 camera_translation_noise=0.0,
                 camera_rotation_noise=0.0,
                 color_background_dir=None,
                 depth_background_dir=None,
                 textures_dir=None,
                 use_textures=False,
                 random_materials=False,
                 color_random_background=False,
                 depth_random_background=False,
                 use_spiral_outputs=False,
                 use_constrained_cameras=False,
                 disk_sample_cameras=False,
                 use_model_materials=False,
                 obj_default_pose=OBJ_DEFAULT_POSE):
        self.width = width
        self.height = height

        if not y_bound:
            y_bound = (x_bound[0] / self.width * self.height,
                       x_bound[1] / self.width * self.height)

        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.size_jitter = size_jitter
        self.min_lights = min_lights
        self.max_lights = max_lights
        self.color_noise_level = color_noise_level
        self.depth_noise_level = depth_noise_level
        self.mask_noise_p = mask_noise_p
        self.color_random_background = color_random_background
        self.depth_random_background = depth_random_background
        self.random_materials = random_materials

        self.num_inputs = num_input_views
        self.num_outputs = num_output_views
        self.use_spiral_outputs = use_spiral_outputs
        self.use_constrained_cameras = use_constrained_cameras
        self.disk_sample_cameras = disk_sample_cameras
        self.camera_angle_min = camera_angle_min
        self.camera_angle_max = camera_angle_max
        self.camera_angle_spread = camera_angle_spread
        self.camera_translation_noise = camera_translation_noise
        self.camera_rotation_noise = camera_rotation_noise

        # Object poses will be pre-rotated to this pose.
        self.obj_default_pose = obj_default_pose

        self.shape_paths = shape_paths

        self.roughness_values = _load_roughness_values()
        self.use_model_materials = use_model_materials

        self.use_textures = use_textures
        if use_textures:
            self.textures_dir = Path(textures_dir)
            logger.info("indexing textures", path=self.textures_dir)
            self.texture_paths = _index_paths(self.textures_dir, ext='.jpg')
        else:
            self.textures_dir = None
            self.texture_paths = []

        if self.color_random_background and color_background_dir:
            self.color_background_dir = Path(color_background_dir)
            logger.info("indexing color backgrounds", path=self.color_background_dir)
            self.color_background_paths = _index_paths(self.color_background_dir, ext='.jpg')
        else:
            self.color_background_dir = None
            self.color_background_paths = []

        if self.depth_random_background and depth_background_dir:
            self.depth_background_dir = Path(depth_background_dir)
            logger.info("indexing depth backgrounds", path=self.depth_background_dir)
            self.depth_background_paths = _index_paths(self.depth_background_dir,
                                                       ext='.png',
                                                       index_name='depth_paths.txt')
        else:
            self.depth_background_dir = None
            self.depth_background_paths = []

        logger.info("dataset indexed",
                    num_shapes=len(self.shape_paths),
                    num_textures=len(self.texture_paths),
                    num_color_backgrounds=len(self.color_background_paths),
                    num_depth_backgrounds=len(self.depth_background_paths))

        self._color_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
            transforms.ToTensor(),
        ])
        self._mask_transform = transforms.Compose([
            # augment.masks.RandomMorphologicalTransform(p=0.5),
            # augment.masks.RandomTranslation(p=0.3),
            # augment.masks.RandomRotation(p=0.3),
            augment.masks.RandomAdd(p=0.15),
            augment.masks.RandomCut(p=0.05),
            augment.masks.RandomEllipses(p=0.2),
        ])
        self._color_bg_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((480, 640), pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
        ])
        self._depth_bg_transform = transforms.Compose([
            augment.tensors.TensorRandomHorizontalFlip(),
            augment.tensors.TensorRandomVerticalFlip(),
            augment.tensors.TensorRandomCrop((480, 640), pad_if_needed=True,
                                             padding_mode='reflect'),
        ])

        self._renderer = None
        self._worker_id = None
        self._log = None

        self.device_id = device_id
        set_egl_device(device_id)

    def load_random_image(self, paths):
        while True:
            image_path = random.choice(paths)
            try:
                image = imageio.imread(image_path)
                if len(image.shape) != 3 or image.shape[2] < 3:
                    continue
                return image[:, :, :3]
            except Exception:
                self._log.warning("failed to read image", path=image_path)

    def load_random_depth(self, paths):
        far = random.randrange(self.z_bound[1], 6.0)
        while True:
            image_path = random.choice(paths)
            try:
                depth = imageio.imread(image_path)
                if len(depth.shape) > 2:
                    depth = depth[:, :, 0]
                depth = torch.tensor(depth.astype(np.float32) / 1000.0)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * far
                return depth
            except Exception as e:
                self._log.warning("failed to read depth image", path=image_path, exc_info=e)

    def get_random_material(self):
        roughness = random.choice(self.roughness_values)
        metalness = random.uniform(0.0, 1.0)

        if self.use_textures and random.random() < 0.9:
            image = self.load_random_image(self.texture_paths)
            # base_color = [1.0, 1.0, 1.0]
            base_color = np.random.uniform(1.0, 2.0, 3)
        else:
            base_color = np.random.uniform(0.2, 1.0, 3)
            image = None

        return MetallicRoughnessMaterial(
            alphaMode='BLEND',
            roughnessFactor=roughness,
            metallicFactor=metalness,
            baseColorFactor=base_color,
            baseColorTexture=image,
        )

    def random_poses(self, n, constrained=False, disk_sample=False):
        translation = latentfusion.three.rigid.random_translation(n, self.x_bound, self.y_bound,
                                                                  self.z_bound)

        if constrained:
            angle = random.uniform(self.camera_angle_min + self.camera_angle_spread,
                                   self.camera_angle_max - self.camera_angle_spread)
            rot_quats = three.orientation.sample_segment_quats(
                n=n,
                up=(0.0, 0.0, 1.0),
                min_angle=angle - self.camera_angle_spread,
                max_angle=angle + self.camera_angle_spread)
        else:
            if disk_sample:
                rot_quats = three.orientation.evenly_distributed_quats(n)
            else:
                rot_quats = three.quaternion.random(n)

        # Rotate to canonical YCB pose (+Z is up)
        canon_quat = (three.quaternion.mat_to_quat(self.obj_default_pose)
                      .unsqueeze(0)
                      .expand_as(rot_quats))
        # Apply sampled rotated.
        rot_quats = three.quaternion.qmul(rot_quats, canon_quat)
        return translation, rot_quats

    def orbit_poses(self, n):
        translation = torch.tensor((0.0, 0.0, self.z_bound[0])).unsqueeze(0).expand(n, -1)

        rot_quat = three.orientation.spiral_orbit(n, c=8)
        # Rotate to canonical YCB pose (+Z is up)
        canon_quat = (three.quaternion.mat_to_quat(self.obj_default_pose)
                      .unsqueeze(0)
                      .expand_as(rot_quat))
        rot_quat = three.quaternion.qmul(rot_quat, canon_quat)
        return translation, rot_quat

    def worker_init_fn(self, worker_id):
        self._worker_id = worker_id
        self._log = logger.bind(worker_id=worker_id)
        self._renderer = rendering.Renderer(width=self.width, height=self.height)
        self._log.info('renderer initialized')

        # Suppress trimesh warnings.
        logging.getLogger('trimesh').setLevel(logging.ERROR)

    def __iter__(self):
        while True:
            yield self._get_item()

    def _get_item(self):
        intrinsic = torch.tensor(consts.INTRINSIC)

        in_translations, in_quaternions = self.random_poses(
            self.num_inputs,
            constrained=self.use_constrained_cameras,
            disk_sample=self.disk_sample_cameras)

        if self.use_spiral_outputs:
            out_translations, out_quaternions = self.orbit_poses(self.num_outputs)
        else:
            out_translations, out_quaternions = self.random_poses(
                self.num_outputs, disk_sample=self.disk_sample_cameras)

        size_jitter = random.uniform(*self.size_jitter)
        while True:
            model_path = random.choice(self.shape_paths)
            file_size = model_path.stat().st_size
            max_size = 2e7
            if file_size > max_size:
                self._log.warning('skipping large model',
                                  path=model_path, max_size=max_size, file_size=file_size)
                continue

            try:
                obj, obj_scale = rendering.load_object(model_path, size=size_jitter,
                                                       load_materials=self.use_model_materials)
                context = rendering.SceneContext(obj, intrinsic)
                break
            except ValueError as e:
                self._log.error('exception while loading mesh', exc_info=e)

        # Assign random materials.
        if self.random_materials:
            for primitive in context.object_node.mesh.primitives:
                primitive.material = self.get_random_material()
                uv_scale = random.uniform(1 / 8, 1.0)
                if primitive.texcoord_0 is not None:
                    primitive.texcoord_0 *= uv_scale
                if primitive.texcoord_1 is not None:
                    primitive.texcoord_1 *= uv_scale

        if self.color_random_background:
            color_bg_base = self.load_random_image(self.color_background_paths)
        else:
            color_bg_base = None

        if self.depth_random_background:
            depth_bg_base = self.load_random_depth(self.depth_background_paths)
        else:
            depth_bg_base = None

        # Render views.
        in_renders = []
        in_depths = []
        in_masks = []
        in_gt_renders = []
        in_gt_depths = []
        in_gt_masks = []
        out_gt_renders = []
        out_gt_depths = []
        out_gt_masks = []

        for translation, quaternion in zip(in_translations, in_quaternions):
            context.randomize_lights(self.min_lights, self.max_lights)
            context.set_pose(translation, quaternion)
            color, depth, mask = self._renderer.render(context)
            in_gt_renders.append(color)
            in_gt_masks.append(mask)
            in_gt_depths.append(depth)
            color = self._color_transform(color.permute(2, 0, 1)).permute(1, 2, 0)
            if color_bg_base is not None:
                color_bg = self._color_bg_transform(color_bg_base).permute(1, 2, 0)
                color = mask[:, :, None] * color + (1.0 - mask[:, :, None]) * color_bg
            if depth_bg_base is not None:
                depth_bg = self._depth_bg_transform(depth_bg_base.unsqueeze(0)).squeeze(0)
                depth = mask * depth + (1.0 - mask) * depth_bg

            if self.color_noise_level > 0.0:
                color = augment.add_noise(color, level=self.color_noise_level)
            if self.depth_noise_level > 0.0:
                depth = augment.add_noise_depth_cuda(depth.cuda(),
                                                     level=self.depth_noise_level).cpu()

            mask = torch.round(mask)
            if random.random() < self.mask_noise_p:
                mask = self._mask_transform(mask.bool()).float()
            in_renders.append(color)
            in_depths.append(depth)
            in_masks.append(mask)

        for translation, quaternion in zip(out_translations, out_quaternions):
            context.set_pose(translation, quaternion)
            color, depth, mask = self._renderer.render(context)
            out_gt_renders.append(color)
            out_gt_depths.append(depth)
            out_gt_masks.append(mask)

        del context

        in_intrinsic = intrinsic.unsqueeze(0).expand(self.num_inputs, -1, -1)
        in_extrinsic_gt = three.rigid.to_extrinsic_matrix(in_translations, in_quaternions)

        # Compute camera translation jitter.
        if self.camera_translation_noise > 0.0:
            translation_jitter = torch.randn_like(in_translations) * self.camera_translation_noise
            in_translations_noisy = in_translations + translation_jitter
        else:
            in_translations_noisy = in_translations

        # Compute camera rotation jitter.
        if self.camera_rotation_noise > 0.0:
            in_quaternions_noisy = three.quaternion.perturb(in_quaternions,
                                                            self.camera_rotation_noise)
        else:
            in_quaternions_noisy = in_quaternions

        in_extrinsic = three.rigid.to_extrinsic_matrix(in_translations_noisy, in_quaternions_noisy)
        out_intrinsic = intrinsic.unsqueeze(0).expand(self.num_outputs, -1, -1)
        out_extrinsic = three.rigid.to_extrinsic_matrix(out_translations, out_quaternions)

        return {
            'in': {
                'render': torch.stack(in_renders, dim=0).permute(0, 3, 1, 2),
                'mask': torch.stack(in_masks, dim=0),
                'depth': torch.stack(in_depths, dim=0),
                'extrinsic': in_extrinsic,
                'intrinsic': in_intrinsic,
            },
            'in_gt': {
                'render': torch.stack(in_gt_renders, dim=0).permute(0, 3, 1, 2),
                'mask': torch.stack(in_gt_masks, dim=0),
                'depth': torch.stack(in_gt_depths, dim=0),
                'extrinsic': in_extrinsic_gt,
                'intrinsic': in_intrinsic,
            },
            'out_gt': {
                'render': torch.stack(out_gt_renders, dim=0).permute(0, 3, 1, 2),
                'mask': torch.stack(out_gt_masks, dim=0),
                'depth': torch.stack(out_gt_depths, dim=0),
                'extrinsic': out_extrinsic,
                'intrinsic': out_intrinsic,
            },
        }
