import json
import math
from pathlib import Path

import copy
import imageio
import numpy as np
import structlog
import torch
from tqdm import trange, tqdm

from latentfusion import utils, three, torchutils, imutils
from latentfusion.augment import gan_denormalize, gan_normalize
from latentfusion.modules.geometry import Camera
from latentfusion.pointcloud import compute_point_mask

logger = structlog.get_logger(__name__)


def render_observation(renderer, scene):
    color, depth, mask = renderer.render(scene)
    camera = Camera(scene.intrinsic, scene.extrinsic,
                    width=renderer.width, height=renderer.height)

    return Observation(color.permute(2, 0, 1).unsqueeze(0),
                       depth.unsqueeze(0).unsqueeze(0),
                       mask.unsqueeze(0).unsqueeze(0),
                       camera,
                       object_scale=scene.obj.scale)


def render_random_observations(renderer, scene, n,
                               x_bound=(0.0, 0.0), y_bound=(0.0, 0.0), z_bound=(0.5, 0.5),
                               disk_sample_quats=True,
                               frame='default'):
    translations = three.rigid.random_translation(n, x_bound, y_bound, z_bound)
    if disk_sample_quats:
        quaternions = three.orientation.evenly_distributed_quats(n)
        # quaternions = three.orientation.disk_sample_quats(n, min_angle=math.pi / 12)
    else:
        quaternions = three.quaternion.random(n)
    observations = []
    for trans, quat in zip(translations, quaternions):
        scene.set_pose(trans.squeeze(0), quat.squeeze(0), frame=frame)
        observations.append(render_observation(renderer, scene))

    return Observation.collate(observations)


def sample_eval_observations(renderer, scene, x_bound=(0, 0), y_bound=(0, 0), z_bound=(0.5, 0.5),
                             rot_std_rad=math.pi / 12, trans_std_m=(0.01, 0.01, 0.05)):
    # Sample reference.
    ref_trans = three.rigid.random_translation(1, x_bound, y_bound, z_bound).squeeze(0)
    ref_quat = three.quaternion.random(1).squeeze(0)
    scene.set_pose(ref_trans, ref_quat)
    ref_obs = render_observation(renderer, scene)

    # Sample target.
    tar_quat = three.quaternion.perturb(ref_quat, rot_std_rad)
    # Resample if angle > 45 deg
    while three.quaternion.angular_distance(tar_quat, ref_quat) >= math.pi / 4:
        tar_quat = three.quaternion.perturb(ref_quat, rot_std_rad)

    tar_trans = ref_trans + torch.randn_like(ref_trans) * torch.tensor(trans_std_m)
    scene.set_pose(tar_trans, tar_quat)
    tar_obs = render_observation(renderer, scene)

    return ref_obs, tar_obs


class Observation(object):

    @classmethod
    def from_dataset(cls, dataset, inds=None):
        if inds is None:
            inds = torch.arange(len(dataset))
        loader = torchutils.IndexedDataLoader(
            dataset, shuffle=False, indices=inds, batch_size=len(inds), num_workers=0)
        return cls.from_dict(next(iter(loader)))

    @classmethod
    def from_dict(cls, d):
        height, width = d['color'].shape[-2:]
        camera = Camera(d['intrinsic'], d['extrinsic'], width=width, height=height)
        return cls(d['color'],
                   d['depth'].unsqueeze(-3),  # Create channel dimension.
                   d['mask'].unsqueeze(-3).float(),
                   camera)

    def __init__(self, color, depth, mask, camera, **kwargs):
        if len(color.shape) == 3:
            color = color.unsqueeze(0)
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        self.color = color
        self.depth = depth
        self.mask = mask
        self.camera = camera

        self.meta = {
            # A scale factor relative to this object's original scale.
            # Useful for 3rd party datasets with different scale objects.
            'object_scale': kwargs.get('object_scale', 1.0),
            'is_zoomed': kwargs.get('is_zoomed', False),
            'is_normalized': kwargs.get('is_normalized', False),
            'is_prepared': kwargs.get('is_prepared', False),
        }

    @property
    def device(self):
        return self.color.device

    def __len__(self):
        return len(self.camera)

    def __getitem__(self, item):
        return Observation(self.color[item], self.depth[item], self.mask[item], self.camera[item],
                           **self.meta)

    def clone(self):
        return Observation(self.color.clone(), self.depth.clone(), self.mask.clone(),
                           self.camera.clone(),
                           **self.meta)

    @classmethod
    def collate(cls, observations):
        color = torch.cat([o.color for o in observations], dim=0)
        depth = torch.cat([o.depth for o in observations], dim=0)
        mask = torch.cat([o.mask for o in observations], dim=0)
        camera = Camera.cat([o.camera for o in observations])
        return cls(color, depth, mask, camera, **observations[0].meta)

    def to_list(self):
        observations = []
        for i in range(len(self)):
            observations.append(Observation(self.color[i].unsqueeze(0),
                                            self.depth[i].unsqueeze(0),
                                            self.mask[i].unsqueeze(0),
                                            self.camera[i],
                                            **self.meta))

        return observations

    def to(self, device):
        return Observation(self.color.to(device),
                           self.depth.to(device),
                           self.mask.to(device),
                           self.camera.clone().to(device),
                           **self.meta)

    def expand(self, n):
        if len(self) > 1:
            raise ValueError(f"Must be single but has batch size {len(self)}.")

        return Observation(self.color.expand(n, -1, -1, -1),
                           self.depth.expand(n, -1, -1, -1),
                           self.mask.expand(n, -1, -1, -1),
                           self.camera.repeat(n),
                           **self.meta)

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        logger.info("saving observation", path=path)

        camera_json = self.camera.to_kwargs()
        camera_json['meta'] = self.meta

        with open(path / f'cameras.json', 'w') as f:
            json.dump(camera_json, f, indent=2, cls=utils.MyEncoder)

        for i in trange(len(self)):
            color_im = (255.0 * self.color[i].permute(1, 2, 0).numpy()).astype(np.uint8)
            depth_im = (1000.0 * self.depth[i][0]).numpy().astype(np.uint16)
            mask_im = self.mask[i][0].numpy().astype(np.uint8) * 255

            imageio.imsave(path / f'{i:04d}.color.png', color_im)
            imageio.imsave(path / f'{i:04d}.depth.png', depth_im)
            imageio.imsave(path / f'{i:04d}.mask.png', mask_im)

    @classmethod
    def load(cls, path, frames=None) -> 'Observation':
        if isinstance(path, str):
            path = Path(path)

        with open(path / f'cameras.json', 'r') as f:
            camera_json = json.load(f)
        if 'meta' in camera_json:
            meta = camera_json.pop('meta')
        else:
            meta = {}

        cameras = Camera(**{
            k: torch.tensor(v, dtype=torch.float32) if isinstance(v, list) else v
            for k, v in camera_json.items()
        })

        color_ims = []
        depth_ims = []
        mask_ims = []
        if frames is None:
            inds = list(range(len(cameras)))
        elif isinstance(frames, int):
            inds = [frames]
        else:
            inds = frames

        cameras = cameras[inds]

        for i in inds:
            color_ims.append(imageio.imread(path / f"{i:04d}.color.png").astype(np.float32) / 255.0)
            depth_ims.append(
                imageio.imread(path / f"{i:04d}.depth.png").astype(np.float32) / 1000.0)
            mask_ims.append(imageio.imread(path / f"{i:04d}.mask.png").astype(np.bool))

        color = torch.stack([torch.tensor(x).permute(2, 0, 1) for x in color_ims], dim=0)
        depth = torch.stack([torch.tensor(x).unsqueeze(0) for x in depth_ims], dim=0)
        mask = torch.stack([torch.tensor(x).float().unsqueeze(0) for x in mask_ims], dim=0)

        return cls(color, depth, mask, cameras, **meta)

    def zoom(self, target_dist, target_size, camera: Camera = None):
        if camera is None:
            camera = self.camera

        color, new_camera = camera.zoom(self.color, target_size, target_dist, scale_mode='bilinear')
        depth, _ = camera.zoom(self.depth, target_size, target_dist, scale_mode='nearest')
        mask, _ = camera.zoom(self.mask, target_size, target_dist, scale_mode='nearest')

        kwargs = copy.deepcopy(self.meta)
        kwargs['is_zoomed'] = True

        return Observation(color, depth, mask, new_camera, **kwargs)

    def uncrop(self, camera=None):
        if camera is None:
            camera = self.camera

        color, new_camera = camera.uncrop(self.color, scale_mode='bilinear')
        depth, _ = camera.uncrop(self.depth, scale_mode='nearest')
        mask, _ = camera.uncrop(self.mask, scale_mode='nearest')

        kwargs = copy.deepcopy(self.meta)
        kwargs['is_zoomed'] = False

        return Observation(color, depth, mask, new_camera, **kwargs)

    def prepare(self, crop_color=True, crop_depth=True):
        if crop_color:
            color = gan_denormalize(gan_normalize(self.color) * self.mask)
        else:
            color = self.color
        if crop_depth:
            depth = self.depth * self.mask
        else:
            depth = self.depth

        kwargs = copy.deepcopy(self.meta)
        kwargs['is_prepared'] = True

        return Observation(color, depth, self.mask.clone(), self.camera.clone(), **kwargs)

    def normalize(self):
        color = gan_normalize(self.color)
        depth = self.camera.normalize_depth(self.depth)

        kwargs = copy.deepcopy(self.meta)
        kwargs['is_normalized'] = True

        return Observation(color, depth, self.mask.clone(), self.camera.clone(), **kwargs)

    def denormalize(self):
        color = gan_denormalize(self.color)
        depth = self.camera.denormalize_depth(self.depth)

        kwargs = copy.deepcopy(self.meta)
        kwargs['is_normalized'] = False

        return Observation(color, depth, self.mask.clone(), self.camera.clone(), **kwargs)

    def estimate_camera(self) -> Camera:
        from latentfusion.pose.initialization import estimate_initial_pose
        return estimate_initial_pose(self.depth, self.mask, self.camera.intrinsic,
                                     self.camera.width, self.camera.height).to(self.device)

    def zoom_estimate(self, target_dist, target_size):
        return self.zoom(target_dist, target_size, camera=self.estimate_camera())

    def pointcloud(self, frame='object', return_colors=False, segment=True):
        if frame == 'object':
            points = (torch.stack(self.camera.depth_object_coords(self.depth), dim=-1)
                      .view(len(self), -1, 3))
        else:
            points = (torch.stack(self.camera.depth_camera_coords(self.depth), dim=-1)
                      .view(len(self), -1, 3))

        if segment:
            mask = self.mask.bool()  # & (obs.depth < 3.0)
            point_mask = compute_point_mask(self.camera, mask, points)
            points = points[point_mask]
        else:
            point_mask = None

        points = points.view(-1, 3)

        if return_colors:
            masked_colors = self.color.permute(0, 2, 3, 1).view(len(self), -1, 3)
            if point_mask is not None:
                masked_colors = torch.cat([masked_colors[i, pm] for i, pm in enumerate(point_mask)],
                                          dim=0)
            return points, masked_colors.view(-1, 3)

        return points

    def dilate(self, kernel_size=5):
        obs = self.clone()
        pad_color = imutils.mean_color(obs.color, obs.mask).mean(dim=0)
        pad_color = pad_color.view(1, 3, 1, 1).expand_as(obs.color)
        fg_mask = self.mask
        dilated_mask = imutils.dilate(obs.mask, 1, kernel_size)
        pad_mask = dilated_mask - fg_mask
        bg_mask = (1.0 - dilated_mask).clamp(min=0)
        obs.color = (fg_mask * obs.color + bg_mask * obs.color + pad_mask * pad_color)
        obs.mask = dilated_mask

        return obs
