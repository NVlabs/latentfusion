import json

import math
import random

import numpy as np
import structlog
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F

from latentfusion import three, meshutils

logger = structlog.get_logger(__name__)


LINEMOD_ID_TO_NAME = {
    '000001': 'ape',
    '000002': 'benchvise',
    '000003': 'bowl',
    '000004': 'camera',
    '000005': 'can',
    '000006': 'cat',
    '000007': 'mug',
    '000008': 'driller',
    '000009': 'duck',
    '000010': 'eggbox',
    '000011': 'glue',
    '000012': 'holepuncher',
    '000013': 'iron',
    '000014': 'lamp',
    '000015': 'phone',
}


def inverse_transform(trans):
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


class BOPDataset(Dataset):

    def __init__(self,
                 dataset_path,
                 scene_path,
                 object_id,
                 center_object=False,
                 object_scale=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.scene_path = scene_path
        self.object_id = object_id

        if dataset_path.name == 'lm' or dataset_path.name == 'lmo':
            base_obj_scale = 1.0
            self.models_path = self.dataset_path / 'models'
        elif dataset_path.name == 'tless':
            base_obj_scale = 0.60
            self.models_path = self.dataset_path / 'models_reconst'
        elif dataset_path.name == 'lm_format':
            base_obj_scale = 1.0
            self.models_path = self.dataset_path / 'models'
        else:
            raise ValueError(f'Unknown dataset type {dataset_path.name}')

        self.model_path = self.models_path / f'obj_{self.object_id:06d}.ply'
        self.pointcloud_path = self.dataset_path / 'models_eval' / f'obj_{self.object_id:06d}.ply'

        models_info_path = self.dataset_path / 'models_eval' / 'models_info.json'
        with open(models_info_path, 'r') as f:
            self.model_info = json.load(f)[str(object_id)]

        self.center_object = center_object
        if object_scale is None:
            self.object_scale = base_obj_scale / self.model_info['diameter']
        else:
            self.object_scale = object_scale

        self.image_scale = 1.0
        self.bounds = torch.tensor([
            (self.model_info['min_x'], self.model_info['min_x'] + self.model_info['size_x']),
            (self.model_info['min_y'], self.model_info['min_y'] + self.model_info['size_y']),
            (self.model_info['min_z'], self.model_info['min_z'] + self.model_info['size_z']),
        ])
        self.centroid = self.bounds.mean(dim=1)

        self.depth_dir = self.scene_path / 'depth'
        self.mask_dir = self.scene_path / 'mask_visib'
        self.color_dir = self.scene_path / 'rgb'
        self.intrinsics_path = self.scene_path / 'scene_camera.json'
        self.extrinsics_path = self.scene_path / 'scene_gt.json'

        self.intrinsics, self.depth_scales = self.load_intrinsics(self.intrinsics_path)
        self.extrinsics, self.scene_object_inds = self.load_extrinsics(self.extrinsics_path)
        self.extrinsics = torch.stack(self.extrinsics, dim=0)

        # Compute quaternions for sampling.
        rotation, translation = three.decompose(self.extrinsics)
        self.quaternions = three.quaternion.mat_to_quat(rotation[:, :3, :3])

        self.depth_paths = sorted([self.depth_dir / f'{frame_ind:06d}.png'
                                   for frame_ind in self.scene_object_inds.keys()])
        self.mask_paths = [
            self.mask_dir / f'{frame_ind:06d}_{obj_ind:06d}.png'
            for frame_ind, obj_ind in self.scene_object_inds.items()
        ]
        self.color_paths = sorted([self.color_dir / f'{frame_ind:06d}.png'
                                   for frame_ind in self.scene_object_inds.keys()])

        assert len(self.depth_paths) == len(self.mask_paths)
        assert len(self.depth_paths) == len(self.color_paths)

    def load_pointcloud(self):
        obj = meshutils.Object3D(self.pointcloud_path)
        points = torch.tensor(obj.vertices, dtype=torch.float32)
        points = points * self.object_scale
        return points

    @classmethod
    def load_intrinsics(cls, path):
        intrinsics = []
        depth_scales = []
        with open(path, 'r') as f:
            intrinsics_json = json.load(f)
            keys = sorted([int(k) for k in intrinsics_json.keys()])
            for key in keys:
                value = intrinsics_json[str(key)]
                intrinsic_3x3 = value['cam_K']
                intrinsics.append(three.intrinsic_to_3x4(
                    torch.tensor(intrinsic_3x3).reshape(3, 3)).float())
                depth_scales.append(value['depth_scale'])

        return intrinsics, depth_scales

    def load_extrinsics(self, path):
        extrinsics = []
        scene_object_inds = {}
        with open(path, 'r') as f:
            extrinsics_json = json.load(f)
            frame_inds = sorted([int(k) for k in extrinsics_json.keys()])
            for frame_ind in frame_inds:
                for obj_ind, cam_d in enumerate(extrinsics_json[str(frame_ind)]):
                    if cam_d['obj_id'] == self.object_id:
                        rotation = torch.tensor(
                            cam_d['cam_R_m2c'], dtype=torch.float32).reshape(3, 3)
                        translation = torch.tensor(cam_d['cam_t_m2c'], dtype=torch.float32)
                        quaternion = three.quaternion.mat_to_quat(rotation)
                        extrinsics.append(three.to_extrinsic_matrix(translation, quaternion))
                        scene_object_inds[frame_ind] = obj_ind

        return extrinsics, scene_object_inds

    def __len__(self):
        return len(self.color_paths)

    def get_ids(self):
        return [p.stem for p in self.color_paths]

    def _load_color(self, path):
        image = Image.open(path)
        new_shape = (int(image.width * self.image_scale), int(image.height * self.image_scale))
        image = image.resize(new_shape)
        image = np.array(image)
        return image

    def _load_mask(self, path):
        image = Image.open(path)
        new_shape = (int(image.width * self.image_scale), int(image.height * self.image_scale))
        image = image.resize(new_shape)
        image = np.array(image, dtype=np.bool)
        if len(image.shape) > 2:
            image = image[:, :, 0]
        return image

    def _load_depth(self, path):
        image = Image.open(path)
        new_shape = (int(image.width * self.image_scale), int(image.height * self.image_scale))
        image = image.resize(new_shape)
        image = np.array(image, dtype=np.float32)
        return image

    def normalize_extrinsic(self, extrinsic):
        extrinsic = extrinsic.clone()
        if self.center_object:
            extrinsic = three.translate_matrix(extrinsic, -self.centroid.to(extrinsic.device))
        extrinsic[..., :3, 3] *= self.object_scale
        return extrinsic

    def denormalize_extrinsic(self, extrinsic):
        extrinsic = extrinsic.clone()
        extrinsic[..., :3, 3] /= self.object_scale
        if self.center_object:
            extrinsic = three.translate_matrix(extrinsic, self.centroid.to(extrinsic.device))
        return extrinsic

    def normalize_intrinsic(self, intrinsic):
        intrinsic = intrinsic.clone()
        intrinsic[..., :2, :] *= self.image_scale
        return intrinsic

    def denormalize_intrinsic(self, intrinsic):
        intrinsic = intrinsic.clone()
        intrinsic[..., :2, :] /= self.image_scale
        return intrinsic

    def sample_evenly(self, n):
        positions = three.extrinsic_to_position(self.extrinsics)
        _, inds = three.utils.farthest_points(positions,
                                              n_clusters=n,
                                              dist_func=F.pairwise_distance,
                                              return_center_indexes=True)
        return inds

    def __getitem__(self, idx):
        color = self._load_color(self.color_paths[idx])
        color = (torch.tensor(color).float() / 255.0).permute(2, 0, 1)
        mask = self._load_mask(self.mask_paths[idx])
        mask = torch.tensor(mask).bool()
        depth = self._load_depth(self.depth_paths[idx])
        depth = torch.tensor(depth) * self.object_scale * self.depth_scales[idx]

        intrinsic = self.normalize_intrinsic(self.intrinsics[idx])
        extrinsic = self.normalize_extrinsic(self.extrinsics[idx])

        return {
            'color': color,
            'mask': mask,
            'depth': depth,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
        }
