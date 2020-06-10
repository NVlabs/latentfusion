import json
from pathlib import Path

import numpy as np
import structlog
import torch
from PIL import Image
from plyfile import PlyData
from sklearn.ensemble import IsolationForest
from torch.nn import functional as F
from torch.utils.data import Dataset

from latentfusion import three

logger = structlog.get_logger(__name__)


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


def read_open3d_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(inverse_transform(mat))
            metastr = f.readline()
    return torch.tensor(np.stack(traj, axis=0), dtype=torch.float32)


def _parse_kinectfusion_poses(poses_path):
    with open(poses_path, 'r') as f:
        text = f.read()

    lines = text.split('\n')

    num_cameras = len(lines) // 4
    poses = []
    for i in range(num_cameras):
        cam_lines = lines[i * 4 + 1:i * 4 + 4]
        pose = torch.tensor([[float(v) for v in line.split(' ')] for line in cam_lines])
        poses.append(pose)

    poses = torch.stack(poses, dim=0)
    # Make matrix 4x4.
    bottom = (torch.tensor((0.0, 0.0, 0.0, 1.0))
              .view(1, 1, 4)
              .expand(num_cameras, -1, -1))
    poses = torch.cat((poses, bottom), dim=1)
    return poses


def read_kinectfusion_trajectory(poses_path):
    rel_poses = _parse_kinectfusion_poses(poses_path)

    canon_pose = rel_poses[0]
    abs_poses = [rel_poses[0]]
    for rel_pose in rel_poses[1:]:
        abs_poses.append(rel_pose @ canon_pose)
    abs_poses = torch.stack(abs_poses, dim=0)

    return abs_poses


def _filter_points(points, n_estimators=100):
    clf = IsolationForest(n_estimators=n_estimators,
                          contamination=0.1)
    y = clf.fit_predict(points.numpy())
    y = torch.tensor(y)
    num_valid = (y > 0).sum()
    num_filtered = (y <= 0).sum()
    logger.info('filtered points',
                num_filtered=num_filtered.item(), num_valid=num_valid.item())
    return points[(y > 0), :]


class RealsenseDataset(Dataset):

    def __init__(self,
                 scene_paths,
                 image_scale=0.2,
                 object_scale='auto',
                 center_object=True,
                 odometry_type='open3d',
                 use_registration=True,
                 mask_type='default',
                 ref_points=None):
        if isinstance(scene_paths, Path):
            scene_paths = [scene_paths]

        self.scene_paths = [Path(p) for p in scene_paths]

        self.odometry_type = odometry_type
        self.use_registration = use_registration

        self.mask_paths = []
        self.depth_paths = []
        self.color_paths = []
        self.intrinsics = []
        self.extrinsics = []
        self.points = []

        if mask_type == 'plane':
            mask_folder = 'mask-plane'
        else:
            mask_folder = 'mask'

        for path in scene_paths:
            intrinsics = self.load_intrinsics(path)
            mask_dir = path / mask_folder
            if not mask_dir.exists():
                raise ValueError(f"Mask directory {mask_dir!s} does not exist.")

            mask_paths = sorted(mask_dir.glob('*.png'))
            valid_ids = [int(p.stem) for p in mask_paths]
            depth_paths = [path / 'depth' / p.name for p in mask_paths]
            color_paths = [path / 'color' / p.with_suffix('.jpg').name for p in mask_paths]

            self.intrinsics.extend([intrinsics] * len(valid_ids))
            self.mask_paths.extend(mask_paths)
            self.depth_paths.extend(depth_paths)
            self.color_paths.extend(color_paths)

            if odometry_type is not None:
                extrinsics = self.load_extrinsics(path)
                extrinsics = extrinsics[valid_ids]
                self.extrinsics.extend(extrinsics)

                points = self.load_points(path)
                self.points.append(points)

        self.intrinsics = torch.stack(self.intrinsics, dim=0)

        if odometry_type is not None:
            self.extrinsics = torch.stack(self.extrinsics, dim=0)
            self.quaternions = three.extrinsic_to_quat(self.extrinsics)
            self.points = torch.cat(self.points, dim=0)
            self.points = _filter_points(self.points)
            self.centroid = three.points_centroid(self.points)
            self.center_object = center_object
        else:
            if object_scale == 'auto':
                raise ValueError("object_scale cannot be auto if odometry is not given.")

        if ref_points is not None:
            self.points = ref_points
            self.centroid = three.points_centroid(self.points)

        if object_scale == 'auto':
            object_scale = 1.2 / (1.0 * three.points_bounding_size(self.points).item())

        self.image_scale = image_scale
        self.object_scale = object_scale

    def load_intrinsics(self, path):
        with open(path / 'intrinsics.json', 'r') as f:
            intrinsics_json = json.load(f)
            intrinsic = three.intrinsic_to_3x4(
                torch.tensor(intrinsics_json['intrinsic_matrix']).reshape(3, 3).t()).float()

        return intrinsic

    def load_extrinsics(self, path):
        if self.odometry_type == 'open3d':
            trajectory_path = path / 'scene' / 'trajectory.log'
            extrinsics = read_open3d_trajectory(trajectory_path)
        elif self.odometry_type == 'kinectfusion':
            trajectory_path = path / 'scene_kf' / 'poses.txt'
            extrinsics = read_kinectfusion_trajectory(trajectory_path)
        else:
            raise ValueError(f"Unknown reg_type {self.odometry_type!r}")

        if self.use_registration:
            transform = self.load_registration(path)
            logger.debug("registering extrinsics", transform=transform.tolist())
            extrinsics = extrinsics @ three.inverse_transform(transform).unsqueeze(0).expand_as(
                extrinsics)

        return extrinsics

    def load_points(self, path):
        if self.odometry_type == 'open3d':
            pointcloud_path = path / 'scene' / 'integrated_cropped.ply'
        elif self.odometry_type == 'kinectfusion':
            pointcloud_path = path / 'scene_kf' / 'integrated_cropped.ply'
        else:
            raise ValueError(f"Unknown odometry_type {self.odometry_type!r}")

        data = PlyData.read(str(pointcloud_path))['vertex']
        points = torch.tensor(np.vstack((data['x'], data['y'], data['z'])).T, dtype=torch.float32)
        if self.use_registration:
            transform = self.load_registration(path)
            logger.debug("registering points", transform=transform.tolist())
            points = three.transform_coords(points.unsqueeze(0), transform.unsqueeze(0)).squeeze()

        return points

    def load_registration(self, path):
        if not self.use_registration:
            return torch.eye(4)

        reg_path = path / 'registration' / 'manual.json'
        if not reg_path.exists():
            reg_path = path / 'registration' / 'registration.json'

        if not reg_path.exists():
            return torch.eye(4)

        with open(reg_path, 'r') as f:
            logger.info("using registration", path=reg_path)
            reg_json = json.load(f)
            transform = torch.tensor(reg_json['transform'], dtype=torch.float32)
        return transform

    def __len__(self):
        return len(self.color_paths)

    def get_ids(self):
        return [p.stem for p in self.mask_paths]

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
        image = np.array(image, dtype=np.float32) / 1000
        return image

    def normalize_points(self, points):
        points = points.clone()
        points *= self.object_scale
        return points

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
        depth = torch.tensor(depth) * self.object_scale

        intrinsic = self.normalize_intrinsic(self.intrinsics[idx])

        if self.odometry_type is not None:
            extrinsic = self.normalize_extrinsic(self.extrinsics[idx])
        else:
            extrinsic = torch.eye(4)

        return {
            'color': color,
            'mask': mask,
            'depth': depth,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
        }
