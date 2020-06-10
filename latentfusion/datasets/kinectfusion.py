import json
from pathlib import Path

import imageio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from latentfusion import three
import numpy as np


def load_points_file(path):
    with open(path, 'r') as f:
        text = f.read()
    lines = [s.strip() for s in text.split('\n') if s.strip() != '']
    points = torch.tensor([[float(v) for v in line.split(' ')] for line in lines])
    return points


def load_poses_file(poses_path):
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


def load_poses(poses_path, points_path):
    points = load_points_file(points_path)
    # centroid = torch.mean(points, dim=0)
    centroid = three.points_centroid(points)

    rel_poses = load_poses_file(poses_path)
    rel_poses[0][:3, 3] = centroid

    canon_pose = rel_poses[0]
    abs_poses = [rel_poses[0]]
    for rel_pose in rel_poses[1:]:
        abs_poses.append(rel_pose @ canon_pose)
    abs_poses = torch.stack(abs_poses, dim=0)

    return abs_poses


def load_depth(path):
    depth = imageio.imread(path)
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    depth = torch.tensor(depth.astype(np.float32) / 1000.0)
    return depth


def load_intrinsics(path):
    with open(path, 'r') as f:
        text = json.load(f)

    intrinsic = torch.tensor(text).reshape(3, 3)
    zeros = torch.zeros(3, 1)
    intrinsic = torch.cat((intrinsic, zeros), dim=1)
    return intrinsic


class KinectFusionDataset(Dataset):

    def __init__(self, path, stride=1):
        self.path = Path(path)
        self.image_dir = self.path / 'images'
        self.poses_path = self.path / 'poses.txt'
        self.points_path = self.path / 'points.xyz'
        self.intrinsics_path = self.path / 'intrinsics.json'
        stride = stride

        self.extrinsics = load_poses(self.poses_path, self.points_path)
        self.intrinsics = (load_intrinsics(self.intrinsics_path)
                           .unsqueeze(0)
                           .expand(self.extrinsics.shape[0], -1, -1))

        self.color_paths = [*sorted(self.image_dir.glob('*-color.png')),
                            *sorted(self.image_dir.glob('*-rgb.png'))]
        self.depth_paths = sorted(self.image_dir.glob('*-depth.png'))

        if stride > 1:
            self.color_paths = self.color_paths[::stride]
            self.depth_paths = self.depth_paths[::stride]
            self.extrinsics = self.extrinsics[::stride]
            self.intrinsics = self.intrinsics[::stride]

        self.color_tform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return min(len(self.color_paths), self.extrinsics.shape[0])

    def __getitem__(self, idx):
        extrinsic = self.extrinsics[idx]
        intrinsic = self.intrinsics[idx]
        color_path = self.color_paths[idx]
        depth_path = self.depth_paths[idx]

        color_im = self.color_tform(imageio.imread(color_path))
        depth_im = load_depth(depth_path)

        dist = torch.norm(extrinsic[:3, 3])
        target_dist = 3.0
        scale = target_dist / dist
        extrinsic[:3, 3] *= scale
        depth_im *= scale

        return {
            'color': color_im,
            'depth': depth_im,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
        }
