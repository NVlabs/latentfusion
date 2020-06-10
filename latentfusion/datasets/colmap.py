from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset

from latentfusion import three


def _parse_image_meta(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if not l.startswith('#')]
    # Skip points lines.
    lines = lines[::2]
    image_dicts = []
    for line in lines:
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line.split(' ')
        rotation = torch.tensor((float(qw), float(qx), float(qy), float(qz)), dtype=torch.float32)
        translation = torch.tensor((float(tx), float(ty), float(tz)), dtype=torch.float32)
        image_dicts.append({
            'id': int(image_id),
            'name': name,
            'camera_id': int(camera_id),
            'rotation': rotation,
            'translation': translation,
        })
    return image_dicts


def _parse_points(path, max_error=2.0):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if not l.startswith('#')]
    points = []
    for line in lines:
        point_id, x, y, z, r, g, b, error = line.split(' ')[:8]
        error = float(error)
        if error <= max_error:
            points.append((float(x), float(y), float(z)))

    return torch.tensor(points, dtype=torch.float32)


def _parse_cameras(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if not l.startswith('#')]

    cameras = {}
    for line in lines:
        camera_id, model, width, height = line.split(' ')[:4]
        if model == 'PINHOLE':
            params = line.split(' ')[4:]
            fx, fy, cx, cy = params
            cameras[int(camera_id)] = {
                'id': int(camera_id),
                'width': int(width),
                'height': int(height),
                'model': model,
                'intrinsic': torch.tensor((
                    (float(fx), 0.0, float(cx), 0.0),
                    (0.0, float(fy), float(cy), 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                ))
            }
        else:
            raise ValueError(f'Camera model {model!r} is not supported yet.')

    return cameras


def _filter_points(points, n_estimators=100):
    clf = IsolationForest(n_estimators=n_estimators,
                          contamination=0.1)
    y = clf.fit_predict(points.numpy())
    y = torch.tensor(y)
    return points[(y > 0), :]


class ColmapDataset(Dataset):

    def __init__(self, path, image_scale=0.2, object_scale='auto',
                 mask_mode='grabcut'):
        self.path = Path(path)
        self.db_path = self.path / 'database.db'
        self.image_meta_path = self.path / 'images.txt'
        self.image_dir = self.path / 'color'
        self.mask_dir = self.path / 'mask'
        self.cameras_path = self.path / 'cameras.txt'
        self.points_path = self.path / 'points3D.txt'

        self.points = _parse_points(self.points_path)
        self.points = _filter_points(self.points)

        self.image_dicts = _parse_image_meta(self.image_meta_path)
        self.cameras = _parse_cameras(self.cameras_path)
        self.centroid = three.points_centroid(self.points)

        if object_scale == 'auto':
            object_scale = 1.0 / (1.0 * three.points_bounding_size(self.points).item())

        self.image_scale = image_scale
        self.object_scale = object_scale
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.image_dicts)

    def _load_image(self, path):
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
        return image

    def __getitem__(self, idx):
        meta = self.image_dicts[idx]
        image = self._load_image(self.image_dir / meta['name'])
        mask = self._load_mask(self.mask_dir / f"{meta['name']}.png")
        height, width = image.shape[:2]
        camera = self.cameras[meta['camera_id']]
        rot_mat = three.quaternion.quat_to_mat(meta['rotation'].unsqueeze(0))
        rot_mat = three.rotation_to_4x4(rot_mat).squeeze(0)
        translation = meta['translation']
        trans_mat = three.translation_to_4x4(translation.unsqueeze(0)).squeeze(0)
        extrinsic = trans_mat @ rot_mat
        extrinsic = three.translate_matrix(extrinsic, -self.centroid)
        extrinsic[:3, 3] *= self.object_scale
        intrinsic = camera['intrinsic'].clone()
        intrinsic[:2, :] *= self.image_scale

        return {
            'color': (torch.tensor(image).float() / 255.0).permute(2, 0, 1),
            'mask': torch.tensor(mask).bool(),
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
        }
