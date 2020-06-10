import json

import torch


def parse_camera_intrinsics(d):
    return torch.tensor([
        [d['fx'], 0.0, d['cx'], 0.0],
        [0.0, d['fy'], d['cy'], 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float32)


def load_camera_intrinsics(path):
    with open(path, 'r') as f:
        d = json.load(f)

    return parse_camera_intrinsics(d)

