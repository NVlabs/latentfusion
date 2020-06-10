import functools
import math
import random

import torch

from latentfusion import three
from latentfusion.augment import gan_normalize
from latentfusion.modules.geometry import Camera
from latentfusion.three import bv2b, b2bv, quaternion


def optimal_camera_dist(focal_length, size, radius, slack=1.5):
    # theta = fov/2.
    theta = math.atan2(size / 2.0, focal_length)
    r = radius
    # Triangle lengths.
    h = radius * math.cos(theta)
    x = h / math.sin(theta)
    # Law of cosines.
    d = math.sqrt(x ** 2 + r ** 2 - 2 * x * r * math.cos(math.pi / 2.0 - theta))
    return d + slack


def repeat_tensor_as(tensor, shape_ref, num_shape_dims=3):
    shape_dims = shape_ref.shape[-num_shape_dims:]
    num_dims = len(shape_ref.shape)
    num_batch_dims = num_dims - num_shape_dims - 1  # tensor already has channel dim.
    for _ in range(num_batch_dims):
        tensor = tensor.unsqueeze(0)
    tensor = tensor.expand(*shape_ref.shape[:num_batch_dims], -1, *shape_dims)
    return tensor


def get_normalized_voxel_coords(tensor):
    depth, height, width = tensor.shape[-3:]
    z, y, x = torch.meshgrid([
        torch.linspace(-1.0, 1.0, depth, device=tensor.device),
        torch.linspace(-1.0, 1.0, height, device=tensor.device),
        torch.linspace(-1.0, 1.0, width, device=tensor.device),
    ])
    coords = torch.stack((z, y, x), dim=-4)
    return repeat_tensor_as(coords, tensor, num_shape_dims=3)


def get_normalized_pixel_coords(tensor):
    height, width = tensor.shape[-2:]
    y, x = torch.meshgrid([
        torch.linspace(-1.0, 1.0, height, device=tensor.device),
        torch.linspace(-1.0, 1.0, width, device=tensor.device),
    ])
    coords = torch.stack((y, x), dim=-3)
    return repeat_tensor_as(coords, tensor, num_shape_dims=2)


def get_normalized_voxel_depth(tensor):
    B, C, D, H, W = tensor.size()
    z_coords = (torch.linspace(-1.0, 1.0, D, device=tensor.device)
                .view(1, 1, D, 1, 1)
                .expand(B, 1, D, H, W))
    return z_coords


def mask_normalized_depth(depth, mask):
    return ((depth / 2.0 + 0.5) * mask) * 2.0 - 1.0


def _process_batch(batch, rotation, cube_size, camera_dist, input_size, device, is_gt):
    # Collapse viewpoint dimension to batch dimension:
    #   (B, V, C, H, W) => (B*V, C, H, W)
    batch_size = batch['mask'].shape[0]
    extrinsic = bv2b(batch['extrinsic'].to(device))
    intrinsic = bv2b(batch['intrinsic'].to(device))
    mask = bv2b(batch['mask'].unsqueeze(2).float().to(device))
    image = bv2b(gan_normalize(batch['render'].to(device)))
    if 'depth' in batch:
        depth = bv2b(batch['depth'].unsqueeze(2).to(device))
    else:
        depth = None

    # Project image features onto canonical volume.
    camera = Camera(intrinsic, extrinsic, z_span=cube_size/2.0,
                    height=image.size(2), width=image.size(3)).to(device)
    if rotation is not None:
        camera.rotate(rotation.expand(camera.length, -1))
        # translation = three.uniform(3, -cube_size/16, cube_size/16).view(1, 3).expand(camera.length, -1).to(device)
        # camera.translate(translation)
    _zoom = functools.partial(camera.zoom, target_size=input_size, target_dist=camera_dist)

    out = dict()
    # Zoom camera to canonical distance and size.
    out['image'], out['camera'] = _zoom(image, scale_mode='bilinear')
    out['mask'] = _zoom(mask, scale_mode='nearest')[0]
    if depth is not None:
        out['depth'] = camera.normalize_depth(_zoom(depth, scale_mode='nearest')[0])

    if is_gt:
        out['image'] = out['image'] * out['mask']
        out['depth'] = mask_normalized_depth(out['depth'], out['mask'])

    for k in {'image', 'depth', 'mask'}:
        out[k] = b2bv(out[k], batch_size=batch_size)

    return out


def process_batch(batch, cube_size, camera_dist, input_size, device,
                  random_orientation=True):
    """
    Processes the batch as follows:
        - Transfers data to the current device.
        - Collapses the viewpoint dimension into the batch dimension.
        - Zooms images to our canonical camera.
        - Applies random rotation to the camera extrinsics if applicable.
    """

    if random_orientation:
        # Apply same random rotation to all cameras.
        rand_rot = quaternion.random(1).to(device)
    else:
        rand_rot = None

    processed_batch = {}
    for k, v in batch.items():
        processed_batch[k] = _process_batch(v, rand_rot, cube_size, camera_dist, input_size, device,
                                            is_gt='gt' in k)

    return processed_batch