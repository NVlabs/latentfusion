import collections

import math
import torch
from tqdm.auto import tqdm

from latentfusion import three
from latentfusion import distances


def camera_rotation_dist(camera1, camera2):
    return three.quaternion.angular_distance(camera1.quaternion, camera2.quaternion)


def camera_translation_dist(camera1, camera2):
    return torch.norm(camera1.translation - camera2.translation, dim=-1)


def camera_metrics(camera_gt, camera_eval, points, scale_to_meters,
                   use_add=True,
                   use_add_sym=True,
                   use_add_s=True,
                   use_proj2d=True,
                   **kwargs):
    """
    Computes evaluation metrics for the given cameras.

    Args:
        camera_gt: the ground truth camera to evaluate against.
        camera_eval: the camera to evaluate.
        points: the points to compute point-based evaluation metrics with (ADD, ADD-S, etc.)
        scale_to_meters: the scale factor to be multiplied to convert the object scale to meters.
        **kwargs:

    Returns:
        A dictionary of metrics.
    """
    if len(camera_gt) > 1:
        return [camera_metrics(c1, c2, points, scale_to_meters)
                for c1, c2 in zip(tqdm(camera_gt), camera_eval)]

    camera_gt = camera_gt.clone().cpu()
    camera_eval = camera_eval.clone().cpu()
    rot_dist = camera_rotation_dist(camera_gt, camera_eval)
    trans_dist = camera_translation_dist(camera_gt, camera_eval) * scale_to_meters

    metrics = {
        'rotation_dist': rot_dist.squeeze().item(),
        'translation_dist': trans_dist.squeeze().item(),
    }
    if points is not None:
        if use_add:
            metrics['add'] = (compute_point_add(camera_gt.obj_to_cam, camera_eval.obj_to_cam, points)
                              * scale_to_meters)
        if use_add_s:
            metrics['add_s'] = (compute_point_add_s(camera_gt.obj_to_cam, camera_eval.obj_to_cam, points)
                                * scale_to_meters)
        if use_add_sym:
            metrics['add_sym'] = (compute_point_add_sym(camera_gt.obj_to_cam, camera_eval.obj_to_cam, points)
                                  * scale_to_meters)
        if use_proj2d:
            metrics['proj2d'] = compute_point_proj2d(camera_gt.obj_to_image, camera_eval.obj_to_image, points)

    return metrics


def compute_point_add_sym(extrinsic_gt, extrinsic_eval, points):
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    rot_z180 = three.quaternion.quat_to_mat(
        three.quaternion.from_axis_angle(z_axis, math.pi))
    rot_z180 = three.rotation_to_4x4(rot_z180)

    add_ident = compute_point_add(extrinsic_gt, extrinsic_eval, points)
    add_zsym = compute_point_add(extrinsic_gt @ rot_z180, extrinsic_eval, points)

    return torch.min(add_ident, add_zsym)


def compute_point_add(extrinsic_gt, extrinsic_eval, points):
    points_gt = three.transform_coords(points, extrinsic_gt)
    points_eval = three.transform_coords(points, extrinsic_eval)

    return torch.mean(torch.norm(points_gt - points_eval, dim=-1))


def compute_point_add_s(extrinsic_gt, extrinsic_eval, points):
    points_gt = three.transform_coords(points, extrinsic_gt)
    points_eval = three.transform_coords(points, extrinsic_eval)
    dists = best_distance(points_gt, points_eval)
    return torch.mean(dists)


@torch.jit.script
def best_distance(x1, x2, batch_size: int = 1000):
    best_dists = []
    num_batches = int(math.ceil(x1.shape[0] / batch_size))
    for i in range(num_batches):
        batch = x1[i * batch_size:(i+1) * batch_size]
        dists, _ = torch.cdist(batch, x2).min(dim=1)
        best_dists.append(dists)

    return torch.cat(best_dists, dim=0)


def compute_point_proj2d(proj_gt, proj_eval, points):
    points_gt = three.transform_coords(points, proj_gt)
    points_eval = three.transform_coords(points, proj_eval)

    return torch.mean(torch.norm(points_gt - points_eval, dim=-1))


def concat_camera_metrics(metrics_list):
    keys = metrics_list[0].keys()
    out = collections.defaultdict(list)
    for key in keys:
        for metrics in metrics_list:
            out[key].append(metrics[key])

    return out
