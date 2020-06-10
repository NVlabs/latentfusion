import math

import torch
from torch import nn

from latentfusion import three
from latentfusion.losses import PerceptualLoss
import torchvision.models

from latentfusion.modules.geometry import Camera


def perturb_camera(camera, translation_std, quaternion_std):
    camera = camera.clone()
    camera.translation.data += torch.randn_like(camera.translation) * translation_std
    camera.log_quaternion.data += torch.randn_like(camera.log_quaternion) * quaternion_std
    return camera


def get_perceptual_loss():
    vgg = torchvision.models.vgg16(pretrained=True).eval()
    perceptual_loss_base = vgg.features
    layers = ['3', '8', '15', '22', '27']
    layer_weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1]
    return PerceptualLoss(perceptual_loss_base, layers, layer_weights, reduction=None)


def sample_cameras_with_estimate(n, camera_est, translation_std=0.0,
                                 hemisphere=False, upright=False) -> Camera:
    device = camera_est.device
    intrinsic = camera_est.intrinsic.expand(n, -1, -1)
    translation = camera_est.translation.expand(n, -1)
    translation = translation + torch.randn_like(translation) * translation_std
    # quaternion = three.orientation.disk_sample_quats(n, min_angle=min_angle)
    # quaternion = three.orientation.evenly_distributed_quats(n)
    quaternion = three.orientation.evenly_distributed_quats(
        n, hemisphere=hemisphere, upright=upright)
    extrinsic = three.to_extrinsic_matrix(translation.cpu(), quaternion).to(device)
    viewport = camera_est.viewport.expand(n, -1)

    return Camera(intrinsic, extrinsic,
                  camera_est.z_span,
                  width=camera_est.width,
                  height=camera_est.height,
                  viewport=viewport)


def parameterize_camera(camera,
                        optimize_rotation=True,
                        optimize_translation=True,
                        optimize_viewport=False):
    camera_opt = camera.clone()

    if optimize_rotation:
        camera_opt.log_quaternion = nn.Parameter(camera_opt.log_quaternion)

    if optimize_translation:
        camera_opt.translation = nn.Parameter(camera_opt.translation)

    if optimize_viewport:
        camera_opt.viewport = nn.Parameter(camera_opt.viewport)

    return camera_opt


def deparameterize_camera(camera):
    camera = camera.clone()
    camera.log_quaternion = camera.log_quaternion.detach()
    camera.translation = camera.translation.detach()
    camera.viewport = camera.viewport.detach()
    return camera


def flip_camera(camera, axis=(0.0, 0.0, 1.0)):
    z_axis = torch.tensor([axis], dtype=torch.float32,
                          device=camera.device).expand(len(camera), -1)
    flip_quat = three.quaternion.from_axis_angle(z_axis, math.pi)
    return camera.clone().rotate(flip_quat)


def zero_invalid_pixels(tensor, invalid_mask):
    """
    Zeros out the loss for places with no depth but positive mask.
    This makes the loss more robust to depth sensor errors.

    Args:
        tensor: the loss the robustify.
        target_depth: the target depth.
        target_mask: the target mask.

    Returns:
        the loss but with invalid pixels ignored.
    """
    valid_mask = ~invalid_mask
    tensor = tensor * valid_mask.float()
    return tensor


def iou_loss(input_mask, target_mask, eps=1e-4):
    intersection = torch.sum(input_mask * target_mask, dim=(1, 2, 3))
    union = (torch.sum(input_mask, dim=(1, 2, 3))
             + torch.sum(target_mask, dim=(1, 2, 3))
             - intersection)
    # iou = intersection / union.clamp(min=eps)
    # return 1.0 - iou
    # return torch.log(1/iou)

    return torch.log(union.clamp(min=eps)) - torch.log(intersection.clamp(min=eps))


def reduce_loss_mask(loss, mask, eps=1e-4):
    if len(loss.shape) == 4:
        loss = loss.squeeze(1)
    if len(mask.shape) == 4:
        mask = mask.squeeze(1)

    return (loss * mask).sum(dim=(-2, -1)).clamp(min=eps/10) / mask.sum(dim=(-2, -1)).clamp(min=eps)


def mask_centroid(mask):
    device = mask.device
    height, width = mask.shape[-2:]
    yy, xx = torch.meshgrid([torch.arange(0, height, device=device, dtype=torch.float32),
                             torch.arange(0, width, device=device, dtype=torch.float32)])
    centroid = torch.stack((
        (mask * yy).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)),
        (mask * xx).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)),
    ), dim=-1)
    return centroid


def mask_contour(mask):
    mask = (mask > 0.5).float()
    yg = (mask[..., 1:, :] - mask[..., :-1, :]).abs()
    xg = (mask[..., :, 1:] - mask[..., :, :-1]).abs()
    contour = (yg[..., 1:, :-2] + xg[..., :-2, 1:]).abs() > 0
    return contour


def shape_loss(input_mask, target_mask):
    if len(input_mask.shape) == 4:
        input_mask = input_mask.squeeze(1)
    if len(target_mask.shape) == 4:
        target_mask = target_mask.squeeze(1)

    if target_mask.shape[0] == 1:
        target_mask = target_mask.expand_as(input_mask)
    device = input_mask.device
    n = input_mask.shape[0]
    height, width = input_mask.shape[-2:]

    input_centroid = mask_centroid(input_mask)
    target_centroid = mask_centroid(target_mask)

    yy, xx = torch.meshgrid([torch.arange(0, height, device=device, dtype=torch.float32),
                             torch.arange(0, width, device=device, dtype=torch.float32)])
    yx_coords = torch.stack((yy, xx), dim=0).unsqueeze(0).expand(n, 2, -1, -1)

    union_mask = ((input_mask + target_mask) > 0).float()
    input_dtc = torch.norm((yx_coords - input_centroid[:, :, None, None]), dim=1)
    target_dtc = torch.norm((yx_coords - target_centroid[:, :, None, None]), dim=1) * target_mask
    target_maxdist, _ = target_dtc.view(n, -1).max(dim=1)
    input_dtc = input_dtc / target_maxdist[:, None, None]
    target_dtc = target_dtc / target_maxdist[:, None, None]

    # input_dtc_target = torch.norm((yx_coords - target_centroid[:, :, None, None]), dim=1) * input_mask
    # loss = (input_dtc - target_dtc).abs() * input_mask
    loss = (input_dtc - target_dtc).abs() * input_mask

    return loss


def contour_loss(input_mask, target_mask):
    input_contour = mask_contour(input_mask).float()
    target_contour = mask_contour(target_mask).float()

    return (target_contour.sum(dim=(1, 2, 3))
            - input_contour.sum(dim=(1, 2, 3))).abs()

    # losses = []
    # for ic, tc in zip(input_contour, target_contour):
    #     closest_dists, _ = outer_distance(ic.nonzero().float(),
    #                                       tc.nonzero().float(),
    #                                       metric='euclidean', p=2).min(dim=1)
    #     losses.append(closest_dists.mean())
    #
    # return torch.stack(losses, dim=0)



