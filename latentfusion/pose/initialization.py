import torch
from skimage import morphology

from latentfusion import three
from latentfusion.modules.geometry import Camera


@torch.jit.script
def _masks_to_viewports(masks, pad: float = 10):
    viewports = []
    padding = torch.tensor([-pad, -pad, pad, pad], dtype=torch.float32, device=masks.device)

    for mask in masks:
        coords = torch.nonzero(mask.squeeze()).float()
        xmin = coords[:, 1].min()
        ymin = coords[:, 0].min()
        xmax = coords[:, 1].max()
        ymax = coords[:, 0].max()
        viewport = torch.stack([xmin, ymin, xmax, ymax])
        viewport = viewport + padding
        viewports.append(viewport)

    return torch.stack(viewports, dim=0)


@torch.jit.script
def _masks_to_centroids(masks):
    viewports = _masks_to_viewports(masks, 0.0)
    cu = (viewports[:, 2] + viewports[:, 0]) / 2.0
    cv = (viewports[:, 3] + viewports[:, 1]) / 2.0

    return torch.stack((cu, cv), dim=-1)


def _erode_mask(mask, size=5):
    device = mask.device
    eroded = mask.cpu().squeeze(0).numpy()
    eroded = morphology.binary_erosion(eroded, selem=morphology.disk(size))
    eroded = torch.tensor(eroded, device=device, dtype=torch.bool).unsqueeze(0)
    if len(eroded) < 10:
        return mask
    return eroded


def _reject_outliers(data, m=1.5):
    mask = torch.abs(data - torch.median(data)) < m * torch.std(data)
    num_rejected = (~mask).sum().item()
    return data[mask], num_rejected


def _reject_outliers_mad(data, m=2.0):
    median = data.median()
    mad = torch.median(torch.abs(data - median))
    mask = torch.abs(data - median) / mad < m
    num_rejected = (~mask).sum().item()
    return data[mask], num_rejected


def _estimate_camera_dist(depth, mask):
    num_batch = depth.shape[0]
    zs = torch.zeros(num_batch, device=depth.device)
    mask = mask.bool()
    for i in range(num_batch):
        # zs[i] = depth[i][mask[i]].median()
        _mask = _erode_mask(mask[i], size=3)
        depth_vals = depth[i][_mask & (depth[i] > 0.0)]
        depth_vals, num_rejected = _reject_outliers_mad(depth_vals, m=3.0)
        _min = depth_vals.min()
        _max = depth_vals.max()
        zs[i] = (_min + _max) / 2.0

    return zs


def estimate_translation(depth, mask, intrinsic):
    z_cam = _estimate_camera_dist(depth, mask)
    centroid_uv = _masks_to_centroids(mask)

    u0 = intrinsic[..., 0, 2]
    v0 = intrinsic[..., 1, 2]
    fu = intrinsic[..., 0, 0]
    fv = intrinsic[..., 1, 1]
    x_cam = (centroid_uv[:, 0] - u0) / fu * z_cam
    y_cam = (centroid_uv[:, 1] - v0) / fv * z_cam

    return x_cam, y_cam, z_cam


def estimate_initial_pose(depth, mask, intrinsic, width, height) -> Camera:
    """Estimate the initial pose based on depth."""
    translation = torch.stack(estimate_translation(depth, mask, intrinsic), dim=-1)
    rotation = three.quaternion.identity(intrinsic.shape[0], intrinsic.device)
    extrinsic = three.to_extrinsic_matrix(translation, rotation)

    camera = Camera(intrinsic, extrinsic, height=height, width=width)

    return camera