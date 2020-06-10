import math

import torch
from torch.nn import functional as F

from latentfusion import three
from latentfusion.distances import outer_distance
from latentfusion.three.batchview import b2bv, bv2b


def depth_to_warp_field(source_cam, target_cam, target_depth):
    """
    Computes a warp field that warps an image in the source view to the target view.

    The returned warp field is to be used with `F.grid_sample()`

    Args:
        source_cam: input view cameras
        target_cam: output view camera
        target_depth: output view depth

    Returns:
        Warp field compatible with `F.grid_sample()` (V_o, V_i, H, W, 2)

    """
    height, width = target_depth.shape[-2:]
    xx, yy, zz = target_cam.depth_camera_coords(target_cam.denormalize_depth(target_depth))
    cam_coords = three.grid_to_coords(torch.stack((xx, yy, zz), dim=-1))
    obj_coords = three.transform_coords(cam_coords, target_cam.cam_to_obj)

    obj_coords = bv2b(obj_coords[:, None, :, :]
                      .expand(-1, source_cam.length, -1, -1))
    obj_to_pix = bv2b(source_cam.obj_to_image[None, :, :, :]
                      .expand(target_cam.length, -1, -1, -1))

    source_pix_coords = three.transform_coords(obj_coords, obj_to_pix)

    source_viewport = source_cam.viewport.repeat(target_cam.length, 1)
    source_width = source_viewport[:, 2] - source_viewport[:, 0]
    source_height = source_viewport[:, 3] - source_viewport[:, 1]

    grid_coords = torch.stack((
        # x coordinates.
        ((source_pix_coords[..., 0] - source_viewport[:, 0, None]) / source_width[:, None]) * 2 - 1,
        # y coordinates.
        ((source_pix_coords[..., 1] - source_viewport[:, 1, None]) / source_height[:, None]) * 2 - 1,
    ), dim=-1)
    grid_coords = grid_coords.view(target_cam.length, source_cam.length, height, width, 2)
    return grid_coords


def reproject_views(image_in, depth_in, depth_out, camera_in, camera_out):
    """
    Reprojects pixels from the input view to the output view.

    Args:
        image_in: pixels to copy from (V_i, C, H, W)
        depth_out: target depth (V_o, C, H, W)
        camera_in: input view cameras
        camera_out: output view cameras

    Returns:
        Each input view image reprojected to output views (V_o, V_i, C, H, W)
        Each input view depth transformed and reprojected to output views (V_o, V_i, C, H, W)

    """
    grid = depth_to_warp_field(camera_in, camera_out, depth_out)

    # Expand and reshape to do batch operations:
    #   batch_dim = output_views
    #   view_dim = input_views
    image_in = bv2b(image_in
                    .unsqueeze(0)
                    .expand(camera_out.length, -1, -1, -1, -1))

    obj_coords_in = torch.stack(camera_in.depth_object_coords(depth_in), dim=-1)
    obj_coords_in = bv2b(obj_coords_in
                         .unsqueeze(0)
                         .expand(camera_out.length, -1, -1, -1, -1))

    # Repeat interleaved to expand to input view dimensions.
    camera_out = camera_out.repeat_interleave(camera_in.length)

    # Transform reprojected input coordinates to output view.
    cam_coords_in_tf = three.transform_coord_grid(obj_coords_in, camera_out.obj_to_cam)
    depth_in_tf = cam_coords_in_tf[..., 2].unsqueeze(1)
    depth_in_tf = camera_out.normalize_depth(depth_in_tf)

    grid = bv2b(grid)

    image_reproj = F.grid_sample(image_in, grid, mode='bilinear')
    depth_reproj = F.grid_sample(depth_in_tf, grid, mode='bilinear')
    return b2bv(image_reproj, camera_in.length), b2bv(depth_reproj, camera_in.length)


def reproject_views_batch(image_in, depth_in, depth_out, camera_in, camera_out):
    """
    Reprojects views but while supporting batch dimension.

    It just does a for loop.

    Args:
        image_in: pixels to copy from (B, V_i, C, H, W)
        depth_in: source depth (B, V_i, C, H, W)
        depth_out: target depth (B, V_o, C, H, W)
        camera_in: input view cameras
        camera_out: output view cameras

    Returns:
        Each input view image reprojected to output views (B, V_o, V_i, C, H, W)
        Each input view camera coordinates reprojected to output views (B, V_o, V_i, C, H, W)
        Camera distances between input and output views (B, V_o, V_i)
    """
    num_objects = image_in.shape[0]
    in_views = image_in.shape[1]
    out_views = depth_out.shape[1]
    image_reproj_list = []
    depth_reproj_list = []
    camera_dists_r = []
    camera_dists_t = []
    for i in range(num_objects):
        _cam_in = camera_in[i * in_views:(i + 1) * in_views]
        _cam_out = camera_out[i * out_views:(i + 1) * out_views]
        _cam_dists_r = three.quaternion.angular_distance(_cam_out.quaternion, _cam_in.quaternion, eps=1e-2) / math.pi
        _cam_dists_t = outer_distance(_cam_out.position, _cam_in.position, metric='cosine') / 2.0

        camera_dists_r.append(_cam_dists_r)
        camera_dists_t.append(_cam_dists_t)
        image_reproj, depth_reproj = reproject_views(image_in[i], depth_in[i], depth_out[i], _cam_in, _cam_out)
        image_reproj_list.append(image_reproj)
        depth_reproj_list.append(depth_reproj)

    return (
        torch.stack(image_reproj_list, dim=0),
        torch.stack(depth_reproj_list, dim=0),
        torch.stack(camera_dists_r, dim=0),
        torch.stack(camera_dists_t, dim=0),
    )


def render_latent_ibr(photographer, z_obj, camera_in, camera_out, image_in, p=0.5,
                      weight_type='cam_dist', eps=0.0001):
    """
    Renders the latent volume using image-based rendering by projecting the provided
    input images onto the output views.
    """
    batch_size = z_obj.shape[0]
    fake_in, _, _ = photographer.decode(z_obj, camera_in)
    fake_out, _, _ = photographer.decode(z_obj, camera_out)

    image_fake_ibr, image_fake_reproj = render_ibr(
        camera_in, camera_out, image_in, fake_in['depth'], fake_out['depth'], p, weight_type, eps)

    return image_fake_ibr, fake_out['depth'], fake_out['mask'], image_fake_reproj


def render_latent_ibr2(photographer, z_obj, camera_in, camera_out, image_in, p=0.5,
                       weight_type='cam_dist', return_latent=True, eps=0.0001,
                       apply_mask=False):
    """
    Renders the latent volume using image-based rendering by projecting the provided
    input images onto the output views.

    Alternative interface that returns y_out and z_out.
    """
    y_in, _, _ = photographer.decode(z_obj, camera_in, apply_mask=apply_mask)
    y_out, z_out, _ = photographer.decode(z_obj, camera_out, return_latent=return_latent,
                                          apply_mask=apply_mask)

    image_fake_ibr, image_fake_reproj = render_ibr(
        camera_in, camera_out, image_in, y_in['depth'], y_out['depth'], p, weight_type, eps)

    if apply_mask:
        y_out['color'] = image_fake_ibr * (y_out['mask'] > 0.5)
    else:
        y_out['color'] = image_fake_ibr

    return y_out, z_out


def render_ibr(camera_in, camera_out, image_in, depth_fake_in, depth_fake_out,
               p=0.5, weight_type='cam_dist', eps=1e-2):
    """
    Renders the latent volume using image-based rendering by projecting the provided
    input images onto the output views.
    """
    image_fake_reproj = []
    image_fake_ibrs = []
    for i in range(image_in.shape[0]):
        num_in_views = camera_in.length // image_in.shape[0]
        num_out_views = camera_out.length // image_in.shape[0]
        _cam_in = camera_in[i * num_in_views:(i + 1) * num_in_views]
        _cam_out = camera_out[i * num_out_views:(i + 1) * num_out_views]
        image_reproj, depth_reproj = reproject_views(image_in[i], depth_fake_in[i], depth_fake_out[i],
                                                     _cam_in, _cam_out)
        image_fake_reproj.append(image_reproj)
        if weight_type == 'cam_dist':
            cam_dists = outer_distance(_cam_out.position, _cam_in.position, metric='cosine', eps=eps) / 2.0
            cam_weights = 1.0 / ((cam_dists.unsqueeze(-1).unsqueeze(-1)) ** p).clamp(min=eps)
            cam_weights = torch.softmax(cam_weights, dim=1)
        elif weight_type == 'cam_angle':
            cam_dists = three.quaternion.angular_distance(_cam_out.quaternion, _cam_in.quaternion) / math.pi
            cam_weights = 1.0 / ((cam_dists.unsqueeze(-1).unsqueeze(-1)) ** p).clamp(min=eps)
            cam_weights = torch.softmax(cam_weights, dim=1)
        elif weight_type == 'cam_hybrid':
            cam_dists_t = outer_distance(_cam_out.position, _cam_in.position, metric='cosine') / 2.0
            cam_dists_r = three.quaternion.angular_distance(_cam_out.quaternion, _cam_in.quaternion)
            cam_dists_r = (cam_dists_r / (math.pi / 8)).clamp(0.0, 1.0)
            cam_dists = 1.0 - (1.0 - cam_dists_t) * (1.0 - cam_dists_r)
            cam_weights = 1.0 / ((cam_dists.unsqueeze(-1).unsqueeze(-1)) ** p).clamp(min=eps)
            cam_weights = torch.softmax(cam_weights, dim=1)
        elif weight_type == 'depth':
            depth_diff = (depth_reproj - depth_fake_out[i].unsqueeze(1).expand_as(depth_reproj)).abs()
            cam_weights = torch.softmax(1.0 / ((depth_diff/depth_diff.max()) ** p + eps), dim=1).squeeze(2)
        else:
            raise ValueError(f'Unknown weight_type {weight_type}')
        image_fake_ibr = (cam_weights.unsqueeze(2) * image_reproj).sum(dim=1)
        image_fake_ibrs.append(image_fake_ibr)
    image_fake_ibr = torch.stack(image_fake_ibrs, dim=0)
    image_fake_reproj = torch.stack(image_fake_reproj, dim=0)

    return image_fake_ibr, image_fake_reproj


def blend_logits(logits, image_reproj):
    blend_weights = torch.softmax(logits, dim=1).unsqueeze(2)
    image_fake = (blend_weights * image_reproj).sum(dim=1)
    return image_fake, blend_weights


def warp_blend_logits(logits, image_reproj, flow_size):
    device = image_reproj.device
    num_input_views = image_reproj.shape[1]
    height, width = image_reproj.shape[-2:]
    blend_logits, flow_x_logits, flow_y_logits = torch.split(logits, num_input_views, dim=1)
    blend_weights = torch.softmax(blend_logits, dim=1).unsqueeze(2)
    flow_dx = flow_size / width * torch.tanh(flow_x_logits)
    flow_dy = flow_size / height * torch.tanh(flow_y_logits)
    flow_y, flow_x = torch.meshgrid([torch.linspace(-1, 1, height, device=device),
                                     torch.linspace(-1, 1, width, device=device)])
    flow_x = flow_x[None, None, :, :].expand_as(flow_dx) + flow_dx
    flow_y = flow_y[None, None, :, :].expand_as(flow_dy) + flow_dy
    flow_grid = torch.stack((flow_x, flow_y), dim=-1).clamp(-1, 1)

    image_fake = F.grid_sample(bv2b(image_reproj), bv2b(flow_grid), mode='bilinear')
    image_fake = b2bv(image_fake, num_input_views)
    image_fake = (blend_weights * image_fake).sum(dim=1)

    return image_fake, blend_weights, flow_dx, flow_dy
