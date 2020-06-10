import time

import abc
import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F

from latentfusion import three, torchutils
from latentfusion.modules import PixelNorm
from latentfusion.modules.equalized import EqualizedConv2d
from latentfusion.three import quaternion as quat
from latentfusion.three.batchview import bv2b, b2bv


def _grid_sample(tensor, grid, **kwargs):
    return F.grid_sample(tensor.float(), grid.float(), **kwargs)


@torch.jit.script
def bbox_to_grid(bbox, in_size, out_size):
    h = in_size[0]
    w = in_size[1]
    xmin = bbox[0].item()
    ymin = bbox[1].item()
    xmax = bbox[2].item()
    ymax = bbox[3].item()
    grid_y, grid_x = torch.meshgrid([
        torch.linspace(ymin / h, ymax / h, out_size[0], device=bbox.device) * 2 - 1,
        torch.linspace(xmin / w, xmax / w, out_size[1], device=bbox.device) * 2 - 1,
    ])

    return torch.stack((grid_x, grid_y), dim=-1)


@torch.jit.script
def bboxes_to_grid(boxes, in_size, out_size):
    grids = torch.zeros(boxes.size(0), out_size[1], out_size[0], 2, device=boxes.device)
    for i in range(boxes.size(0)):
        box = boxes[i]
        grids[i, :, :, :] = bbox_to_grid(box, in_size, out_size)

    return grids


class Camera(nn.Module, torchutils.Scatterable):

    def __init__(self, intrinsic, extrinsic, z_span=0.5,
                 viewport=None, width=640, height=480,
                 log_quaternion=None,
                 translation=None):
        super().__init__()
        device = intrinsic.device

        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic.unsqueeze(0)
        if intrinsic.shape[1] == 3 and intrinsic.shape[2] == 3:
            intrinsic = three.intrinsic_to_3x4(intrinsic)

        if viewport is None:
            viewport = (torch.tensor((0, 0, width, height), dtype=torch.float32, device=device)
                        .view(1, 4)
                        .expand(intrinsic.shape[0], -1))

        if len(viewport.shape) == 1:
            viewport = viewport.unsqueeze(0)

        self.width = width
        self.height = height

        # The span the 3D camera frustum will cover.
        # The near and far values are computed using this span.
        self.z_span = z_span

        # The viewport is represented as a bounding box: (xmin, ymin, xmax, ymax).
        self.register_buffer('viewport', viewport)
        # The intrinsic matrix is a 4x3 matrix.
        self.register_buffer('intrinsic', intrinsic)

        if extrinsic is not None:
            if len(extrinsic.shape) == 2:
                extrinsic = extrinsic.unsqueeze(0)
            rotation, translation = three.decompose(extrinsic)
            quaternion = quat.mat_to_quat(rotation[:, :3, :3].contiguous())
            translation = translation[:, :3, -1].contiguous()
            # The real part of the log of a unit quaternion is always 0.
            log_quaternion = three.quaternion.qlog(quaternion)[:, 1:]

        if translation is None:
            raise ValueError("translation must be given through extrinsic or explicitly.")
        elif len(translation.shape) == 1:
            translation = translation.unsqueeze(0)

        if log_quaternion is None:
            raise ValueError("log_quaternion must be given through extrinsic or explicitly.")
        elif len(log_quaternion.shape) == 1:
            log_quaternion = log_quaternion.unsqueeze(0)

        self.register_buffer('log_quaternion', log_quaternion)

        # The rotation as a unit quaternion in the (w, x, y, z) convention.
        # self.register_buffer('quaternion', quaternion)
        # The translation as a 3-vector (x, y, z).
        self.register_buffer('translation', translation)

    @property
    def quaternion(self):
        return three.quaternion.qexp(self.log_quaternion)

    @quaternion.setter
    def quaternion(self, q):
        self.log_quaternion = three.quaternion.qlog(q)[:, 1:]

    def to_kwargs(self):
        return {
            'intrinsic': self.intrinsic,
            'extrinsic': self.extrinsic,
            'z_span': self.z_span,
            'viewport': self.viewport,
            'height': self.height,
            'width': self.width,
        }

    @classmethod
    def from_kwargs(cls, kwargs):
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                _kwargs[k] = torch.tensor(v, dtype=torch.float32)
            else:
                _kwargs[k] = v
        return cls(**_kwargs)

    @property
    def extrinsic(self):
        return self.translation_matrix @ self.rotation_matrix

    @extrinsic.setter
    def extrinsic(self, extrinsic):
        rotation, translation = three.decompose(extrinsic)
        quaternion = quat.mat_to_quat(rotation[:, :3, :3].contiguous())
        translation = translation[:, :3, -1].contiguous()
        log_quaternion = three.quaternion.qlog(quaternion)[:, 1:]
        self.log_quaternion.copy_(log_quaternion)
        self.translation.copy_(translation)

    @property
    def rotation_matrix(self):
        q = quat.normalize(self.quaternion)
        R = quat.quat_to_mat(q)
        R = F.pad(R, (0, 1, 0, 1))
        R[:, -1, -1] = 1.0
        return R

    @property
    def translation_matrix(self):
        eye = torch.eye(4, device=self.translation.device)
        return F.pad(self.translation.unsqueeze(2), (3, 0, 0, 1)) + eye

    @property
    def inv_translation_matrix(self):
        eye = torch.eye(4, device=self.translation.device)
        return F.pad(-self.translation.unsqueeze(2), (3, 0, 0, 1)) + eye

    @property
    def device(self):
        return self.intrinsic.device

    @property
    def viewport_height(self):
        return self.viewport[:, 3] - self.viewport[:, 1]

    @property
    def viewport_width(self):
        return self.viewport[:, 2] - self.viewport[:, 0]

    @property
    def viewport_centroid(self):
        cx = (self.viewport[:, 2] + self.viewport[:, 0]) / 2.0
        cy = (self.viewport[:, 3] + self.viewport[:, 1]) / 2.0
        return torch.stack((cx, cy), dim=-1)

    @property
    def u0(self):
        return self.intrinsic[:, 0, 2]

    @property
    def v0(self):
        return self.intrinsic[:, 1, 2]

    @property
    def fu(self):
        return self.intrinsic[:, 0, 0]

    @property
    def fv(self):
        return self.intrinsic[:, 1, 1]

    @property
    def fov_u(self):
        return torch.atan2(self.fu, self.viewport_width / 2.0)

    @property
    def fov_v(self):
        return torch.atan2(self.fv, self.viewport_height / 2.0)

    @property
    def obj_to_cam(self):
        return self.translation_matrix @ self.rotation_matrix

    @property
    def cam_to_obj(self):
        return self.rotation_matrix.transpose(2, 1) @ self.inv_translation_matrix

    @property
    def obj_to_image(self):
        return self.intrinsic @ self.obj_to_cam

    @property
    def position(self):
        # C = (-R^T) t
        position = -self.rotation_matrix[:, :3, :3].transpose(2, 1) @ self.translation_matrix[:, :3, 3, None]
        position = position.squeeze(-1)
        return position

    @property
    def direction(self):
        position = self.posiiton
        return position / torch.norm(position, dim=1, keepdim=True)

    @property
    def length(self):
        return self.intrinsic.size(0)

    def rotate(self, q):
        self.quaternion = quat.qmul(self.quaternion, q)
        return self

    def translate(self, offset):
        offset, unsqueezed = three.ensure_batch_dim(offset, 1)
        if offset.shape[0] == 1:
            offset = offset.expand_as(self.position)
        position = three.homogenize(self.position + offset).unsqueeze(-1)
        translation = -(self.rotation_matrix @ position).squeeze(2)
        translation = three.dehomogenize(translation)
        self.translation = translation
        return self

    @property
    def znear(self):
        return self.translation_matrix[:, 2, -1] - self.z_span

    @property
    def zfar(self):
        return self.translation_matrix[:, 2, -1] + self.z_span

    @property
    def z_bounds(self):
        return self.znear, self.zfar

    def uncrop(self, image=None, scale_mode='nearest', scale=1.0):
        new_cam = Camera(self.intrinsic, None, self.z_span,
                         width=self.width,
                         height=self.height,
                         log_quaternion=self.log_quaternion,
                         translation=self.translation)

        if image is None:
            return new_cam

        width = int(self.width * scale)
        height = int(self.height * scale)
        viewport = self.viewport * scale
        viewport_height = self.viewport_height * scale
        viewport_width = self.viewport_width * scale

        yy, xx = torch.meshgrid([torch.arange(0, height, device=self.device, dtype=torch.float32),
                                 torch.arange(0, width, device=self.device, dtype=torch.float32)])
        yy = yy.unsqueeze(0).expand(image.shape[0], -1, -1)
        xx = xx.unsqueeze(0).expand(image.shape[0], -1, -1)
        yy = (yy - viewport[:, 1, None, None]) / viewport_height[:, None, None] * 2 - 1
        xx = (xx - viewport[:, 0, None, None]) / viewport_width[:, None, None] * 2 - 1
        grid = torch.stack((xx, yy), dim=-1)

        return _grid_sample(image, grid, mode=scale_mode, padding_mode='border'), new_cam

    def crop_to_viewport(self, image, target_size, scale_mode='nearest'):
        in_size = torch.tensor((self.height, self.width), device=self.device)
        out_size = torch.tensor((target_size, target_size), device=self.device)

        grid = bboxes_to_grid(self.viewport, in_size, out_size)
        return _grid_sample(image, grid, mode=scale_mode)

    def zoom(self, image, target_size, target_dist, target_fu=None, target_fv=None,
             image_scale=1.0, zs=None, centroid_uvs=None, scale_mode='bilinear'):
        """
        Transforms the image as if the image were taken with the given target parameters.

        Args:
            image: the image to transform
            target_size: the target image size
            target_dist: the target distance from the origin
            target_fu: the target horizontal focal length
            target_fv: the target vertical focal length

        Returns:
            The transformed image and the transformed camera
        """
        K = self.intrinsic
        T = self.translation_matrix
        if zs is None:
            zs = T[:, 2, -1]
        fu = K[:, 0, 0]
        fv = K[:, 1, 1]
        if target_fu is None:
            target_fu = fu
        if target_fv is None:
            target_fv = fv

        bbox_u = target_dist * (1.0 / zs) / fu * target_fu * target_size / self.width * image_scale
        bbox_v = target_dist * (1.0 / zs) / fv * target_fv * target_size / self.height * image_scale

        if centroid_uvs is None:
            origin = (torch.tensor((0, 0, 0, 1.0), device=self.device)
                      .view(1, -1, 1)
                      .expand(self.length, -1, -1))
            uvs = K @ self.obj_to_cam @ origin
            uvs = (uvs[:, :2] / uvs[:, 2, None]).transpose(2, 1).squeeze(1)
            centroid_uvs = uvs.clone().float()

        center_u = centroid_uvs[:, 0] / self.width
        center_v = centroid_uvs[:, 1] / self.height

        boxes = torch.zeros(centroid_uvs.size(0), 4, device=self.device)
        boxes[:, 0] = (center_u - bbox_u / 2) * float(self.width)
        boxes[:, 1] = (center_v - bbox_v / 2) * float(self.height)
        boxes[:, 2] = (center_u + bbox_u / 2) * float(self.width)
        boxes[:, 3] = (center_v + bbox_v / 2) * float(self.height)

        camera_new = Camera(self.intrinsic, None, self.z_span, viewport=boxes,
                            log_quaternion=self.log_quaternion,
                            translation=self.translation,
                            width=self.width,
                            height=self.height)

        if image is None:
            return camera_new

        in_size = torch.tensor((self.height, self.width), device=self.device)
        out_size = torch.tensor((target_size, target_size), device=self.device)
        grids = bboxes_to_grid(boxes, in_size, out_size)
        image_new = _grid_sample(image, grids, mode=scale_mode)

        return image_new, camera_new

    def __getitem__(self, item):
        """
        Splits the camera into sections. Similar to `torch.split`.
        """
        intrinsics = self.intrinsic[item]
        viewports = self.viewport[item]

        return Camera(intrinsics, None, self.z_span, viewports,
                      log_quaternion=self.log_quaternion[item],
                      translation=self.translation[item],
                      width=self.width,
                      height=self.height)

    def __setitem__(self, item, value):
        self.intrinsic[item] = value.intrinsic
        self.viewport[item] = value.viewport
        self.log_quaternion[item] = value.log_quaternion
        self.translation[item] = value.translation

    def __len__(self):
        return self.length

    def __iter__(self):
        cameras = [self[i] for i in range(len(self))]
        return iter(cameras)

    def split(self, sections):
        """
        Splits the camera into sections. Similar to `torch.split`.
        """
        intrinsics = torch.split(self.intrinsic, sections)
        viewports = torch.split(self.viewport, sections)
        log_quaternions = torch.split(self.log_quaternion, sections)
        translations = torch.split(self.translation, sections)

        cameras = []
        for intrinsic, viewport, log_quat, trans in zip(intrinsics, viewports,
                                                        log_quaternions, translations):
            cameras.append(Camera(intrinsic, None, self.z_span, viewport,
                                  log_quaternion=log_quat,
                                  translation=trans,
                                  width=self.width,
                                  height=self.height))

        return cameras

    @classmethod
    def cat(cls, cameras):
        z_span = cameras[0].z_span
        height = cameras[0].height
        width = cameras[0].width
        intrinsic = torch.cat([o.intrinsic for o in cameras], dim=0)
        viewport = torch.cat([o.viewport for o in cameras], dim=0)
        log_quaternion = torch.cat([o.log_quaternion for o in cameras], dim=0)
        translation = torch.cat([o.translation for o in cameras], dim=0)

        return cls(intrinsic, None, z_span, viewport,
                   log_quaternion=log_quaternion,
                   translation=translation,
                   width=width,
                   height=height)

    @classmethod
    def vcat(cls, cameras, batch_size=-1):
        """
        Concatenate the view dimension of the camera parameters.
        Args:
            cameras: cameras to concatenate
            batch_size: the batch size of the data

        Returns:
            Cameras concatenated in the view dimension then flattened
        """
        z_span = cameras[0].z_span
        height = cameras[0].height
        width = cameras[0].width
        intrinsic = torch.cat([b2bv(o.intrinsic, batch_size=batch_size) for o in cameras], dim=1)
        viewport = torch.cat([b2bv(o.viewport, batch_size=batch_size) for o in cameras], dim=1)
        log_quaternion = torch.cat([b2bv(o.log_quaternion, batch_size=batch_size) for o in cameras], dim=1)
        translation = torch.cat([b2bv(o.translation, batch_size=batch_size) for o in cameras], dim=1)

        return cls(bv2b(intrinsic), None, z_span, bv2b(viewport),
                   log_quaternion=bv2b(log_quaternion),
                   translation=bv2b(translation),
                   width=width,
                   height=height)

    def repeat(self, n):
        z_span = self.z_span
        intrinsic = self.intrinsic.repeat(n, 1, 1)
        log_quaternion = self.log_quaternion.repeat(n, 1)
        translation = self.translation.repeat(n, 1)
        viewport = self.viewport.repeat(n, 1)

        return Camera(intrinsic, None, z_span, viewport,
                      log_quaternion=log_quaternion,
                      translation=translation,
                      width=self.width,
                      height=self.height)

    def repeat_interleave(self, n):
        z_span = self.z_span
        intrinsic = torch.repeat_interleave(self.intrinsic, n, dim=0)
        viewport = torch.repeat_interleave(self.viewport, n, dim=0)
        log_quaternion = torch.repeat_interleave(self.log_quaternion, n, dim=0)
        translation = torch.repeat_interleave(self.translation, n, dim=0)

        return Camera(intrinsic, None, z_span, viewport,
                      log_quaternion=log_quaternion,
                      translation=translation,
                      width=self.width,
                      height=self.height)

    def pixel_coords_uvz(self, out_size):
        """
        Computes meshgrid coordinates for the viewport in pixel-space.
        """
        if isinstance(out_size, int):
            out_size = (out_size, out_size, out_size)

        z_pixel, v_pixel, u_pixel = torch.meshgrid([
            torch.linspace(0.0, 1.0, out_size[0], device=self.device),
            torch.linspace(0.0, 1.0, out_size[1], device=self.device),
            torch.linspace(0.0, 1.0, out_size[2], device=self.device),
        ])

        u_pixel = u_pixel.unsqueeze(0).expand(self.length, -1, -1, -1)
        v_pixel = v_pixel.unsqueeze(0).expand(self.length, -1, -1, -1)
        z_pixel = z_pixel.unsqueeze(0).expand(self.length, -1, -1, -1)

        u_pixel = (u_pixel * self.viewport_width.view(-1, 1, 1, 1)
                   + self.viewport[:, 0].view(-1, 1, 1, 1))
        v_pixel = (v_pixel * self.viewport_height.view(-1, 1, 1, 1)
                   + self.viewport[:, 1].view(-1, 1, 1, 1))

        z_pixel = z_pixel * self.z_span + self.znear.view(-1, 1, 1, 1)

        return u_pixel, v_pixel, z_pixel

    def pixel_coords_uv(self, out_size):
        if isinstance(out_size, int):
            out_size = (out_size, out_size)

        v_pixel, u_pixel = torch.meshgrid([
            torch.linspace(0.0, 1.0, out_size[0], device=self.device),
            torch.linspace(0.0, 1.0, out_size[1], device=self.device),
        ])

        u_pixel = u_pixel.expand(self.length, -1, -1)
        u_pixel = (u_pixel
                   * self.viewport_width.view(-1, 1, 1)
                   + self.viewport[:, 0].view(-1, 1, 1))
        v_pixel = v_pixel.expand(self.length, -1, -1)
        v_pixel = (v_pixel
                   * self.viewport_height.view(-1, 1, 1)
                   + self.viewport[:, 1].view(-1, 1, 1))

        return u_pixel, v_pixel

    def camera_coords(self, out_size):
        """
        Computes meshgrid coordinates for the viewport in camera-space.
        """
        if isinstance(out_size, int):
            out_size = (out_size, out_size, out_size)

        u_pixel, v_pixel, z_pixel = self.pixel_coords_uvz(out_size)
        u0 = self.u0.view(-1, 1, 1, 1)
        v0 = self.v0.view(-1, 1, 1, 1)
        fu = self.fu.view(-1, 1, 1, 1)
        fv = self.fv.view(-1, 1, 1, 1)
        z_cam = z_pixel
        y_cam = (v_pixel - v0) / fv * z_cam
        x_cam = (u_pixel - u0) / fu * z_cam

        return x_cam, y_cam, z_cam

    def depth_camera_coords(self, depth):
        u_pixel, v_pixel = self.pixel_coords_uv((depth.shape[-2], depth.shape[-1]))
        z_cam = depth.view_as(u_pixel)

        u0 = self.u0.view(-1, 1, 1)
        v0 = self.v0.view(-1, 1, 1)
        fu = self.fu.view(-1, 1, 1)
        fv = self.fv.view(-1, 1, 1)
        x_cam = (u_pixel - u0) / fu * z_cam
        y_cam = (v_pixel - v0) / fv * z_cam

        return x_cam, y_cam, z_cam

    def depth_object_coords(self, depth):
        xx, yy, zz = self.depth_camera_coords(depth)
        cam_grid = torch.stack((xx, yy, zz), dim=-1)
        cam_coords = three.grid_to_coords(cam_grid)
        obj_coords = (three.transform_coords(cam_coords, self.cam_to_obj)
                      .view_as(cam_grid))
        x_obj, y_obj, z_obj = torch.split(obj_coords, 1, dim=-1)
        return x_obj.squeeze(-1), y_obj.squeeze(-1), z_obj.squeeze(-1)

    def denormalize_depth(self, depth, eps=0.01):
        znear = (self.znear - eps).view(*depth.shape[:-3], 1, 1, 1)
        zfar = (self.zfar + eps).view(*depth.shape[:-3], 1, 1, 1)
        return (depth / 2.0 + 0.5) * (zfar - znear) + znear

    def normalize_depth(self, depth, eps=0.01):
        znear = (self.znear - eps).view(-1, 1, 1, 1)
        zfar = (self.zfar + eps).view(-1, 1, 1, 1)
        depth = (depth - znear) / (zfar - znear)
        depth = depth.clamp(0, 1) * 2.0 - 1.0
        return depth

    def clone(self):
        return Camera(self.intrinsic.clone(),
                      None,
                      self.z_span,
                      viewport=self.viewport.clone(),
                      log_quaternion=self.log_quaternion.clone(),
                      translation=self.translation.clone(),
                      width=self.width,
                      height=self.height)

    def detach(self):
        return Camera(self.intrinsic.detach(),
                      None,
                      self.z_span,
                      viewport=self.viewport.detach(),
                      log_quaternion=self.log_quaternion.detach(),
                      translation=self.translation.detach(),
                      width=self.width,
                      height=self.height)

    def __repr__(self):
        return (
            f"Camera(count={self.intrinsic.size(0)})"
        )


class BaseTransformBlock(abc.ABC, nn.Module):

    def __init__(self, cube_size):
        super().__init__()
        self.cube_size = cube_size

    def get_obj_coords(self, size, device=None):
        z_coords, y_coords, x_coords = torch.meshgrid([
            torch.linspace(-self.cube_size / 2, self.cube_size / 2, size, requires_grad=False,
                           device=device),
            torch.linspace(-self.cube_size / 2, self.cube_size / 2, size, requires_grad=False,
                           device=device),
            torch.linspace(-self.cube_size / 2, self.cube_size / 2, size, requires_grad=False,
                           device=device),
        ])
        return torch.stack((x_coords,
                            y_coords,
                            z_coords,
                            torch.ones(x_coords.shape, device=device)), dim=-1).view(-1, 4)


class CameraToObjectTransform(BaseTransformBlock):
    """
    Takes a camera-space volume and projects it onto an object-space volume.

    The camera space volume corresponds to a near-far bounded camera frustum.
    """

    def __init__(self, cube_size, padding_mode='border'):
        super().__init__(cube_size)
        self.padding_mode = padding_mode

    def forward(self, cam_volume, camera: Camera):
        size = cam_volume.size(-1)
        obj_coords = self.get_obj_coords(size, device=cam_volume.device)
        obj_coords = (obj_coords.t()
                      .unsqueeze(0)
                      .expand(cam_volume.size(0), -1, -1))
        cam_coords = camera.obj_to_cam @ obj_coords

        # Project canonical volume coordinates onto camera volume coordinates.
        # Camera volume coordinates is a range limited view frustum with
        # unnormalized coordinates (the z-coordinate is undivided).
        pixel_coords = (camera.intrinsic @ cam_coords)
        pixel_coords[:, :2] /= pixel_coords[:, 2, None]

        znear = camera.znear.view(-1, 1)
        zfar = camera.zfar.view(-1, 1)

        # grid_sample takes values between -1 and 1, so transform to be in this range.
        # Only sample from pixels within the viewport.
        grid_coords = torch.stack((
            # x coordinates.
            ((pixel_coords[:, 0] - camera.viewport[:, 0, None])
             / camera.viewport_width[:, None]) * 2 - 1,
            # y coordinates.
            ((pixel_coords[:, 1] - camera.viewport[:, 1, None])
             / camera.viewport_height[:, None]) * 2 - 1,
            # z coordinates.
            (pixel_coords[:, 2] - znear) / (zfar - znear)
        ), dim=-1)
        grid = grid_coords.view(-1, size, size, size, 3)
        grid_samples = _grid_sample(cam_volume, grid, padding_mode=self.padding_mode)

        return grid_samples


class ObjectToCameraTransform(BaseTransformBlock):
    """
    Takes an object-space volume and projects it onto a camera-space volume.
    """

    def __init__(self, cube_size, padding_mode='border'):
        super().__init__(cube_size)
        self.padding_mode = padding_mode

    def forward(self, obj_volume, camera: Camera):
        size = obj_volume.size(-1)
        x_cam, y_cam, z_cam = camera.camera_coords(size)

        # Flatten coordinates so we can do matrix math.
        cam_coords = (torch.stack((
            x_cam,
            y_cam,
            z_cam,
            torch.ones(*x_cam.size(), device=camera.device)
        ), dim=-1)).view(camera.length, -1, 4)

        # Transform to object coordinates, and then divide by size to get grid coordinates.
        obj_coords = camera.cam_to_obj @ cam_coords.transpose(2, 1)
        obj_coords = obj_coords[:, :3, :].transpose(1, 2)
        grid_coords = obj_coords / (self.cube_size / 2)
        grid = grid_coords.view(-1, size, size, size, 3)

        obj_volume = obj_volume.expand(camera.length, -1, -1, -1, -1)
        grid_samples = _grid_sample(obj_volume, grid, padding_mode=self.padding_mode)

        return grid_samples


class TileProjection2d3d(nn.Module):

    def __init__(self, in_channels, out_channels, out_size, relu_slope=0.2,
                 norm_module=PixelNorm):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.activation = nn.LeakyReLU(relu_slope)
        self.norm = norm_module()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x.unsqueeze(2).expand(-1, -1, self.out_size, -1, -1)


class FactorProjection2d3d(nn.Module):
    def __init__(self, in_channels, out_channels, out_size, relu_slope=0.2,
                 norm_module=PixelNorm):
        super().__init__()
        self.out_size = out_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = EqualizedConv2d(in_channels, out_channels * out_size,
                                    kernel_size=1, padding=0)
        self.activation = nn.LeakyReLU(relu_slope)
        self.norm = norm_module()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x.view(x.size(0), self.out_channels, -1, x.size(-2), x.size(-1))


class FactorProjection3d2d(nn.Module):
    def __init__(self, in_channels, out_channels, out_size, relu_slope=0.2,
                 norm_module=PixelNorm):
        super().__init__()
        self.out_size = out_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = EqualizedConv2d(in_channels * out_size, out_channels,
                                    kernel_size=1, padding=0)
        self.activation = nn.LeakyReLU(relu_slope)
        self.norm = norm_module()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x
