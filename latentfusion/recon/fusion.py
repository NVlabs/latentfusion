import abc
from typing import Dict, Tuple, Any

import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from latentfusion.functional import absolute_max_pool
from latentfusion.modules import unet, EqualizedConv3d, EqualizedConv2d
from latentfusion.modules.geometry import CameraToObjectTransform, Camera
from latentfusion.modules.gru import ConvGRUCell
from latentfusion.modules.lstm import ConvLSTMCell
from latentfusion.recon import utils
from latentfusion.three.batchview import bv2b, b2bv


def get_fuser(fuser_type, in_channels, cube_size, block_config=None,
              conv_module=EqualizedConv3d):
    if fuser_type.startswith('pool:'):
        _, pool_type = fuser_type.split(':')
        return PoolFuser(pool_type)
    elif fuser_type == 'concat':
        return ConcatFuser()
    elif fuser_type == 'blend':
        return BlendFuser(block_config,
                          in_channels=in_channels,
                          cube_size=cube_size,
                          conv_module=conv_module)
    elif fuser_type == 'gru':
        return GRUFuser(in_channels=in_channels,
                        cube_size=cube_size,
                        conv_module=conv_module)
    elif fuser_type == 'lstm':
        return LSTMFuser(in_channels=in_channels,
                         cube_size=cube_size,
                         conv_module=conv_module)
    else:
        raise ValueError(f"Unknown fuser type {type!r}")


def from_checkpoint(checkpoint):
    return globals()[checkpoint['type']].from_checkpoint(checkpoint)


def pool_tensor(tensor, pool_type, dim=0):
    if pool_type == 'max':
        tensor, _ = tensor.max(dim=dim, keepdim=True)
    elif pool_type == 'abs_max':
        tensor = absolute_max_pool(tensor, dim=dim)
    elif pool_type == 'mean':
        tensor = tensor.mean(dim=dim, keepdim=True)
    elif pool_type == 'median':
        tensor, _ = tensor.median(dim=dim, keepdim=True)
    else:
        raise ValueError(f"Unknown pool_type value {pool_type}")

    return tensor


class Fuser(nn.Module, abc.ABC):

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls()

    def create_checkpoint(self):
        return {
            'type': self.__class__.__qualname__,
        }

    @abc.abstractmethod
    def forward(self, z_obj, z_cam_mid, z_obj_mid, camera: Camera) \
            -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplemented


class PoolFuser(Fuser):

    def __init__(self, pool_type='mean'):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, z_obj, z_cam_mid, z_obj_mid, camera):
        return pool_tensor(z_obj, self.pool_type, dim=1), {}


class ConcatFuser(Fuser):

    def forward(self, z_obj, z_cam_mid, z_obj_mid, camera):
        N, V, C, D, H, W = z_obj.size()
        z_fused = z_obj.reshape(N, 1, V * C, D, H, W)
        return z_fused, {}


class BlendFuser(Fuser):

    def __init__(self, block_config, in_channels, cube_size=1.0, conv_module=EqualizedConv3d):
        super().__init__()

        self.block_config = block_config
        self.in_channels = in_channels
        self.cube_size = cube_size

        self.unet = unet.BaseUNet(in_channels + 1, 1, block_config, conv_module=conv_module)
        self.transform_block = CameraToObjectTransform(cube_size)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        checkpoint = super().create_checkpoint()
        return {
            **checkpoint,
            'args': {
                'block_config': self.block_config,
                'in_channels': self.in_channels,
                'cube_size': self.cube_size,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def compute_blend_weights(self, z_cam, camera):
        num_views = z_cam.shape[1]

        z_cam = bv2b(z_cam)

        # Concatenate camera-space coordinates to input.
        coords = utils.get_normalized_voxel_depth(z_cam)

        w = torch.cat((z_cam, coords), dim=1)
        w = self.unet(w)
        w = self.transform_block(w, camera)
        w = b2bv(w, num_views)

        # Softmax along view dimension.
        w = torch.softmax(w, dim=1)

        return w

    def forward(self, z_obj, z_cam_mid, z_obj_mid, camera):
        blend_weights = self.compute_blend_weights(z_cam_mid[-1], camera)
        extra = {
            'blend_weights': blend_weights.squeeze(2),
        }
        z_fused = torch.sum(z_obj * blend_weights, dim=1, keepdim=True)
        return z_fused, extra


class GRUFuser(Fuser):

    def __init__(self, in_channels, cube_size=1.0, conv_module=EqualizedConv3d):
        super().__init__()

        self.in_channels = in_channels
        self.cube_size = cube_size
        self.conv_module = conv_module

        num_coord_channels = 2 if conv_module == EqualizedConv2d else 3
        self.gru = ConvGRUCell(in_channels + num_coord_channels, in_channels,
                               kernel_size=3,
                               bias=True,
                               conv_module=conv_module)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        checkpoint = super().create_checkpoint()
        return {
            **checkpoint,
            'args': {
                'in_channels': self.in_channels,
                'cube_size': self.cube_size,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def forward(self, z_obj, z_cam_mid, z_obj_mid, camera):
        with autocast(enabled=self.training):
            num_views = z_obj.shape[1]

            h = z_obj[:, 0]
            if self.conv_module == EqualizedConv2d:
                # Concatenate pixel coords if 2d.
                coords = utils.get_normalized_pixel_coords(h)
            else:
                coords = utils.get_normalized_voxel_coords(h)

            for i in range(1, num_views):
                x = torch.cat((z_obj[:, i], coords), dim=1)
                h = self.gru(x, h)

            h = h.unsqueeze(1)

            return h, {}


class LSTMFuser(Fuser):

    def __init__(self, in_channels, cube_size=1.0, conv_module=EqualizedConv3d):
        super().__init__()

        self.in_channels = in_channels
        self.cube_size = cube_size

        self.lstm = ConvLSTMCell(in_channels + 3, in_channels,
                                 kernel_size=3,
                                 bias=True,
                                 conv_module=conv_module)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        checkpoint = super().create_checkpoint()
        return {
            **checkpoint,
            'args': {
                'in_channels': self.in_channels,
                'cube_size': self.cube_size,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def forward(self, z_obj, z_cam_mid, z_obj_mid, camera):
        num_views = z_obj.shape[1]

        h = z_obj[:, 0]
        c = torch.zeros_like(h)
        coords = utils.get_normalized_voxel_coords(h)
        for i in range(1, num_views):
            x = torch.cat((z_obj[:, i], coords), dim=1)
            h, c = self.lstm(x, (h, c))

        h = h.unsqueeze(1)

        return h, {}
