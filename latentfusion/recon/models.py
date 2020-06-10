import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from latentfusion import pggan
from latentfusion import torchutils
from latentfusion.augment import gan_normalize
from latentfusion.modules import unet, EqualizedConv3d
from latentfusion.modules.blocks import create_blocks, OutputBlock3d, OutputBlock2d
from latentfusion.modules.geometry import (TileProjection2d3d, FactorProjection2d3d,
                                           CameraToObjectTransform, Camera,
                                           ObjectToCameraTransform, FactorProjection3d2d)
from latentfusion.recon import fusion
from latentfusion.recon.utils import get_normalized_voxel_depth
from latentfusion.three import b2bv, bv2b


def _get_activation(activation_type, relu_slope=0.2):
    if activation_type is None or activation_type == 'none':
        return None
    elif activation_type == 'lrelu':
        return nn.LeakyReLU(relu_slope)
    if activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unknown activation type {activation_type}')


def load_models(checkpoint, kwargs=None, device=None, return_generator=False):
    if kwargs is None:
        kwargs = checkpoint['args']

    # Fix legacy checkpoints.
    sculptor_checkpoint = checkpoint['modules']['sculptor']
    if 'input_color' not in sculptor_checkpoint['args']:
        sculptor_checkpoint['args']['input_color'] = True
    if 'input_depth' not in sculptor_checkpoint['args']:
        sculptor_checkpoint['args']['input_depth'] = kwargs['generator_input_depth']
    if 'input_mask' not in sculptor_checkpoint['args']:
        sculptor_checkpoint['args']['input_mask'] = kwargs['generator_input_mask']

    photographer_checkpoint = checkpoint['modules']['photographer']
    if 'predict_color' not in photographer_checkpoint['args']:
        photographer_checkpoint['args']['predict_color'] = kwargs['predict_color']
    if 'predict_depth' not in photographer_checkpoint['args']:
        photographer_checkpoint['args']['predict_depth'] = kwargs['predict_depth']
    if 'predict_mask' not in photographer_checkpoint['args']:
        photographer_checkpoint['args']['predict_mask'] = kwargs['predict_mask']

    sculptor = Sculptor.from_checkpoint(sculptor_checkpoint).to(device)
    photographer = Photographer.from_checkpoint(photographer_checkpoint).to(device)
    fuser = fusion.from_checkpoint(checkpoint['modules']['fuser']).to(device)

    if not kwargs.get('no_discriminator', False):
        discriminator = pggan.MultiScaleDiscriminator.from_checkpoint(
            checkpoint.get('modules', checkpoint)['discriminator']).to(device)
    else:
        discriminator = None

    if return_generator:
        if 'generator'in checkpoint.get('modules', {}):
            generator = unet.UNet2d.from_checkpoint(checkpoint['modules']['generator']).to(device)
        else:
            generator = None
        return sculptor, fuser, photographer, discriminator, generator

    return sculptor, fuser, photographer, discriminator


def autoencode(sculptor, fuser, photographer, camera, color, depth=None, mask=None):
    z_obj, _ = sculptor.encode(fuser, camera, color, depth, mask)
    y, z_pix, _ = photographer.decode(z_obj, camera, return_latent=True, interpret_logits=True)

    # Autoencoded targets have one view. Squeeze the view dimension.
    y = {k: v.squeeze(1) for k, v in y.items()}
    z_pix = z_pix.squeeze(1)

    return y, z_pix


class Sculptor(nn.Module):

    def __init__(self, in_size, image_config, camera_config, object_config,
                 relu_slope=0.2, cube_size=1.0, cube_activation_type=None,
                 projection_type='tile', input_color=True, input_depth=False, input_mask=True,
                 scale_mode='bilinear',
                 **kwargs):
        super().__init__()

        self.image_config = image_config
        self.camera_config = camera_config
        self.object_config = object_config

        self.input_color = input_color
        self.input_depth = input_depth
        self.input_mask = input_mask

        self.relu_slope = relu_slope
        self.cube_size = cube_size
        self.cube_activation_type = cube_activation_type
        self.projection_type = projection_type
        self.scale_mode = scale_mode

        # Set number if input channels based on inputs.
        self.in_channels = 0
        if input_color:
            self.in_channels += 3
        if input_mask:
            self.in_channels += 1
        if input_depth:
            self.in_channels += 1
        self.in_size = in_size

        # Network blocks.
        self.image_encoder = unet.UNet2d(self.in_channels, None, self.image_config)

        if projection_type == 'tile':
            self.projection_block = TileProjection2d3d(
                in_channels=self.image_encoder.out_channels,
                out_channels=self.camera_config[0],
                out_size=self.image_out_size)
        elif projection_type == 'factor':
            self.projection_block = FactorProjection2d3d(
                in_channels=self.image_encoder.out_channels,
                out_channels=self.camera_config[0],
                out_size=self.image_out_size)
        else:
            raise ValueError(f"Unknown projection type {projection_type!r}")

        self.camera_blocks = create_blocks(self.camera_config, EqualizedConv3d, 0.5,
                                           scale_mode=scale_mode)
        self.transform_block = CameraToObjectTransform(cube_size)

        if self.object_config:
            self.object_blocks = create_blocks(self.object_config, EqualizedConv3d, 0.5,
                                               scale_mode=scale_mode)
        else:
            self.object_blocks = nn.ModuleList()

        self.output_block = OutputBlock3d(self.out_channels, self.out_channels,
                                          activation=_get_activation(cube_activation_type))

    @property
    def image_out_size(self):
        return self.image_encoder.output_size(self.in_size)

    @property
    def camera_out_size(self):
        return self.image_out_size // (2 ** self.camera_config.count('D'))

    @property
    def out_size(self):
        if self.object_config:
            return self.camera_out_size // (2 ** self.object_config.count('D'))
        else:
            return self.camera_out_size

    @property
    def image_bottleneck_size(self):
        return self.image_encoder.bottleneck_size(self.in_size)

    @property
    def out_channels(self):
        if self.object_config:
            return self.object_config[-1]
        else:
            return self.camera_config[-1]

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        return {
            'args': {
                'in_channels': self.in_channels,
                'in_size': self.in_size,
                'image_config': self.image_config,
                'camera_config': self.camera_config,
                'object_config': self.object_config,
                'relu_slope': self.relu_slope,
                'cube_size': self.cube_size,
                'cube_activation_type': self.cube_activation_type,
                'projection_type': self.projection_type,
                'input_color': self.input_color,
                'input_depth': self.input_depth,
                'input_mask': self.input_mask,
                'scale_mode': self.scale_mode,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def forward(self, x, camera: Camera):
        with autocast(enabled=self.training):
            z = self.image_encoder(x)

            # Projection from 2D to 3D.
            z = self.projection_block(z)

            z_cam_mid = []
            z_obj_mid = []

            # Camera-space encoder blocks.
            for block in self.camera_blocks:
                z = block(z)
                z_cam_mid.append(self.transform_block(z, camera))

            # Camera-space => Object-space.
            z = self.transform_block(z, camera)

            # Object-space encoder blocks.
            if self.object_blocks:
                for block in self.object_blocks:
                    z = block(z)
                    z_obj_mid.append(z)

            z = self.output_block(z)

            return z, z_cam_mid, z_obj_mid

    def encode(self, fuser, camera, color, depth=None, mask=None, data_parallel=False):
        device = torchutils.module_device(self)
        if len(color.shape) == 5:
            num_views = color.shape[1]
        else:
            num_views = 1

        x = []
        if self.input_color:
            if len(color.shape) == 5:
                color = bv2b(color)
            x.append(color)
        if self.input_depth:
            if len(depth.shape) == 5:
                depth = bv2b(depth)
            x.append(depth)
        if self.input_mask:
            if len(mask.shape) == 5:
                mask = bv2b(mask)
            x.append(gan_normalize(mask))
        x = torch.cat(x, dim=1).to(device)

        sculptor = torchutils.MyDataParallel(self) if data_parallel else self
        fuser = torchutils.MyDataParallel(fuser) if data_parallel else fuser

        z_obj, z_cam_mid, z_obj_mid = sculptor(x, camera.to(device))
        z_obj = b2bv(z_obj, num_views)
        z_cam_mid = [b2bv(z, num_views) for z in z_cam_mid]
        z_obj_mid = [b2bv(z, num_views) for z in z_obj_mid]

        z_obj, z_extra = fuser(z_obj, z_cam_mid, z_obj_mid, camera)

        return z_obj, z_extra


class Photographer(nn.Module):

    def __init__(self, in_size, image_config, camera_config, object_config,
                 projection_type='sum', occlusion_config=False, in_views=1, skip_connections=False,
                 relu_slope=0.2, cube_size=1.0, predict_color=False, predict_depth=True, predict_mask=True,
                 scale_mode='bilinear',
                 **kwargs):
        super().__init__()

        self.image_config = image_config
        self.camera_config = camera_config
        self.occlusion_config = occlusion_config
        self.object_config = object_config
        self.projection_type = projection_type

        self.predict_color = predict_color
        self.predict_depth = predict_depth
        self.predict_mask = predict_mask

        self.in_views = in_views
        self.relu_slope = relu_slope
        self.skip_connections = skip_connections
        self.cube_size = cube_size
        self.scale_mode = scale_mode

        self.in_size = in_size

        # Set output channels based on prediction flags.
        self.out_channels = []
        if predict_color:
            self.out_channels.append(3)
        if predict_depth:
            self.out_channels.append(1)
        if predict_mask:
            self.out_channels.append(1)

        if self.object_config:
            self.object_blocks = create_blocks(self.object_config, EqualizedConv3d, 2.0,
                                               in_views=in_views,
                                               skip_connections=skip_connections,
                                               scale_mode=scale_mode)
        else:
            self.object_blocks = nn.ModuleList()
        self.transform_block = ObjectToCameraTransform(cube_size)
        if occlusion_config:
            self.occlusion_module = unet.UNet3d(self.object_config[-1] + 1, 1, occlusion_config)
        else:
            self.occlusion_module = None

        self.camera_blocks = create_blocks(self.camera_config, EqualizedConv3d, 2.0,
                                           skip_connections=skip_connections,
                                           skip_connect_start=True,
                                           skip_connection_views=in_views,
                                           scale_mode=scale_mode)

        if projection_type == 'factor':
            self.projection_block = FactorProjection3d2d(self.camera_config[-1],
                                                         self.image_config[0][0],
                                                         out_size=self.camera_out_size)
        else:
            self.projection_block = None

        if isinstance(self.out_channels, int):
            # If the output channels are treated as a single n-channel image.
            self.image_decoder = unet.UNet2d(None, self.out_channels, self.image_config)
            self.output_blocks = None
        else:
            # If the outputs should be split into branches.
            self.image_decoder = unet.UNet2d(None, None, self.image_config)
            self.output_blocks = nn.ModuleList([
                OutputBlock2d(self.image_decoder.out_channels, c) for c in self.out_channels
            ])

    @property
    def object_out_size(self):
        return self.in_size * (2 ** self.object_config.count('U'))

    @property
    def camera_out_size(self):
        return self.object_out_size * (2 ** self.camera_config.count('U'))

    @property
    def out_size(self):
        return self.image_decoder.output_size(self.camera_out_size)

    @property
    def image_bottleneck_size(self):
        return self.image_decoder.bottleneck_size(self.camera_out_size)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        return {
            'args': {
                'image_config': self.image_config,
                'camera_config': self.camera_config,
                'occlusion_config': self.occlusion_config,
                'object_config': self.object_config,
                'projection_type': self.projection_type,
                'relu_slope': self.relu_slope,
                'out_channels': self.out_channels,
                'in_views': self.in_views,
                'in_size': self.in_size,
                'skip_connections': self.skip_connections,
                'cube_size': self.cube_size,
                'predict_color': self.predict_color,
                'predict_depth': self.predict_depth,
                'predict_mask': self.predict_mask,
                'scale_mode': self.scale_mode,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def _compute_depth_weights(self, z_cam):
        # Concatenate camera-space coordinates to input.
        coords = get_normalized_voxel_depth(z_cam)
        z = torch.cat((z_cam, coords), dim=1)

        # Softmax along depth dimension.
        logits = self.occlusion_module(z)
        logits_resized = F.interpolate(logits, z_cam.size(-1))
        weights = torch.softmax(logits, dim=2)
        weights_resized = torch.softmax(logits_resized, dim=2)

        return weights, weights_resized

    def _depth_from_weight(self, depth_weights):
        voxel_depth = get_normalized_voxel_depth(depth_weights)
        # Depth is weighted sum of z-coordinates.
        depth = (voxel_depth * depth_weights).sum(dim=2)
        return depth

    def forward(self, z_obj, camera, z_cam_mid=None, z_obj_mid=None, return_latent=False):
        if z_obj.shape[0] != len(camera):
            raise ValueError(f"batch dimension of z_obj and camera much match. ({z_obj} != {len(camera)})")
        if z_cam_mid is None and self.skip_connections:
            raise ValueError("z_cam_intermediate required for skip connections.")
        if z_obj_mid is None and self.skip_connections:
            raise ValueError("z_obj_intermediate required for skip connections.")

        with autocast(enabled=self.training):
            if self.skip_connections:
                # Transform intermediate representations to this camera.
                z_cam_mid = [self.transform_block(z_cam, camera) for z_cam in z_cam_mid]

            z = z_obj

            # Object-space decoder blocks.
            for block_id, block in enumerate(self.object_blocks):
                if self.skip_connections and block_id >= 1:
                    z = torch.cat((z, z_obj_mid[-block_id - 1]), dim=1)
                z = block(z)

            # Transform from object-space to camera-space.
            z = self.transform_block(z, camera)

            # Camera-space decoder blocks.
            for block_id, block in enumerate(self.camera_blocks):
                if self.skip_connections:
                    z = torch.cat((z, z_cam_mid[-block_id - 1]), dim=1)
                z = block(z)

            if self.occlusion_module:
                z_weights, depth_weights_resized = self._compute_depth_weights(z)
                z_depth = self._depth_from_weight(z_weights)
                z = z * depth_weights_resized
            else:
                z_weights = None
                z_depth = None

            # Projection from camera-space to image-space.
            if self.projection_type == 'sum':
                z = z.sum(dim=2)
            elif self.projection_type == 'factor':
                z = self.projection_block(z)
            # Image-space decoder U-Net.
            y = self.image_decoder(z)

            # If there are output blocks run them and concatenate along channel dim.
            if self.output_blocks:
                outputs = []
                for output_block in self.output_blocks:
                    outputs.append(output_block(y))
                y = torch.cat(outputs, dim=1)

            if return_latent:
                return y, z, z_depth

            return y, None, z_depth

    def interpret_logits(self, logits, apply_mask=False):
        # Process logits.
        channel_base = 0
        y = {}

        if self.predict_color:
            y['color_logits'] = logits[:, channel_base:channel_base + 3]
            y['color'] = torch.tanh(y['color_logits'])
            channel_base += 3

        if self.predict_depth:
            y['depth_logits'] = logits[:, channel_base:channel_base + 1]
            y['depth'] = torch.tanh(y['depth_logits'])
            channel_base += 1

        if self.predict_mask:
            y['mask_logits'] = logits[:, channel_base:channel_base + 1]
            y['mask'] = torch.sigmoid(y['mask_logits'])
            channel_base += 1
        else:
            y['mask'] = (y['depth'].detach() > -1.0).float()
            y['mask_logits'] = 100 * y['mask'] + (-100) * (1.0 - y['mask'])

        if apply_mask and self.predict_mask:
            if self.predict_depth:
                y['depth'] = (y['depth'] + 1) * (y['mask'] > 0.5) - 1
            if self.predict_color:
                y['color'] = y['color'] * (y['mask'] > 0.5)

        return y

    def decode(self, z_obj, camera, interpret_logits=True, return_latent=False,
               data_parallel=False, apply_mask=False):
        # Auto expand latent cube to match camera.
        num_batch = z_obj.shape[0]
        num_views = camera.length // z_obj.shape[0]
        # z_obj = z_obj.expand(-1, num_views, -1, -1, -1, -1)
        # z_obj = z_obj.reshape(-1, *z_obj.shape[2:])
        z_obj = z_obj.expand(-1, num_views, -1, -1, -1, -1)
        z_obj = z_obj.reshape(-1, *z_obj.shape[2:])

        photographer = torchutils.MyDataParallel(self) if data_parallel else self
        y, z, z_depth = photographer(z_obj, camera, return_latent=return_latent)
        if z is not None:
            z = b2bv(z, num_views)

        if interpret_logits:
            y = self.interpret_logits(y, apply_mask=apply_mask)
            y = {k: b2bv(v, num_views) for k, v in y.items()}

        return y, z, z_depth
