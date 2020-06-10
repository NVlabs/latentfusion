import functools
import typing

import torch
from torch import nn
from torch.nn import functional as F

from latentfusion.modules import EqualizedConv2d, Interpolate, PixelNorm

__all__ = ['Encoder', 'Decoder', 'EncoderDecoder']


class InputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, relu_slope=0.2, padding=0):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.LeakyReLU(relu_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class OutputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 scale_mode='nearest', kernel_size=3, padding=1,
                 relu_slope=0.2):
        super().__init__()
        self.interpolate = Interpolate(scale_factor, mode=scale_mode)
        self.activation = nn.LeakyReLU(relu_slope)
        self.norm = PixelNorm()

        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.interpolate(x)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.norm(x)

        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, block_config: typing.List[int], intermediate_inputs=False,
                 scale_mode='nearest'):
        super().__init__()

        self.block_config = block_config

        self.input_blocks = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()

        for block_id, (block_in, block_out) in enumerate(zip(block_config[:-1], block_config[1:])):
            if intermediate_inputs or block_id == 0:
                self.input_blocks.append(InputBlock(in_channels, block_in))
            self.encoder_blocks.append(
                Block(block_in, block_out, scale_factor=0.5, scale_mode=scale_mode))

        self.input_level = 0

    @property
    def num_blocks(self):
        return len(self.block_config) - 1

    def forward(self, x):
        input_block = self.input_blocks[self.input_level]

        # Scale input to match input level.
        if self.input_level > 0:
            input_scale = 2 ** (-self.input_level)
            x = F.interpolate(x, scale_factor=input_scale)

        z_intermediates = []
        z = input_block(x)
        for block in self.encoder_blocks:
            z = block(z)
            z_intermediates.append(z)

        return z, z_intermediates


class Decoder(nn.Module):

    def __init__(self, out_channels, block_config: typing.List[int], intermediate_outputs=False,
                 style_size=8, skip_connections=True, scale_mode='nearest', output_activation=None):
        super().__init__()

        self.style_size = style_size
        self.skip_connections = skip_connections

        self.decoder_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()

        block_config = list(reversed(block_config))
        # Add size of latent style vector to first block.
        block_config[0] += self.style_size
        self.block_config = block_config

        for block_id, (block_in, block_out) in enumerate(zip(block_config[:-1], block_config[1:])):
            if self.skip_connections and block_id >= 1:
                block_in *= 2
            self.decoder_blocks.append(
                Block(block_in, block_out, scale_factor=2, scale_mode=scale_mode))
            if intermediate_outputs or block_id == self.num_blocks - 1:
                self.output_blocks.append(OutputBlock(block_out, out_channels))

        if output_activation is None:
            self.output_activation = None
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'clamp':
            self.output_activation = functools.partial(torch.clamp, min=-1, max=1)
        else:
            raise ValueError(f"Unknown output activation {output_activation}")

        self.output_level = 0

    @property
    def num_blocks(self):
        return len(self.block_config) - 1

    def forward(self, z_content, z_content_intermediates=None, z_style=None):
        if z_style is None and self.style_size > 0:
            raise ValueError("z_style cannot be None if style_size > 0. "
                             f"(style_size={self.style_size})")

        if z_content_intermediates is None and self.skip_connections:
            raise ValueError("z_content_intermediates cannot be None if skip connections are on.")

        if z_style is not None:
            assert z_style.size(0) == z_content.size(0)
            assert z_style.size(1) == self.style_size
            # Expand z_style to the spatial size of z_content.
            z_style = z_style.view(*z_style.shape, 1, 1).expand(-1, -1, *z_content.shape[2:])
            # Concatenate z to x in the channel dimension.
            z = torch.cat((z_content, z_style), dim=1)
        else:
            z = z_content

        for block_id, block in enumerate(self.decoder_blocks):
            if self.skip_connections and block_id >= 1:
                z = torch.cat((z, z_content_intermediates[-block_id - 1]), dim=1)
            z = block(z)

        output_block = self.output_blocks[-self.output_level-1]
        y = output_block(z)

        if self.output_activation is not None:
            y = self.output_activation(y)

        return y


class EncoderDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, block_config=None, intermediate_inputs=False,
                 style_size=8, skip_connections=True, scale_mode='bilinear',
                 output_activation=None):
        super().__init__()

        if block_config is None:
            block_config = [32, 64, 128, 256, 512, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_config = block_config
        self.style_size = style_size

        self.skip_connections = skip_connections
        self.intermediate_inputs = intermediate_inputs
        self.scale_mode = scale_mode
        self.output_activation = output_activation

        self.encoder = Encoder(in_channels, block_config, intermediate_inputs,
                               scale_mode=scale_mode)
        self.decoder = Decoder(out_channels, block_config, intermediate_inputs,
                               style_size=style_size, skip_connections=skip_connections,
                               scale_mode=scale_mode, output_activation=output_activation)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        return {
            'args': {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'block_config': self.block_config,
                'intermediate_inputs': self.intermediate_inputs,
                'style_size': self.style_size,
                'skip_connections': self.skip_connections,
                'scale_mode': self.scale_mode,
                'output_activation': self.output_activation,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def forward(self, x, z_style=None):
        z_content, z_content_intermediates = self.encoder(x)
        if not self.skip_connections:
            z_content_intermediates = None
        y = self.decoder(z_content, z_content_intermediates, z_style)

        return y, z_content
