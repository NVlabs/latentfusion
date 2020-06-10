import torch
from torch import nn

from latentfusion.modules import EqualizedConv2d, EqualizedConv3d
from latentfusion.modules.blocks import create_blocks, InputBlock, OutputBlock, count_blocks


class BaseUNet(nn.Module):

    def __init__(self, in_channels, out_channels, block_config, conv_module):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.block_config = block_config
        self._conv_module = conv_module

        if in_channels is not None:
            self.input_block = InputBlock(in_channels, self.down_block_config[0],
                                          conv_module=conv_module)
        else:
            self.input_block = None

        self.down_blocks = create_blocks(self.down_block_config, conv_module, 0.5)
        self.up_blocks = create_blocks(self.up_block_config, conv_module, 2.0,
                                       skip_connections=True,
                                       skip_connect_end=min(count_blocks(self.down_block_config),
                                                            count_blocks(self.up_block_config)))

        if out_channels is None:
            self.output_block = None
        elif isinstance(out_channels, int):
            self.output_block = OutputBlock(self.up_block_config[-1], out_channels,
                                            conv_module=conv_module)
        else:
            self.output_block = nn.ModuleList([
                OutputBlock(self.up_block_config[-1], c, conv_module=conv_module)
                for c in self._out_channels
            ])

    @classmethod
    def from_checkpoint(cls, checkpoint):
        checkpoint['args'].pop('conv_module')
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        if self._conv_module is not None:
            conv_module = self._conv_module.__class__.__qualname__
        else:
            conv_module = None

        return {
            'args': {
                'in_channels': self._in_channels,
                'out_channels': self._out_channels,
                'block_config': self.block_config,
                'conv_module': conv_module,
            },
            'state_dict': self.cpu().state_dict(),
        }

    @property
    def in_channels(self):
        if self._in_channels is not None:
            return sum(self._in_channels)

        return self.down_block_config[0]

    @property
    def out_channels(self):
        if self._out_channels is not None:
            return sum(self._out_channels)

        return self.up_block_config[-1]

    @property
    def down_block_config(self):
        return self.block_config[0]

    @property
    def up_block_config(self):
        return self.block_config[1]

    def bottleneck_size(self, in_size):
        num_downsamples = self.block_config[0].count('I') + self.block_config[0].count('D')
        return in_size // (2 ** num_downsamples)

    def output_size(self, in_size):
        bottleneck_size = self.bottleneck_size(in_size)
        num_upsamples = self.block_config[1].count('I') + self.block_config[1].count('U')
        return bottleneck_size * (2 ** num_upsamples)

    def forward(self, z, z_inject=None, return_intermediate=False):
        if self.input_block is not None:
            z = self.input_block(z)

        x_intermediate = []
        for block in self.down_blocks:
            z = block(z)
            x_intermediate.insert(0, z)

        if z_inject is not None:
            assert z_inject.size(0) == z.size(0)
            # Expand z_inject to the spatial size of z_content.
            z_inject = z_inject.view(*z_inject.shape, *[1 for _ in z.shape[2:]]).expand(-1, -1, *z.shape[2:])
            # Concatenate z to x in the channel dimension.
            z = torch.cat((z, z_inject), dim=1)

        for block_id, block in enumerate(self.up_blocks):
            if 1 <= block_id < len(x_intermediate):
                z = torch.cat((z, x_intermediate[block_id]), dim=1)
            z = block(z)

        if isinstance(self.output_block, OutputBlock):
            z = self.output_block(z)
        elif self.output_block is not None:
            outputs = []
            for output_block in self.output_block:
                outputs.append(output_block(z))
            z = torch.cat(outputs, dim=1)

        if return_intermediate:
            return z, x_intermediate

        return z


class UNet2d(BaseUNet):

    def __init__(self, in_channels, out_channels, block_config):
        super().__init__(in_channels, out_channels, block_config, conv_module=EqualizedConv2d)


class UNet3d(BaseUNet):

    def __init__(self, in_channels, out_channels, block_config):
        super().__init__(in_channels, out_channels, block_config, conv_module=EqualizedConv3d)
