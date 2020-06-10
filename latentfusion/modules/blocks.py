from torch import nn

from latentfusion.modules import PixelNorm, Interpolate, EqualizedConv2d, EqualizedConv3d


def count_blocks(config):
    return sum(1 for b in config if isinstance(b, int)) - 1


def create_blocks(config, conv_module, scale_factor, scale_mode='bilinear', kernel_size=3,
                  skip_connections=False,
                  skip_connect_start=1,
                  skip_connect_end=None,
                  in_views=1,
                  skip_connection_views=None) -> nn.ModuleList:
    """
    Creates convolutional blocks based on the given block configuration.

    Args:
        config: the block configuration to use
        conv_module: the convolution module to use (e.g., EqualizedConv2d)
        scale_factor: factor to upsample/downsample by (e.g., 0.5 or 2.0)
        scale_mode: how to do the sampling (e.g., bilinear or nearest)
        kernel_size: the kernel size of the convolutions
        skip_connections: whether to allocate channels for skip connections
        skip_connect_start: index of the block with the first skip connection
        skip_connect_end: index of the block with the last skip connection (exclusive)
        in_views: the number of views used in each input (for concat-pooled inputs)

    Returns:
        nn.ModuleList: the list of blocks modules

    """
    if conv_module == EqualizedConv3d and scale_mode == 'bilinear':
        scale_mode = 'trilinear'

    if skip_connection_views is None:
        skip_connection_views = in_views

    num_blocks = count_blocks(config)
    if skip_connect_end is None:
        skip_connect_end = num_blocks

    skip_connect_end = min(num_blocks, skip_connect_end)

    blocks = []
    num_conv_blocks = 0
    scale_next_block = 1.0
    block_in = config[0]
    for i, block_out in enumerate(config[1:]):
        if isinstance(block_out, int) or block_out.isdigit():
            skip_in = 0
            if skip_connections and (skip_connect_start <= num_conv_blocks < skip_connect_end):
                # Skip connections will contain all views (in the case of concat pooling).
                skip_in = block_in * skip_connection_views
            if num_conv_blocks == 0:
                block_in *= in_views
            blocks.append(Block(block_in + skip_in, int(block_out),
                                kernel_size=kernel_size,
                                conv_module=conv_module,
                                scale_mode=scale_mode,
                                scale_factor=scale_next_block))
            block_in = block_out
            num_conv_blocks += 1
            if scale_next_block != 1.0:
                scale_next_block = 1.0
        elif block_out == 'I':
            scale_next_block = scale_factor
        elif block_out == 'U':
            scale_next_block = 2.0
        elif block_out == 'D':
            scale_next_block = 0.5
        else:
            raise ValueError(f"Unknown block type {block_out!r}")
    return nn.ModuleList(blocks)


class InputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_module, kernel_size=1, relu_slope=0.2,
                 padding=0):
        super().__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, kernel_size,
                                padding=padding)
        self.activation = nn.LeakyReLU(relu_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class InputBlock2d(InputBlock):

    def __init__(self, in_channels, out_channels, kernel_size=1, relu_slope=0.2, padding=0):
        super().__init__(in_channels, out_channels, EqualizedConv2d,
                         kernel_size=kernel_size, relu_slope=relu_slope, padding=padding)


class InputBlock3d(InputBlock):

    def __init__(self, in_channels, out_channels, kernel_size=1, relu_slope=0.2, padding=0):
        super().__init__(in_channels, out_channels, EqualizedConv3d,
                         kernel_size=kernel_size, relu_slope=relu_slope, padding=padding)


class OutputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_module, kernel_size=1, padding=0,
                 activation=None):
        super().__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, padding=padding)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x


class OutputBlock2d(OutputBlock):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation=None):
        super().__init__(in_channels, out_channels, EqualizedConv2d,
                         kernel_size=kernel_size, padding=padding, activation=activation)


class OutputBlock3d(OutputBlock):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation=None):
        super().__init__(in_channels, out_channels, EqualizedConv3d,
                         kernel_size=kernel_size, padding=padding, activation=activation)


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 relu_slope=0.2, conv_module=EqualizedConv3d,
                 scale_factor=1.0, scale_mode='bilinear'):
        super().__init__()
        self.activation = nn.LeakyReLU(relu_slope)
        self.norm = PixelNorm()

        self.conv1 = conv_module(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = conv_module(out_channels, out_channels, kernel_size, padding=padding)

        self.interpolate = None
        if scale_factor != 1.0 and scale_factor is not None:
            self.interpolate = Interpolate(scale_factor, mode=scale_mode)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.norm(x)

        if self.interpolate:
            x = self.interpolate(x)

        return x


class PreActivationBasicBlock(nn.Module):
    """
    Pre-activation residual block from:

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu_slope=0.2, scale_mode='bilinear',
                 conv_module=EqualizedConv2d):
        super().__init__()
        self.conv1 = conv_module(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.conv2 = conv_module(out_channels, out_channels, kernel_size, padding=1)
        self.shortcut = conv_module(in_channels, out_channels, kernel_size=1, stride=1)

        self.activation = nn.LeakyReLU(relu_slope)
        self.downscale = Interpolate(scale_factor=0.5, mode=scale_mode)

    def forward(self, x):
        shortcut = self.shortcut(self.downscale(x))
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.downscale(x)

        return x + shortcut
