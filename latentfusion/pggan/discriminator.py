import torch
from torch import nn
from torch.nn import functional as F

from latentfusion.modules import EqualizedConv2d

__all__ = ['Discriminator', 'MultiScaleDiscriminator']


def minibatch_mean_variance(x, eps=1e-8):
    mean = torch.mean(x, dim=0, keepdim=True)
    vals = torch.sqrt(torch.mean((x - mean) ** 2, dim=0) + eps)
    vals = torch.mean(vals)
    return vals


class MinibatchStatsConcat(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean_var = minibatch_mean_variance(x, self.eps)
        # Expand mean variance to spatial dimension of x and concatenate to end of channel
        # dimension.
        mean_var = mean_var.view(1, 1, 1, 1).expand(x.size(0), -1, x.size(2), x.size(3))
        return torch.cat((x, mean_var), dim=1)


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, norm=None,
                 minibatch_stats=False, relu_slope=0.2, padding=0):
        super().__init__()

        self.minibatch_stats = None
        if minibatch_stats:
            self.minibatch_stats = MinibatchStatsConcat()
            in_channels += 1

        self.norm = None
        if norm:
            self.norm = norm(out_channels)

        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                    padding=padding)
        self.activation = nn.LeakyReLU(relu_slope)

    def forward(self, x):
        if self.minibatch_stats is not None:
            x = self.minibatch_stats(x)
        x = self.conv(x)
        # Original PatchGAN seems to have norm before activation.
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, in_channels, block_config=None):
        super().__init__()

        if block_config is None:
            block_config = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.block_config = block_config

        self.blocks = nn.ModuleList()

        # Add input block.
        self.blocks.append(
            DiscriminatorBlock(in_channels, block_config[0], kernel_size=4, stride=2,
                               padding=1))

        # Add intermediate blocks.
        for block_id, (block_in, block_out) in enumerate(zip(block_config[:-1], block_config[1:])):
            is_last = (block_id == len(block_config) - 2)
            stride = 1 if is_last else 2
            block = DiscriminatorBlock(block_in, block_out, kernel_size=4, stride=stride,
                                       norm=nn.InstanceNorm2d, minibatch_stats=is_last,
                                       padding=1)
            self.blocks.append(block)

        self.output_block = EqualizedConv2d(block_config[-1], 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, mask=None):
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            x = mask * x

        for block in self.blocks:
            x = block(x)
        x = self.output_block(x)
        return x


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, in_channels, block_config=None, num_scales=3):
        super().__init__()

        self.in_channels = in_channels
        self.block_config = block_config
        self.num_scales = num_scales

        self.discriminators = nn.ModuleList()
        for scale in range(num_scales):
            self.discriminators.append(Discriminator(in_channels, block_config))

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def create_checkpoint(self):
        return {
            'args': {
                'in_channels': self.in_channels,
                'block_config': self.block_config,
                'num_scales': self.num_scales,
            },
            'state_dict': self.cpu().state_dict(),
        }

    def forward(self, x, mask=None):
        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        responses = []
        for scale, discriminator in enumerate(self.discriminators):
            responses.append(discriminator(x, mask))
            if scale != len(self.discriminators) - 1:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                if mask is not None:
                    mask = F.interpolate(mask, scale_factor=0.5)

        return responses
