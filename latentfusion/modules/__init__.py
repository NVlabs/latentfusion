import torch
from torch import nn
from torch.nn import functional as F

from latentfusion.functional import extract_features


class PixelNorm(nn.Module):
    """
    Mentioned in '4.2 PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR'
    'Local response normalization'
    """

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class Interpolate(nn.Module):

    __constants__ = ['scale_factor']

    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = None
        if mode == 'bilinear' or mode == 'trilinear':
            self.align_corners = False

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)

    def extra_repr(self):
        return f"scale_factor={self.scale_factor}"


class LayerExtractor(nn.Module):

    def __init__(self, submodule: nn.Module, layers):
        super().__init__()
        self.submodule = submodule
        self.layers = [str(l) for l in layers]

    def forward(self, x):
        return extract_features(x, self.submodule, self.layers)


from .equalized import EqualizedConv2d, EqualizedConv3d, EqualizedLinear
