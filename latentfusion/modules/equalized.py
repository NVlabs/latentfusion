import math
from functools import reduce
from operator import mul

import torch
from torch import nn


# class Equalized(nn.Module):
#
#     def __init__(self, module, bias_zero_init=True):
#         r"""
#         equalized (bool): if True use He's constant to normalize at runtime.
#         bias_zero_init (bool): if true, bias will be initialized to zero
#         """
#         super().__init__()
#
#         self.multiplier = get_he_constant(module.weight)
#         self.module = module
#
#         module.weight.data.normal_()
#         if hasattr(module, 'bias') and bias_zero_init:
#             module.bias.data.zero_()
#
#         # Move weight parameter so that we can override the original.
#         module.register_parameter('weight_orig', nn.Parameter(module.weight.data))
#         del module.weight
#
#     def forward(self, *args, **kwargs):
#         # Set weight to equalized one.
#         self.module.weight = self.multiplier * self.module.weight_orig
#         return self.module(*args, **kwargs)


class Equalized(nn.Module):

    def __init__(self, module, equalized=True, lr_scale=1.0, bias=True):
        r"""
        equalized (bool): if True use He's constant to normalize at runtime.
        bias_zero_init (bool): if true, bias will be initialized to zero
        """
        super().__init__()

        assert module.bias is None

        self.module = module
        self.equalized = equalized

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.module.out_channels))

        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lr_scale
            self.weight = self.get_he_constant() * lr_scale

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.weight
        dims = [1 for _ in x.shape]
        dims[1] = -1
        x = x + self.bias.view(*dims)
        return x

    def get_he_constant(self):
        r"""
        Get He's constant for the given layer
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
        """
        size = self.module.weight.size()
        fan_in = reduce(mul, size[1:], 1)

        return math.sqrt(2.0 / fan_in)


class EqualizedConv2d(Equalized):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding: int = 0, padding_mode='zeros', **kwargs):
        module = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False,
                           padding=padding, padding_mode=padding_mode)
        super().__init__(module, **kwargs)


class EqualizedConv3d(Equalized):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding: int = 0, padding_mode='zeros', **kwargs):
        module = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=False,
                           padding=padding, padding_mode=padding_mode)
        super().__init__(module, **kwargs)


class EqualizedLinear(Equalized):

    def __init__(self, in_channels, out_channels, **kwargs):
        module = nn.Linear(in_channels, out_channels, bias=False)
        super().__init__(module, **kwargs)
