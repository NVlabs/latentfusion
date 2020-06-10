"""
PGGAN based encoder-decoder implementation.

References:
    https://arxiv.org/abs/1710.10196
    https://github.com/google/neural_rerendering_in_the_wild
    https://github.com/shanexn/pytorch-pggan
    https://github.com/facebookresearch/pytorch_GAN_zoo
"""

from .discriminator import *
from .generator import *
