import abc
import numbers
import random

import torch
from torch.nn import functional as F
from torchvision.transforms import ColorJitter


def tensor_center_crop(tensor, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = tensor.size(-1), tensor.size(-2)
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return tensor[:, i:i + th, j:j + tw]


def crop(tensor, i, j, h, w):
    return tensor[:, i:i + h, j:j + w]


def get_random_crop_params(input_size, output_size):
    """Get parameters for ``crop`` for a random crop.
    Args:
        input_size (tuple): Expected input size of the crop.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    h, w = input_size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


def get_color_jitter_params(brightness, contrast, saturation, hue):
    jitter = ColorJitter(brightness, contrast, saturation, hue)
    return jitter.get_params(jitter.brightness,
                             jitter.contrast,
                             jitter.saturation,
                             jitter.hue)


def _pad(tensor, padding, value, mode):
    tensor = tensor.unsqueeze(0)
    tensor = F.pad(tensor, padding, value=value, mode=mode)
    return tensor.squeeze(0)


class TensorCrop(abc.ABC):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='zeros'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(tensor, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = tensor.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class TensorRandomCrop(TensorCrop):
    def __call__(self, tensor):
        """
        Args:
            tensor (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding and self.padding > 0:
            padding = [self.padding] * 4
            tensor = _pad(tensor, padding, value=self.fill, mode=self.padding_mode)

        # pad the width if needed
        left_pad = 0
        top_pad = 0
        if self.pad_if_needed and tensor.shape[0] < self.size[1]:
            top_pad = int((1 + self.size[1] - tensor.shape[0]) / 2)
        # pad the height if needed
        if self.pad_if_needed and tensor.shape[1] < self.size[0]:
            left_pad = int((1 + self.size[0] - tensor.shape[1]) / 2)

        tensor = _pad(tensor, [left_pad, left_pad, top_pad, top_pad],
                      value=self.fill, mode=self.padding_mode)

        i, j, h, w = self.get_params(tensor, self.size)

        return tensor[:, i:i + h, j:j + w]


class TensorCenterCrop(object):
    """Crops the given tensor at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return tensor_center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class TensorRandomFlip(object):
    """Vertically flip the given tensor randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, dim, p=0.5):
        self.p = p
        self.dim = dim

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return torch.flip(img, dims=(self.dim,))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class TensorRandomHorizontalFlip(TensorRandomFlip):
    """Horizontally flip the given tensor randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(dim=-1, p=p)


class TensorRandomVerticalFlip(TensorRandomFlip):
    """Horizontally flip the given tensor randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(dim=-2, p=p)
