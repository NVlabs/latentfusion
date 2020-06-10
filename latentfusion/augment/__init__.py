import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

import latentfusion.functional
from . import tensors
from . import masks

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def identity_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :] if x.size(0) > 3 else x),
    ])


def get_transform(hflip_p=0.5, angle_range=(-30.0, 30.0),
                  perspective_scale=0.1, perspective_p=0.5):
    hflip = random.random() < hflip_p
    angle = random.uniform(*angle_range)
    return transforms.Compose([
        transforms.Lambda(lambda x: TF.hflip(x) if hflip else x),
        transforms.Lambda(lambda x: TF.rotate(x, angle, resample=Image.BILINEAR, expand=True)),
        transforms.Resize(256),
        # get_perspective(perspective_scale, perspective_p),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :] if x.size(0) > 3 else x),
    ])


def get_perspective(distortion_scale, p=0.5):
    distort = random.random() < p
    return transforms.Lambda(lambda x: TF.perspective(
        x, *transforms.RandomPerspective.get_params(*x.size, distortion_scale)) if distort else x)


def normalize(tensor):
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def denormalize(tensor):
    return latentfusion.functional.denormalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def gan_normalize(tensor):
    return tensor * 2.0 - 1.0


def gan_denormalize(tensor):
    return ((tensor + 1.0) / 2.0).clamp(0, 1)


def imagenet_normalize(tensor):
    return latentfusion.functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def imagenet_denormalize(tensor):
    return latentfusion.functional.denormalize(tensor, IMAGENET_MEAN, IMAGENET_STD).clamp(0, 1)


def get_mask_extremities(mask):
    if torch.is_tensor(mask):
        nz = torch.nonzero(mask)
        ymin, ymax = nz[:, 0].min(), nz[:, 0].max()
        xmin, xmax = nz[:, 1].min(), nz[:, 1].max()
    else:
        yy, xx = np.where(mask)
        ymin, ymax = yy.min(), yy.max()
        xmin, xmax = xx.min(), xx.max()
    return ymin, ymax, xmin, xmax


def mask_bbox(mask):
    ymin, ymax, xmin, xmax = get_mask_extremities(mask)
    return ymin, xmin, ymax - ymin, xmax - xmin


def mask_center(mask):
    ymin, ymax, xmin, xmax = get_mask_extremities(mask)
    return (ymax + ymin) // 2, (xmax + xmin) // 2


def mask_square_bbox(mask, pad=1):
    ymin, ymax, xmin, xmax = get_mask_extremities(mask)
    size = max(ymax - ymin, xmax - xmin) + pad * 2
    # Round to nearest even number.
    size += size % 2

    ycent, xcent = mask_center(mask)
    return ycent - size // 2, xcent - size // 2, size, size


def crop_bbox(tensor, bbox, size=None, pad=0):
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    ymin, xmin, h, w = bbox

    # Pad before cropping.
    temp_pad = max(h, w) // 2
    tensor = F.pad(tensor, [temp_pad, temp_pad, temp_pad, temp_pad])
    ymin = ymin + temp_pad
    xmin = xmin + temp_pad

    cropped_tensor = tensor[:, ymin:ymin + h, xmin:xmin + w].unsqueeze(0)
    if size is not None:
        cropped_tensor = F.interpolate(cropped_tensor, size - 2 * pad, mode='bilinear')
    cropped_tensor = F.pad(cropped_tensor, [pad, pad, pad, pad])
    return cropped_tensor.squeeze()


def paste_cropped(source, target, target_mask, pad=0):
    target = target.clone()

    if len(source.shape) == 2:
        source = source.unsqueeze(0)
    if len(target.shape) == 2:
        target = target.unsqueeze(0)

    # Unpad source.
    source = source[:, pad:-pad - 1, pad:-pad - 1]

    bbox = mask_square_bbox(target_mask)
    ylo, xlo, h, w = bbox

    source_mask = crop_bbox(target_mask, bbox)
    source = F.interpolate(source.unsqueeze(0), size=(h.item(), w.item()))[0]
    target[:, target_mask > 0] = source[:, source_mask > 0]

    return target


def add_noise_numpy(image, level=0.05):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.8:
        row, col, ch = image.shape
        mean = 0
        var = np.random.rand(1) * level * 256
        sigma = var ** 0.5
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')


def add_noise_depth_cuda(image, level=0.05):
    noise_level = random.uniform(0, level)
    gauss = torch.randn_like(image) * noise_level
    noisy = image + gauss
    return noisy


def add_noise(image, level=0.05):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.8:
        noise_level = random.uniform(0, level)
        gauss = torch.randn_like(image) * noise_level
        noisy = image + gauss
        noisy = torch.clamp(noisy, 0, 1.0)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = torch.zeros((size, size), device=image.device)
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = torch.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = torch.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        kernel_motion_blur = kernel_motion_blur.view(1, 1, size, size)
        kernel_motion_blur = kernel_motion_blur.repeat(image.size(2), 1, 1, 1)

        motion_blur_filter = nn.Conv2d(in_channels=image.size(2),
                                       out_channels=image.size(2),
                                       kernel_size=size,
                                       groups=image.size(2),
                                       bias=False,
                                       padding=int(size / 2))

        motion_blur_filter.weight.data = kernel_motion_blur
        motion_blur_filter.weight.requires_grad = False
        noisy = motion_blur_filter(image.permute(2, 0, 1).unsqueeze(0))
        noisy = noisy.squeeze(0).permute(1, 2, 0)

    return noisy
