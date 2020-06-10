"""
Data augmentation for masks.

Adapted from https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py
"""
import abc
import random

import cv2
import numpy as np
import torch


def _build_matrix_of_indices(height, width):
    """ Builds a [height, width, 2] numpy array containing coordinates.

        @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)


def _mask_to_tight_box_numpy(mask):
    """ Return bbox given mask

        @param mask: a [H x W] numpy array
    """
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


def _mask_to_tight_box_pytorch(mask):
    """ Return bbox given mask

        @param mask: a [H x W] torch tensor
    """
    a = torch.nonzero(mask)
    bbox = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(a[:, 1]), torch.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


def mask_to_tight_box(mask):
    if type(mask) == torch.Tensor:
        return _mask_to_tight_box_pytorch(mask)
    elif type(mask) == np.ndarray:
        return _mask_to_tight_box_numpy(mask)
    else:
        raise Exception("Data type {} not understood for mask_to_tight_box...".format(type(mask)))


def _translate(img, tx, ty, interpolation=cv2.INTER_LINEAR):
    """ Translate img by tx, ty

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    M = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)


def _rotate(img, angle, center=None, interpolation=cv2.INTER_LINEAR):
    """ Rotate img <angle> degrees counter clockwise w.r.t. center of image

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    if center is None:
        center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)


def random_morphological_transform(label, alpha=1.0, beta=19.0, max_iters=3):
    """
    Randomly erode/dilate the label
    """
    # Sample whether we do erosion or dilation, and kernel size for that
    x_min, y_min, x_max, y_max = mask_to_tight_box(label)
    sidelength = np.mean([x_max - x_min, y_max - y_min])

    morphology_kernel_size = 0
    num_ksize_tries = 0
    while morphology_kernel_size == 0:
        if num_ksize_tries >= 50:  # 50 tries for this
            return label

        dilation_percentage = np.random.beta(alpha, beta)
        morphology_kernel_size = int(round(sidelength * dilation_percentage))

        num_ksize_tries += 1

    iterations = np.random.randint(1, max_iters + 1)

    # Erode/dilate the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
    if np.random.rand() < 0.5:
        morphed_label = cv2.erode(label, kernel, iterations=iterations)
    else:
        morphed_label = cv2.dilate(label, kernel, iterations=iterations)

    return morphed_label


def random_ellipses(label, num_ellipses_mean=50, gamma_base_shape=1.0, gamma_base_scale=1.0,
                    size_percentage=0.025):
    """
    Randomly add/drop a few ellipses in the mask

    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    """
    H, W = label.shape

    new_label = label.copy()

    # Sample number of ellipses to include/dropout
    num_ellipses = np.random.poisson(num_ellipses_mean)

    # Sample ellipse centers by sampling from Gaussian at object center
    pixel_indices = _build_matrix_of_indices(H, W)
    h_idx, w_idx = np.where(new_label)
    mu = np.mean(pixel_indices[h_idx, w_idx, :], axis=0)  # Shape: [2]. y_center, x_center
    sigma = 2 * np.cov(pixel_indices[h_idx, w_idx, :].T)  # Shape: [2 x 2]
    if np.any(np.isnan(mu)) or np.any(np.isnan(sigma)):
        print(mu, sigma, h_idx, w_idx)
    ellipse_centers = np.random.multivariate_normal(mu, sigma, size=num_ellipses)  # Shape: [num_ellipses x 2]
    ellipse_centers = np.round(ellipse_centers).astype(int)

    # Sample ellipse radii and angles
    x_min, y_min, x_max, y_max = mask_to_tight_box(new_label)
    scale_factor = max(x_max - x_min, y_max - y_min) * size_percentage  # Mean of gamma r.v.
    x_radii = np.random.gamma(gamma_base_shape * scale_factor, gamma_base_scale,
                              size=num_ellipses)
    y_radii = np.random.gamma(gamma_base_shape * scale_factor, gamma_base_scale,
                              size=num_ellipses)
    angles = np.random.randint(0, 360, size=num_ellipses)

    # Dropout ellipses
    for i in range(num_ellipses):
        center = ellipse_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # include or dropout the ellipse
        mask = np.zeros_like(new_label)
        mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360,
                           color=1, thickness=-1)
        if np.random.rand() < 0.5:
            new_label[mask == 1] = 0  # Drop out ellipse
        else:
            new_label[mask == 1] = 1  # Add ellipse

    return new_label


def random_translation(label, alpha=1.0, beta=19.0, percentage_min=0.05):
    """
    Randomly translate mask
    """

    # Get tight bbox of mask
    x_min, y_min, x_max, y_max = mask_to_tight_box(label)
    sidelength = max(x_max - x_min, y_max - y_min)

    # sample translation pixels
    translation_percentage = np.random.beta(alpha, beta)
    translation_percentage = max(translation_percentage, percentage_min)
    translation_max = int(round(translation_percentage * sidelength))
    translation_max = max(translation_max, 1)  # To make sure things don't error out

    tx = np.random.randint(-translation_max, translation_max)
    ty = np.random.randint(-translation_max, translation_max)

    translated_label = _translate(label, tx, ty, interpolation=cv2.INTER_NEAREST)

    return translated_label


def random_rotation(label, angle_max=10):
    """
    Randomly rotate mask
    """
    H, W = label.shape

    # Rotate about center of box
    pixel_indices = _build_matrix_of_indices(H, W)
    h_idx, w_idx = np.where(label)
    mean = np.mean(pixel_indices[h_idx, w_idx, :], axis=0)  # Shape: [2]. y_center, x_center

    # Sample an angle
    applied_angle = np.random.uniform(-angle_max, angle_max)

    rotated_label = _rotate(label, applied_angle, center=tuple(mean[::-1]), interpolation=cv2.INTER_NEAREST)

    return rotated_label


def random_cut(label, percentage_min=0.25, percentage_max=0.5):
    """
    Randomly cut part of mask
    """

    cut_label = label.copy()

    # Sample cut percentage
    cut_percentage = np.random.uniform(percentage_min, percentage_max)

    x_min, y_min, x_max, y_max = mask_to_tight_box(label)
    if np.random.rand() < 0.5:  # choose width

        sidelength = x_max - x_min
        if np.random.rand() < 0.5:  # from the left
            x = int(round(cut_percentage * sidelength)) + x_min
            cut_label[y_min:y_max + 1, x_min:x] = 0
        else:  # from the right
            x = x_max - int(round(cut_percentage * sidelength))
            cut_label[y_min:y_max + 1, x:x_max + 1] = 0

    else:
        # choose height
        sidelength = y_max - y_min
        if np.random.rand() < 0.5:  # from the top
            y = int(round(cut_percentage * sidelength)) + y_min
            cut_label[y_min:y, x_min:x_max + 1] = 0
        else:  # from the bottom
            y = y_max - int(round(cut_percentage * sidelength))
            cut_label[y:y_max + 1, x_min:x_max + 1] = 0

    return cut_label


def random_add(label, percentage_min=0.1, percentage_max=0.4):
    """
    Randomly add part of mask
    """
    added_label = label.copy()

    # Sample add percentage
    add_percentage = np.random.uniform(percentage_min, percentage_max)

    x_min, y_min, x_max, y_max = mask_to_tight_box(label)

    # Sample translation from center
    translation_percentage_x = np.random.uniform(0, 2 * add_percentage)
    tx = int(round((x_max - x_min) * translation_percentage_x))
    translation_percentage_y = np.random.uniform(0, 2 * add_percentage)
    ty = int(round((y_max - y_min) * translation_percentage_y))

    if np.random.rand() < 0.5:  # choose x direction

        sidelength = x_max - x_min
        ty = np.random.choice([-1, 1]) * ty  # mask will be moved to the left/right. up/down doesn't matter

        if np.random.rand() < 0.5:  # mask copied from the left.
            x = int(round(add_percentage * sidelength)) + x_min
            try:
                temp = added_label[y_min + ty: y_max + 1 + ty, x_min - tx: x - tx]
                added_label[y_min + ty: y_max + 1 + ty, x_min - tx: x - tx] = np.logical_or(
                    temp, added_label[y_min: y_max + 1, x_min: x])
            except ValueError:  # indices were out of bounds
                return None
        else:  # mask copied from the right
            x = x_max - int(round(add_percentage * sidelength))
            try:
                temp = added_label[y_min + ty: y_max + 1 + ty, x + tx: x_max + 1 + tx]
                added_label[y_min + ty: y_max + 1 + ty, x + tx: x_max + 1 + tx] = np.logical_or(
                    temp, added_label[y_min: y_max + 1, x: x_max + 1])
            except ValueError:  # indices were out of bounds
                return None

    else:  # choose y direction
        sidelength = y_max - y_min
        tx = np.random.choice([-1, 1]) * tx  # mask will be moved up/down. lef/right doesn't matter

        if np.random.rand() < 0.5:  # from the top
            y = int(round(add_percentage * sidelength)) + y_min
            try:
                temp = added_label[y_min - ty: y - ty, x_min + tx: x_max + 1 + tx]
                added_label[y_min - ty: y - ty, x_min + tx: x_max + 1 + tx] = np.logical_or(
                    temp, added_label[y_min: y, x_min: x_max + 1])
            except ValueError:  # indices were out of bounds
                return None
        else:  # from the bottom
            y = y_max - int(round(add_percentage * sidelength))
            try:
                temp = added_label[y + ty: y_max + 1 + ty, x_min + tx: x_max + 1 + tx]
                added_label[y + ty: y_max + 1 + ty, x_min + tx: x_max + 1 + tx] = np.logical_or(
                    temp, added_label[y: y_max + 1, x_min: x_max + 1])
            except ValueError:  # indices were out of bounds
                return None

    return added_label


class _RandomTransform(abc.ABC):
    def __init__(self, p, max_tries=10):
        self.p = p
        self.max_tries = max_tries

    def __call__(self, mask):
        if random.random() > self.p:
            return mask

        for num_tries in range(self.max_tries):
            try:
                new_mask = self.run(mask.numpy().astype(np.uint8))
            except ValueError:
                continue
            if self._check_valid(mask, new_mask):
                return torch.tensor(new_mask, dtype=torch.bool)

        return mask

    @classmethod
    def _check_valid(cls, input_mask, mask):
        if mask is None:
            return False

        if mask.shape != input_mask.shape:
            return False

        if np.isnan(mask).sum() > 0:
            return False

        return ((np.count_nonzero(mask) / mask.size > 0.001)
                and (np.count_nonzero(mask) / mask.size < 0.98))

    @abc.abstractmethod
    def run(self, mask):
        pass


class RandomMorphologicalTransform(_RandomTransform):

    def __init__(self, alpha=1.0, beta=19.0, max_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.max_iters = max_iters

    def run(self, mask):
        return random_morphological_transform(mask, self.alpha, self.beta, self.max_iters)


class RandomEllipses(_RandomTransform):

    def __init__(self, num_ellipses_mean=50, gamma_base_shape=1.0, gamma_base_scale=1.0,
                 size_percentage=0.025, **kwargs):
        super().__init__(**kwargs)
        self.num_ellipses_mean = num_ellipses_mean
        self.gamma_base_shape = gamma_base_shape
        self.gamma_base_scale = gamma_base_scale
        self.size_percentage = size_percentage

    def run(self, mask):
        return random_ellipses(mask, self.num_ellipses_mean, self.gamma_base_shape, self.gamma_base_scale,
                               self.size_percentage)


class RandomTranslation(_RandomTransform):
    def __init__(self, alpha=1.0, beta=19.0, percentage_min=0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.percentage_min = percentage_min

    def run(self, mask):
        return random_translation(mask, self.alpha, self.beta, self.percentage_min)


class RandomRotation(_RandomTransform):
    def __init__(self, angle_max=10, **kwargs):
        super().__init__(**kwargs)
        self.angle_max = angle_max

    def run(self, mask):
        return random_rotation(mask, self.angle_max)


class RandomCut(_RandomTransform):
    def __init__(self, percentage_min=0.25, percentage_max=0.5, **kwargs):
        super().__init__(**kwargs)
        self.percentage_min = percentage_min
        self.percentage_max = percentage_max

    def run(self, mask):
        return random_cut(mask, self.percentage_min, self.percentage_max)


class RandomAdd(_RandomTransform):
    def __init__(self, percentage_min=0.1, percentage_max=0.4, **kwargs):
        super().__init__(**kwargs)
        self.percentage_min = percentage_min
        self.percentage_max = percentage_max

    def run(self, mask):
        return random_add(mask, self.percentage_min, self.percentage_max)
