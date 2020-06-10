import cv2
import numpy as np
import torch
from skimage import morphology


def keep_largest_object(mask):
    labels, num_labels = morphology.label(mask, return_num=True)
    best_mask = None
    best_count = -1
    for i in range(1, num_labels + 1):
        cur_mask = (labels == i)
        cur_count = cur_mask.sum()
        if cur_count > best_count:
            best_mask = cur_mask
            best_count = cur_count

    if best_mask is None:
        return np.zeros_like(mask)

    return best_mask


def mask_chroma(image, hue_min=(40, 65, 65), hue_max=(180, 255, 255),
                use_bgr=False):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV if use_bgr else cv2.COLOR_RGB2HSV)
    hue_min = np.array(hue_min)
    hue_max = np.array(hue_max)
    mask = ~cv2.inRange(image_hsv, hue_min, hue_max)
    mask = morphology.binary_closing(mask, selem=morphology.disk(5))
    return mask


def grabcut(image, fg_init_mask, bg_init_mask=None):
    if image.dtype == np.float32 or image.dtype == np.double:
        image = (image * 255.0).astype(np.uint8)
    # Initialize mask based on sparse pointcloud.
    mask = np.full(image.shape[:2], fill_value=cv2.GC_PR_BGD, dtype=np.uint8)
    mask[fg_init_mask] = cv2.GC_PR_FGD
    if bg_init_mask is not None:
        mask[bg_init_mask] = cv2.GC_BGD

    # Perform grab cut.
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.grabCut(image, mask, None, bg_model, fg_model, 3, cv2.GC_INIT_WITH_MASK)

    # Post process mask.
    out_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    out_mask = morphology.binary_closing(out_mask, selem=morphology.disk(5))

    return out_mask


def mean_color(image, mask):
    return (image * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))


def dilate(labels, iters, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    morphed_labels = []
    for label in labels:
        morphed_labels.append(
            cv2.dilate(label.squeeze(0).numpy(), kernel, iterations=iters))
    return torch.tensor(np.stack(morphed_labels, axis=0),
                        dtype=torch.float32).unsqueeze(1)


def erode(labels, iters, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    morphed_labels = []
    for label in labels:
        morphed_labels.append(
            cv2.erode(label.squeeze(0).numpy(), kernel, iterations=iters))
    return torch.tensor(np.stack(morphed_labels, axis=0),
                        dtype=torch.float32).unsqueeze(1)
