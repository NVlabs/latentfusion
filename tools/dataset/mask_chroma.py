import argparse
from pathlib import Path

import cv2
import numpy as np
import imageio
import skimage.transform
from skimage import morphology
from tqdm import tqdm

from latentfusion import imutils

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='dataset_dir', type=Path)
    parser.add_argument('--width', type=int, default=2000)
    args = parser.parse_args()

    input_dir = args.dataset_dir / 'Photos'
    color_dir = args.dataset_dir / 'color'
    mask_dir = args.dataset_dir / 'mask'
    color_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)

    for i, image_path in enumerate(tqdm(sorted(input_dir.iterdir()))):
        image = imageio.imread(image_path)
        height, width = image.shape[:2]
        new_width = args.width
        new_height = args.width / width * height
        image = skimage.transform.resize(image, (int(new_height), new_width))
        image = (image * 255.0).astype(np.uint8)
        mask = imutils.mask_chroma(image, (32, 100, 100), (70, 255, 255))
        # mask = morphology.binary_dilation(mask, selem=morphology.disk(5))
        # mask = imutils.grabcut(image, mask)
        # image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # plt.subplot(121)
        # plt.imshow(image_hsv)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()

        imageio.imwrite(color_dir / f'{i:04d}.jpg', image)
        imageio.imwrite(mask_dir / f'{i:04d}.jpg.png', mask.astype(np.uint8) * 255)


if __name__ == '__main__':
    main()
