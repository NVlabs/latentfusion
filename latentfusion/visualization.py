from collections import namedtuple

import math
from contextlib import contextmanager
from pathlib import Path

import imageio
import numpy as np
import structlog
import tempfile
import torch
import torchvision
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.nn import functional as F
from tqdm.auto import tqdm

logger = structlog.get_logger(__name__)


_colormap_cache = {}


def _build_colormap(name, num_bins=256):
    base = cm.get_cmap(name)
    color_list = base(np.linspace(0, 1, num_bins))
    cmap_name = base.name + str(num_bins)
    colormap = LinearSegmentedColormap.from_list(cmap_name, color_list, num_bins)
    colormap = torch.tensor(colormap(np.linspace(0, 1, num_bins)), dtype=torch.float32)[:, :3]
    return colormap


def get_colormap(name):
    if name not in _colormap_cache:
        _colormap_cache[name] = _build_colormap(name)
    return _colormap_cache[name]


def colorize_tensor(tensor, cmap='magma', cmin=0, cmax=1):
    if len(tensor.shape) > 4:
        tensor = tensor.view(-1, *tensor.shape[-3:])
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(1)
    tensor = tensor.detach().cpu()
    tensor = (tensor - cmin) / (cmax - cmin)
    tensor = (tensor * 255).clamp(0.0, 255.0).long()
    colormap = get_colormap(cmap)
    colorized = colormap[tensor].permute(0, 3, 1, 2)
    return colorized


def colorize_depth(depth):
    if depth.min().item() < -0.1:
        return colorize_tensor(depth.squeeze(1) / 2.0 + 0.5)
    else:
        return colorize_tensor(depth.squeeze(1), cmin=depth.max() - 1.0, cmax=depth.max())


def colorize_numpy(array, to_byte=True):
    array = torch.tensor(array)
    colorized = colorize_tensor(array)
    colorized = colorized.squeeze().permute(1, 2, 0).numpy()
    if to_byte:
        colorized = (colorized * 255).astype(np.uint8)
    return colorized


def make_grid(images, d_real=None, d_fake=None, output_size=128, count=None, row_size=1,
              shuffle=False, stride=1):
    # Ensure that the view dimension is collapsed.
    images = [im.view(-1, *im.shape[-3:]) for im in images if im is not None]

    if count is None:
        count = images[0].size(0)
    # Select `count` random examples.
    if shuffle:
        inds = torch.randperm(images[0].size(0))[::stride][:count]
    else:
        inds = torch.arange(0, images[0].size(0))[::stride][:count]
    images = [im.detach().cpu()[inds] for im in images]

    # Expand 1 channel images to 3 channels.
    images = [im.expand(-1, 3, -1, -1) for im in images]

    # Resize images to output size.
    images = [F.interpolate(im, output_size) for im in images]

    if d_real and d_fake:
        d_real = [t[inds] for t in d_real]
        d_fake = [t[inds] for t in d_fake]

        # Create discriminator score grid.
        d_real = colorize_tensor(
            torch.cat([F.interpolate(h.detach().cpu().clamp(0, 1), output_size // 2)
                       for h in d_real], dim=3).squeeze(1))
        d_fake = colorize_tensor(
            torch.cat([F.interpolate(h.detach().cpu().clamp(0, 1), output_size // 2)
                       for h in d_fake], dim=3).squeeze(1))
        d_grid = torch.cat((d_real, d_fake), dim=2)

        # Create final grid.
        grid = torch.cat((*images, d_grid), dim=3)
    else:
        grid = torch.cat(images, dim=3)

    return torchvision.utils.make_grid(grid, nrow=row_size, padding=2)


def save_video(frames, path, fps=15):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    temp_dir = tempfile.TemporaryDirectory()
    logger.info("saving video", num_frames=len(frames), fps=fps,
                path=path, temp_dir=temp_dir.name)
    try:
        for i, frame in enumerate(tqdm(frames)):
            if torch.is_tensor(frame):
                frame = frame.permute(1, 2, 0).detach().cpu().numpy()
            frame_path = Path(temp_dir.name, f'{i:08d}.jpg')
            imageio.imsave(frame_path, (frame * 255).astype(np.uint8))

        video = ImageSequenceClip(temp_dir.name, fps=fps)
        video.write_videofile(str(path), preset='ultrafast', fps=fps)
    finally:
        temp_dir.cleanup()


def save_frames(frames, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, frame in enumerate(tqdm(frames)):
        imageio.imsave(save_dir / f'{i:04d}.jpg', (frame * 255).astype(np.uint8))


def batch_grid(batch, nrow=4):
    batch = batch.view(-1, *batch.shape[-3:])
    grid = torchvision.utils.make_grid(batch.detach().cpu(), nrow=nrow)
    return grid


@contextmanager
def plot_to_tensor(out_tensor, dpi=100):
    """
    A context manager that yields an axis object. Plots will be copied to `out_tensor`.
    The output tensor should be a float32 tensor.

    Usage:
        ```
        tensor = torch.tensor(3, 480, 640)
        with plot_to_tensor(tensor) as ax:
            ax.plot(...)
        ```

    Args:
        out_tensor: tensor to write to
        dpi: the DPI to render at
    """
    height, width = out_tensor.shape[-2:]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.tight_layout(pad=0)

    yield ax

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    out_tensor.copy_((torch.tensor(data).float() / 255.0).permute(2, 0, 1))


@contextmanager
def plot_to_array(height, width, rows=1, cols=1, dpi=100):
    """
    A context manager that yields an axis object. Plots will be copied to `out_tensor`.
    The output tensor should be a float32 tensor.

    Usage:
        ```
        with plot_to_array(480, 640, 2, 2) as (fig, axes, out_image):
            axes[0][0].plot(...)
        ```

    Args:
        height: the height of the canvas
        width: the width of the canvas
        rows: the number of axis rows
        cols: the number of axis columns
        dpi: the DPI to render at
    """
    out_array = np.empty((height, width, 3), dtype=np.uint8)
    fig, axes = plt.subplots(rows, cols, figsize=(width / dpi, height / dpi), dpi=dpi)

    yield fig, axes, out_array

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    np.copyto(out_array, data)


def apply_mask_gray(image, mask):
    image = (image - 0.5) * 2.0
    image = image * mask
    return (image + 1.0) / 2.0


def show_batch(batch, nrow=16, title=None, padding=2, pad_value=1):
    batch = batch.view(-1, *batch.shape[-3:])
    grid = torchvision.utils.make_grid(batch.detach().cpu(),
                                       nrow=nrow,
                                       padding=padding,
                                       pad_value=pad_value).permute(1, 2, 0)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.imshow(grid)


def plot_image_batches(path, images, num_cols=None, size=5):
    titles, images = list(zip(*images))

    num_images = len(images)
    num_batch = max(len(x) for x in images if x is not None)
    grid_row_size = int(math.ceil(math.sqrt(num_batch)))

    if num_cols is None:
        num_cols = num_images
    num_rows = int(math.ceil(len(images) / num_cols))

    aspect_ratio = images[0].shape[-1] / images[0].shape[-2]
    width = num_cols * size * aspect_ratio
    height = num_rows * (size + 1)  # Room for titles.

    fig = plt.figure(figsize=(width, height))
    for i in range(num_images):
        if images[i] is None:
            continue
        plt.subplot(num_rows, num_cols, i+1)
        show_batch(images[i],
                   nrow=min(len(images[i]), grid_row_size),
                   title=titles[i])

    fig.tight_layout()
    fig.savefig(path)
    plt.close('all')


def plot_grid(num_cols, figsize, plots):
    if num_cols is None:
        num_cols = len(plots)
    num_rows = int(math.ceil(len(plots) / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        if i >= len(plots) or plots[i] is None:
            ax.axis('off')
            continue
        plot = plots[i]
        args = plot.args if plot.args else []
        kwargs = plot.kwargs if plot.kwargs else {}
        if isinstance(plot.func, str):
            getattr(ax, plot.func)(*args, **kwargs)
        else:
            plot.func(*args, **kwargs, ax=ax)
        ax.set_title(plot.title)
        if plot.params:
            for param_key, param_value in plot.params.items():
                getattr(ax, f'set_{param_key}')(param_value)
    # fig.set_facecolor('white')
    fig.tight_layout()

    return fig


def depth_to_disparity(depth):
    depth[depth > 0] = 1/depth[depth > 0]
    valid = depth[depth > 0]
    cmin = valid.min()
    cmax = valid.max()
    return (depth - cmin) / (cmax - cmin)


def depth_to_disparity(depth):
    depth[depth > 0] = 1/depth[depth > 0]
    valid = depth[depth > 0]
    cmin = valid.min()
    cmax = valid.max()
    return (depth - cmin) / (cmax - cmin)



Plot = namedtuple('Plot', ['title', 'args', 'kwargs', 'params', 'func'],
                  defaults=[None, None, None, 'plot'])

