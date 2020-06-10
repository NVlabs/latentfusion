import textwrap
from collections import defaultdict

import torch
import torchnet as tnt
from tabulate import tabulate
from torch.nn import functional as F


def format_code(code):
    return textwrap.indent(
        textwrap.dedent(f"{code!s}\n```".replace('\n', '\n')),
        '    ',
    )


def format_dict_table(dict):
    table = [[k, str(v)] for k, v in dict.items()]
    table = tabulate(table, tablefmt='github', headers=['Key', 'Value'])

    return table


class TensorboardPlotter(object):

    def __init__(self, writer, plot_interval=2000, smooth=5, intervals=None):
        self.writer = writer
        self.run_name = ''
        self.plot_interval = plot_interval
        self.smooth = smooth

        self.live_meters = defaultdict(
            lambda: tnt.meter.MovingAverageValueMeter(smooth))
        self.epoch_meters = defaultdict(tnt.meter.AverageValueMeter)

        self.global_step = 0
        self.last_written = defaultdict(lambda: -1)
        self.intervals = {}
        if intervals is not None:
            self.intervals = intervals

    def reset_meters(self):
        self.live_meters = defaultdict(
            lambda: tnt.meter.MovingAverageValueMeter(self.smooth))
        self.epoch_meters = defaultdict(tnt.meter.AverageValueMeter)
        self.last_written = defaultdict(lambda: -1)

    def set_run_type(self, run_name):
        self.run_name = run_name

    def set_step(self, global_step):
        self.global_step = global_step

    def set_intervals(self, interval_dict):
        self.intervals.update(interval_dict)

    def get_mean(self, tag):
        return self.epoch_meters[tag].value()[0]

    def put_scalar(self, tag, value, global_step=None):
        if global_step is None:
            global_step = self.global_step

        if torch.is_tensor(value):
            value = value.item()

        self.epoch_meters[tag].add(value)
        self.live_meters[tag].add(value)

        # Write if at least plot_interval has passed.
        if global_step - self.last_written[tag] >= self.plot_interval:
            moving_mean, _ = self.live_meters[tag].value()
            self.writer.add_scalar(
                f'live-{tag}/{self.run_name}', moving_mean, global_step)
            self.last_written[tag] = self.global_step

    def put_embeddings(self, tag, features, frames, global_step=None,
                       normalize=False):
        if global_step is None:
            global_step = self.global_step

        if torch.is_tensor(features):
            features = features.detach().cpu()

        if normalize:
            features = features / torch.norm(features, dim=1, keepdim=True)

        center_idx = frames.size(2) // 2
        label_img = frames[:, :, center_idx]
        label_img = F.interpolate(label_img, scale_factor=0.5)

        self.writer.add_embedding(features,
                                  label_img=label_img,
                                  global_step=global_step,
                                  tag=f'{tag}/{self.run_name}')

    def put_audio(self, tag, waveform, global_step=None):
        if global_step is None:
            global_step = self.global_step

        if torch.is_tensor(waveform):
            waveform = waveform.detach().cpu()

        self.writer.add_audio(
            f'{tag}/{self.run_name}', waveform, global_step, sample_rate=16000)

    def put_image(self, tag, image, global_step=None):
        if global_step is None:
            global_step = self.global_step

        if torch.is_tensor(image):
            image = image.detach().cpu()

        self.writer.add_image(f'{tag}/{self.run_name}', image, global_step)

    def put_histogram(self, tag, tensor, global_step=None):
        if global_step is None:
            global_step = self.global_step

        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu()

        self.writer.add_histogram(f'{tag}/{self.run_name}', tensor,
                                  global_step)

    def plot_gradients(self, tag, module, global_step=None):
        if global_step is None:
            global_step = self.global_step

        for name, param in module.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'{tag}/{name}', param.grad, global_step)

    def is_it_time_yet(self, tag, interval=None, global_step=None):
        if interval is None:
            interval = self.intervals[tag]

        if interval < 0:
            return False

        if global_step is None:
            global_step = self.global_step

        if self.last_written[tag] < 0 or global_step - self.last_written[tag] >= interval:
            self.last_written[tag] = self.global_step
            return True

        return False
