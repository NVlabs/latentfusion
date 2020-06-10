import itertools
import json
import pathlib

import math
import numpy
import os
import random
import torch
import tqdm.auto
from bisect import bisect_right
from functools import partial

import latentfusion


def seed_all(seed):
    torch.random.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def list_arg(cast_type=str, delimiter=','):
    def f(s):
        if len(s) > 0:
            return [cast_type(item) for item in s.split(delimiter)]
        else:
            return []

    return f


def parse_block_str(s):
    if s in {'I', 'U', 'D'}:
        return s

    return int(s)


def parse_block_config(s, delimiter=',', group_delimiter=':'):
    if s.lower() == 'none' or len(s) == 0:
        return []

    _parse_blocks = list_arg(parse_block_str, delimiter=delimiter)

    if group_delimiter in s:
        sections = s.split(group_delimiter)
        return [_parse_blocks(section) for section in sections]
    else:
        return _parse_blocks(s)


def block_config_arg(delimiter=',', group_delimiter=':'):
    return partial(parse_block_config, delimiter=delimiter, group_delimiter=group_delimiter)


def list_choices_arg(valid_choices=None):
    """
    Argparse type that checks for a comma delimited list of choices.
    Args:
        valid_choices (iterable): An iterable of valid choices
    Returns:
        callable: A function which can be passed to the argparse type parameter.
    """

    def fn(s):
        choices = [str(item) for item in s.split(',')]

        for value in choices:
            if valid_choices is not None and value not in valid_choices:
                raise ValueError(f"Invalid choice {value!s}")
        return choices

    return fn


def flatten_list(l):
    return list(itertools.chain.from_iterable(l))


def pbar(*args, **kwargs):
    if latentfusion.is_notebook():
        kwargs['ncols'] = '100%'
    else:
        kwargs['dynamic_ncols'] = True
    return tqdm.auto.tqdm(*args, **kwargs)


def trange(*args, **kwargs):
    if latentfusion.is_notebook():
        kwargs['ncols'] = '100%'
    else:
        kwargs['dynamic_ncols'] = True
    return tqdm.auto.trange(*args, **kwargs)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pathlib.PurePath):
            return str(obj)
        if torch.is_tensor(obj):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def relative_device_id(abs_device_id):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if abs_device_id not in device_ids:
            raise ValueError(f"Device {abs_device_id} is not in CUDA_VISIBLE_DEVICES.")
        return device_ids.index(abs_device_id)
    else:
        return abs_device_id


def absolute_device_id(rel_device_id):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_device_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        return int(cuda_device_ids[rel_device_id])
    else:
        return int(rel_device_id)


class MultiStepMilestoneScheduler(object):

    def __init__(self, initial_value, milestones, gamma):
        self.initial_value = initial_value
        self.milestones = milestones
        self.gamma = gamma

    def get(self, step):
        if self.milestones is None:
            return self.initial_value

        return self.initial_value * self.gamma ** bisect_right(self.milestones, step)


class LinearScheduler(object):

    def __init__(self, initial_value, end_value, num_steps):
        self.initial_value = initial_value
        self.end_value = end_value
        self.num_steps = num_steps

    def get(self, step):
        alpha = step / self.num_steps
        return (1.0 - alpha) * self.initial_value + alpha * self.end_value


class ExponentialScheduler(object):

    def __init__(self, initial_value, final_value, num_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.mean_lifetime = -(num_steps - 1) / math.log(final_value / initial_value)
        self.num_steps = num_steps

    def get(self, step):
        if step >= self.num_steps:
            return self.final_value
        return self.initial_value * math.exp(-step / self.mean_lifetime)
