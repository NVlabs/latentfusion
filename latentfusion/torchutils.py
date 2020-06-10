import time

import abc
import copy
import random
from contextlib import contextmanager
from functools import partial
from itertools import chain

import structlog
import torch
import torch.autograd
from torch import nn
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.utils.data import Sampler, DataLoader

logger = structlog.get_logger(__name__)


def dict_to(d, device):
    return {k: v.to(device) for k, v in d.items()}


def deparameterize_module(module):
    module = module.clone()
    for name, parameter in module.named_parameters():
        parameter = parameter.detach()
        setattr(module, name, parameter)
    return module


@contextmanager
def manual_seed(seed):
    torch_state = torch.get_rng_state()
    torch.manual_seed(seed)
    yield
    torch.set_rng_state(torch_state)


def module_device(module):
    return next(module.parameters()).device


def save_checkpoint(save_dir, name, state):
    if not save_dir.exists():
        logger.info("creating directory", path=save_dir)
        save_dir.mkdir(parents=True)

    path = save_dir / f'{name}.pth'
    logger.info("saving checkpoint", name=name, path=path)

    with path.open('wb') as f:
        torch.save(state, f)


def save_if_better(save_dir, state, meters, key, bigger_is_better=False):
    best_key = f'best-{key}'.replace('/', '-')
    if best_key not in state:
        state[best_key] = -1 if bigger_is_better else float('inf')

    if bigger_is_better:
        better = meters[key].mean >= state[best_key]
    else:
        better = meters[key].mean <= state[best_key]

    if better:
        state[best_key] = meters[key].mean
        save_checkpoint(save_dir, best_key, state)


class DeterministicShuffledSampler(Sampler):
    """Shuffles the dataset once and then samples deterministically."""

    def __init__(self, data_source, replacement=False, num_samples=None):
        super().__init__(data_source)

        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is not None and replacement is False:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(
                self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        n = len(self.data_source)
        if self.replacement:
            self.permutation = torch.randint(
                high=n, size=(self.num_samples,), dtype=torch.int64).tolist()
        else:
            self.permutation = torch.randperm(n).tolist()

    def __iter__(self):
        return iter(self.permutation)

    def __len__(self):
        return len(self.data_source)


class Scatterable(abc.ABC):
    """
    A mixin to make an object scatterable across GPUs for data-parallelism.

    The object should be serialized to a dictionary in the `to_kwargs` method. Each entry should be
    a scatterable tensor object.

    The `from_kwargs` method should be able to take the dictionary format and reconstruct the original
    class object.
    """

    @classmethod
    @abc.abstractmethod
    def to_kwargs(self):
        pass

    @classmethod
    @abc.abstractmethod
    def from_kwargs(cls, kwargs):
        pass


class MyDataParallel(nn.DataParallel):
    """
    A Scatterable-aware data parallel class.
    """

    def scatter(self, inputs, kwargs, device_ids):
        _inputs = []
        _kwargs = {}
        input_constructors = {}
        kwargs_constructors = {}
        for i, item in enumerate(inputs):
            if isinstance(item, Scatterable):
                input_constructors[i] = item.from_kwargs
                _inputs.append(item.to_kwargs())
            else:
                input_constructors[i] = lambda x: x
                _inputs.append(item)

        for key, item in kwargs.items():
            if isinstance(item, Scatterable):
                kwargs_constructors[key] = item.from_kwargs
                _kwargs[key] = item.to_kwargs()
            else:
                kwargs_constructors[key] = lambda x: x
                _kwargs[key] = item

        _inputs, _kwargs = scatter_kwargs(_inputs, _kwargs, device_ids, dim=self.dim)

        _inputs = [
            [input_constructors[i](item) for i, item in enumerate(_input)]
            for _input in _inputs
        ]
        _kwargs = [
            {k: kwargs_constructors[k](item) for k, item in _kwarg.items()}
            for _kwarg in _kwargs
        ]

        return _inputs, _kwargs


class ListSampler(Sampler):
    r"""Samples given elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
        indices (iterable): list of indices to sample
    """

    def __init__(self, data_source, indices):
        super().__init__(data_source)
        if indices is None:
            self.indices = list(range(len(data_source)))
        else:
            self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ShuffledSubsetSampler(Sampler):
    r"""Samples given elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
        indices (iterable): list of indices to sample
    """

    def __init__(self, data_source, indices):
        super().__init__(data_source)
        self.indices = copy.deepcopy(indices)
        random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class IndexedDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, num_workers=4, indices=None, drop_last=False,
                 pin_memory=False, shuffle=False):
        if shuffle:
            sampler = ShuffledSubsetSampler(dataset, indices)
        else:
            sampler = ListSampler(dataset, indices)

        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=pin_memory)


class SequentialDataLoader(IndexedDataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, shuffle=False)


class _InfiniteSampler(object):
    """ Sampler that repeats forever.

    Hack to force dataloader to use same workers.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class WorkerPreservingDataLoader(DataLoader):
    """
    Hack to force dataloader to use same workers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = _InfiniteSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


@contextmanager
def profile():
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        yield
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


@contextmanager
def measure_time(s: str):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    print(f"[{s}] elapsed: {time.time() - start:.02f}")

