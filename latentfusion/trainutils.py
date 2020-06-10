import abc
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import structlog
import torch
import torch.utils.data
import torchvision
from torch import nn
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from latentfusion import tbutils
from latentfusion import torchutils, utils
from latentfusion.losses import HardPixelLoss, PerceptualLoss
from latentfusion.tbutils import TensorboardPlotter
from latentfusion.utils import MyEncoder, list_arg

logger = structlog.get_logger(__name__)


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--num-epochs', default=10000, type=int,
                        help="How many epochs to train for.")
    parser.add_argument('--batch-size', default=4, type=int,
                        help="The batch size.")
    parser.add_argument('--batch-groups', default=1, type=int,
                        help="Number of groups to split batch into.")
    parser.add_argument('--batches-per-epoch', default=1600, type=int,
                        help="How many batches one 'epoch' is.")

    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--data-parallel', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'adamw'], default='adam')
    parser.add_argument('--use-amp', action='store_true')

    # Visualization parameters.
    parser.add_argument('--plot-interval', type=int, default=20)
    parser.add_argument('--show-interval', type=int, default=25)
    parser.add_argument('--histogram-interval', type=int, default=-1)
    parser.add_argument('--grad-histogram-interval', type=int, default=-1)
    parser.add_argument('--save-interval', type=int, default=20)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser):
    is_resume = '--resume' in sys.argv
    parser.add_argument('--dataset-type', choices=['shapenet'], default='shapenet')
    parser.add_argument('--dataset-path', type=Path, required=not is_resume)
    parser.add_argument('--textures-path', type=Path, required=False)
    parser.add_argument('--color-background-path', type=Path, required=False)
    parser.add_argument('--depth-background-path', type=Path, required=False)
    parser.add_argument('--dataset-gpu-id', type=int, default=0)
    parser.add_argument('--dataset-x-bound', default='-0.4,0.4', type=list_arg(float))
    parser.add_argument('--dataset-y-bound', default='-0.2,0.2', type=list_arg(float))
    parser.add_argument('--dataset-z-bound', default='1.5,3.0', type=list_arg(float))
    parser.add_argument('--dataset-size-jitter', default='0.5,1.0', type=list_arg(float))
    parser.add_argument('--blacklist-categories', default='', type=list_arg(str))
    parser.add_argument('--depth-noise-level', default=0.00, type=float)
    parser.add_argument('--color-noise-level', default=0.05, type=float)
    parser.add_argument('--camera-translation-noise', default=0.00, type=float)
    parser.add_argument('--camera-rotation-noise', default=0.00, type=float)
    parser.add_argument('--use-constrained-cameras', action='store_true')
    parser.add_argument('--mask-noise-p', default=0.5, type=float)
    parser.add_argument('--crop-random-background', action='store_true')
    parser.add_argument('--color-random-background', action='store_true')
    parser.add_argument('--depth-random-background', action='store_true')
    parser.add_argument('--model-ids', type=utils.list_choices_arg())
    parser.add_argument('--input-size', default=128, type=int)

    return parser


def load_checkpoint_args(kwargs, args=None):
    # For backward compatibility.
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = kwargs['batch_objects']

    if args:
        for key, value in vars(args).items():
            if key not in kwargs:
                kwargs[key] = value

        if args.override:
            for arg_name in args.override:
                new_value = vars(args)[arg_name]
                logger.info('overriding checkpoint argument',
                            name=arg_name, old_value=kwargs.get(arg_name), new_value=new_value)
                kwargs[arg_name] = new_value

    del kwargs['override']

    return kwargs


def get_optimizer(parameters, name, lr):
    if name == 'adam':
        return optim.Adam(parameters, lr=lr, betas=(0.0, 0.99))
    if name == 'adamw':
        return optim.AdamW(parameters, lr=lr, betas=(0.0, 0.99))
    elif name == 'sgd':
        return optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f'Unknown optimizer {name!r}')


def get_recon_criterion(loss_type, k=2000):
    if loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'hard_l1':
        return HardPixelLoss(nn.L1Loss, k=k)
    elif loss_type == 'hard_smooth_l1':
        return HardPixelLoss(nn.SmoothL1Loss, k=k)
    elif loss_type == 'perceptual_vgg16':
        vgg = torchvision.models.vgg16(pretrained=True).eval()
        perceptual_loss_base = vgg.features
        layers = ['3', '8', '15', '22', '27']
        layer_weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1]
        return PerceptualLoss(perceptual_loss_base, layers, layer_weights, reduction=None)
    elif loss_type == 'binary_cross_entropy':
        return nn.BCEWithLogitsLoss(reduction='none')
    else:
        raise ValueError(f"Unknown recon_loss_type {loss_type!r}.")


def get_dataset(dataset_type, device_id, kwargs):
    logger.info("loading dataset",
                type=dataset_type,
                path=kwargs['dataset_path'],
                x_bound=kwargs['dataset_x_bound'],
                y_bound=kwargs['dataset_y_bound'],
                z_bound=kwargs['dataset_z_bound'],
                size_jitter=kwargs['dataset_size_jitter'],
                model_ids=kwargs['model_ids'])

    if dataset_type == 'shapenet':
        from latentfusion.datasets.training.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(
            kwargs['dataset_path'],
            textures_dir=kwargs['textures_path'],
            num_input_views=kwargs['num_input_views'],
            num_output_views=kwargs['num_output_views'],
            x_bound=kwargs['dataset_x_bound'],
            y_bound=kwargs['dataset_y_bound'],
            z_bound=kwargs['dataset_z_bound'],
            size_jitter=kwargs['dataset_size_jitter'],
            depth_noise_level=kwargs.get('depth_noise_level', 0.0),
            color_noise_level=kwargs.get('color_noise_level', 0.0),
            camera_translation_noise=kwargs.get('camera_translation_noise', 0.0),
            camera_rotation_noise=kwargs.get('camera_rotation_noise', 0.0),
            mask_noise_p=kwargs.get('mask_noise_p', 0.0),
            color_background_dir=kwargs.get('color_background_path'),
            depth_background_dir=kwargs.get('depth_background_path'),
            color_random_background=kwargs.get('color_random_background', False),
            depth_random_background=kwargs.get('depth_random_background', False),
            use_constrained_cameras=kwargs.get('use_constrained_cameras', False),
            blacklist_categories=kwargs.get('blacklist_categories', [])
        )
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    return dataset


class Trainer(abc.ABC):

    def __init__(self, kwargs, *,
                 name,
                 save_dir,
                 modules,
                 train_modules,
                 dataset,
                 device=torch.device("cuda:0"),
                 meter_hists=None,
                 epoch=-1):
        self.kwargs = kwargs
        self.name = name
        self.device = device
        self._epoch = epoch
        self.log = logger.bind(name=name, epoch=epoch)

        self.save_dir = save_dir
        self.tensorboard_dir = self.save_dir / 'tb' / name

        self._writer = None
        self._tic = time.time()
        self._modules = modules

        self._train_modules = train_modules
        self._optimizers = {}
        self._lr_schedulers = {}
        self._checkpoint_metric_tags = set()
        self.log.info("initialized trainer",
                      modules=list(modules.keys()),
                      save_dir=self.save_dir,
                      tensorboard_dir=self.tensorboard_dir)

        if meter_hists is None:
            self.meter_hists = {
                'train': defaultdict(list),
            }
        else:
            self.meter_hists = meter_hists

        self.plotter = None

        self.plot_interval = kwargs['plot_interval']
        self.save_interval = kwargs['save_interval']
        self.intervals = {
            'show': kwargs['show_interval'],
            'histogram': kwargs['histogram_interval'],
            'grad_histogram': kwargs['grad_histogram_interval'],
        }
        self.num_epochs = kwargs['num_epochs']
        self.batch_groups = kwargs.get('batch_groups', 1)
        self.batch_size = kwargs['batch_size']
        self.epoch_size = kwargs['batches_per_epoch']

        self.data_parallel = kwargs['data_parallel']
        self.num_workers = kwargs['num_workers']

        # Init dataset.
        self.dataset = dataset
        worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
        self._loader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size // self.batch_groups,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn)

        if kwargs['use_amp']:
            self._scaler = GradScaler()
        else:
            self._scaler = None

    def __getattr__(self, item):
        if item in self.kwargs:
            return self.kwargs[item]

        raise AttributeError(f"{self.__class__.__qualname__!r} has no attribute {item!r}")

    @property
    def writer(self):
        if self._writer is None:
            self._writer = SummaryWriter(str(self.tensorboard_dir))
        return self._writer

    def mark_time(self):
        elapsed = time.time() - self._tic
        self._tic = time.time()
        return elapsed

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self.log = self.log.bind(epoch=epoch)
        self._epoch = epoch

    def create_checkpoint(self):
        module_checkpoints = {
            name: module.create_checkpoint() if module else None
            for name, module in self._modules.items()
        }
        return {
            'args': self.kwargs,
            'epoch': self.epoch,
            'name': self.name,
            'meter_hists': self.meter_hists,
            'modules': module_checkpoints,
        }

    def _save_module_text(self):
        for name, module in self._modules.items():
            self.writer.add_text(f'model-{name}', tbutils.format_code(module))
            with open(self.save_dir / f'model-{name}.txt', 'w') as f:
                f.write(str(module))

    def set_train(self, value):
        for name, module in self._modules.items():
            if name in self._train_modules and module is not None:
                module.train(value).to(self.device)

    def start(self, train):
        self.log.info('** train start **')
        self.log.info('creating save directory', path=self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._save_module_text()
        self.writer.add_text('params', tbutils.format_dict_table(self.kwargs))
        with open(self.save_dir / 'params.json', 'w') as f:
            json.dump(self.kwargs, f, indent=2, cls=MyEncoder)

        # Save initial checkpoint.
        torchutils.save_checkpoint(self.save_dir, f'epoch-latest', self.create_checkpoint())

        # Increment epoch before starting. If load from checkpoint we're training the next epoch.
        # If fresh, the default epoch is -1.
        self.epoch += 1

        while self.epoch < self.num_epochs:
            for scheduler_name, scheduler in self._lr_schedulers.items():
                scheduler.step(epoch=self.epoch)
                logger.info('stepping LR scheduler',
                            scheduler_name=scheduler_name,
                            epoch=self.epoch,
                            current_lr=scheduler.get_lr())

            train_meters = self.run_epoch(train)

            for meter_name, meter in train_meters.items():
                mean, std = meter.value()
                self.meter_hists['train'][meter_name].append(mean)
                self.writer.add_scalar(f'summary-{meter_name}/train', meter.mean,
                                       global_step=self.epoch)

            checkpoint = self.create_checkpoint()

            for tag in self._checkpoint_metric_tags:
                torchutils.save_if_better(self.save_dir, checkpoint, train_meters, tag)

            torchutils.save_checkpoint(self.save_dir, f'epoch-latest', checkpoint)
            if self.epoch % self.save_interval == 0:
                torchutils.save_checkpoint(self.save_dir, f'epoch-{self.epoch:03d}', checkpoint)

            self.epoch += 1

    def run_epoch(self, train):
        self.log.info('running epoch',
                      epoch=self.epoch,
                      batch_size=self.batch_size,
                      batch_groups=self.batch_groups)
        self.plotter = TensorboardPlotter(self.writer, self.plot_interval, intervals=self.intervals)

        self.set_train(train)
        for module in self._modules.values():
            if module:
                module.to(self.device)

        loader_iter = iter(self._loader)
        num_iters = self.epoch_size * self.batch_groups
        loader_pbar = utils.pbar(range(num_iters))
        for iter_idx in loader_pbar:
            batch_idx = iter_idx // self.batch_groups
            misc_time = self.mark_time()
            batch = next(loader_iter)
            data_load_time = self.mark_time()

            global_step = self.epoch * num_iters * self.batch_size + iter_idx * self.batch_size
            self.plotter.set_step(global_step)

            is_step = (iter_idx + 1) % self.batch_groups == 0
            if train and is_step:
                for optimizer in self._optimizers.values():
                    optimizer.zero_grad()

            self.run_iteration(batch, train, is_step)

            batch_group_idx = iter_idx % self.batch_groups
            loader_pbar.set_description(
                f"global_step={global_step}"
                f", batch={batch_idx}"
                f", batch_group={batch_group_idx}"
            )

            self.plotter.put_scalar('time/data_load', data_load_time)
            self.plotter.put_scalar('time/misc', misc_time)

            for scheduler_name, scheduler in self._lr_schedulers.items():
                self.plotter.put_scalar(f'params/lr-{scheduler_name}', scheduler.get_lr()[0])

        return self.plotter.epoch_meters

    @abc.abstractmethod
    def run_iteration(self, batch, train, is_step):
        raise NotImplemented
