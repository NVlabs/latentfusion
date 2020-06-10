import argparse
import copy
import itertools
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.distributions import normal
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from latentfusion import consts, trainutils
from latentfusion import pggan
from latentfusion import recon
from latentfusion import utils
from latentfusion import visualization as viz
from latentfusion.augment import gan_denormalize
from latentfusion.losses import (multiscale_lsgan_loss, reduce_loss, beta_prior_loss)
from latentfusion.modules.geometry import Camera
from latentfusion.recon import fusion
from latentfusion.recon.utils import optimal_camera_dist, mask_normalized_depth, process_batch
from latentfusion.three.batchview import bv2b
from latentfusion.utils import list_arg, block_config_arg

logger = structlog.get_logger(__name__)


def get_args():
    is_resume = '--resume' in sys.argv
    is_branch = '--branch' in sys.argv
    parser = argparse.ArgumentParser()
    parser = trainutils.add_common_args(parser)
    parser = trainutils.add_dataset_args(parser)

    parser.add_argument('--save-dir', type=Path, required=not is_resume)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--override', type=list_arg(str))
    parser.add_argument('--branch', action='store_true')
    parser.add_argument('--base-name', type=str, required=not is_resume)
    parser.add_argument('--branch-name', type=str, required=is_branch)

    # Architecture parameters.
    parser.add_argument('--camera-dist', default=None, type=float)
    parser.add_argument('--cube-size', default=1.0, type=float)
    parser.add_argument('--cube-activation-type',
                        choices=['tanh', 'lrelu', 'relu', 'none'],
                        default='none')
    parser.add_argument('--fuser-type',
                        choices=['pool:max', 'pool:abs_max', 'pool:mean', 'pool:median',
                                 'concat', 'blend', 'hierarchical', 'gru', 'unet_gru', 'lstm'],
                        default='max')
    parser.add_argument('--sculptor-image-config',
                        default='64,D,64,D,128,D,256,D,512,D,512,D,512:512,U,512,U,512,U,256',
                        type=block_config_arg())
    parser.add_argument('--sculptor-camera-config', default='32,64,128',
                        type=block_config_arg())
    parser.add_argument('--sculptor-object-config', default='128,256',
                        type=block_config_arg())
    parser.add_argument('--photographer-object-config', default='256,256',
                        type=block_config_arg())
    parser.add_argument('--photographer-occlusion-config',
                        type=block_config_arg())
    parser.add_argument('--photographer-camera-config', default='256,256,256',
                        type=block_config_arg())
    parser.add_argument('--photographer-image-config',
                        default='256,D,512,D,512,D,512:512,U,512,U,512,U,256,U,128,U,64,U,32',
                        type=block_config_arg())
    parser.add_argument('--fuser-config', default='4,D,4,D,8,D,16:16,U,8,U,4,U,4',
                        type=block_config_arg())
    parser.add_argument('--photographer-projection-type', choices=['sum', 'factor'],
                        default='factor')
    parser.add_argument('--sculptor-projection-type', choices=['tile', 'factor'], default='factor')
    parser.add_argument('--discriminator-config', default='64,128,256,512', type=list_arg(int))
    parser.add_argument('--discriminator-scales', default=3, type=int)
    parser.add_argument('--no-discriminator', action='store_true')
    parser.add_argument('--random-orientation', action='store_true')
    parser.add_argument('--scale-mode',
                        choices=['nearest', 'bilinear'],
                        default='bilinear')

    # Training parameters.
    parser.add_argument('--num-input-views', default=16, type=int)
    parser.add_argument('--num-output-views', default=8, type=int)
    parser.add_argument('--generator-lr', default=0.001, type=float)
    parser.add_argument('--generator-lr-milestones', type=list_arg(int), default='100')
    parser.add_argument('--generator-lr-gamma', default=0.5, type=float)
    parser.add_argument('--discriminator-lr', default=0.001, type=float)

    parser.add_argument('--g-gan-loss-weight', default=1.0, type=float)
    parser.add_argument('--g-color-recon-loss-weight', default=50.0, type=float)
    parser.add_argument('--g-color-recon-loss-type',
                        choices=['l1', 'smooth_l1', 'hard_l1', 'hard_smooth_l1',
                                 'perceptual_vgg16'],
                        default='l1')
    parser.add_argument('--g-color-recon-loss-k', type=int, default=2000)
    parser.add_argument('--g-depth-recon-loss-weight', default=50.0, type=float)
    parser.add_argument('--g-depth-recon-loss-type',
                        choices=['l1', 'smooth_l1', 'hard_l1', 'hard_smooth_l1'],
                        default='l1')
    parser.add_argument('--g-depth-recon-loss-k', type=int, default=2000)
    parser.add_argument('--g-depth-recon-loss-k-milestones', type=list_arg(int))
    parser.add_argument('--g-mask-recon-loss-weight', default=50.0, type=float)
    parser.add_argument('--g-mask-beta-loss-weight', default=1.0, type=float)
    parser.add_argument('--g-mask-beta-loss-param', default=0.01, type=float)
    parser.add_argument('--g-mask-recon-loss-type',
                        choices=['l1', 'smooth_l1', 'hard_l1', 'hard_smooth_l1',
                                 'binary_cross_entropy'],
                        default='binary_cross_entropy')
    parser.add_argument('--g-mask-recon-loss-k', type=int, default=2000)
    parser.add_argument('--reconstruct-input', action='store_true')

    parser.add_argument('--input-noise-mean', default=0.0, type=float)
    parser.add_argument('--input-noise-std', default=0.2, type=float)
    parser.add_argument('--input-noise-epochs', default=1000, type=float)
    parser.add_argument('--depth-noise-mean', default=0.0, type=float)
    parser.add_argument('--depth-noise-std', default=0.25, type=float)
    parser.add_argument('--no-generator-input-color', action='store_true')
    parser.add_argument('--generator-input-mask', action='store_true')
    parser.add_argument('--generator-input-depth', action='store_true')
    parser.add_argument('--discriminator-input-color', action='store_true')
    parser.add_argument('--discriminator-input-depth', action='store_true')
    parser.add_argument('--discriminator-input-mask', action='store_true')
    parser.add_argument('--predict-color', action='store_true')
    parser.add_argument('--predict-mask', action='store_true')
    parser.add_argument('--predict-depth', action='store_true')
    parser.add_argument('--use-occlusion-depth', action='store_true')
    parser.add_argument('--crop-predicted-mask', action='store_true')

    args = parser.parse_args()

    if args.batch_size % args.batch_groups != 0:
        raise ValueError("batch_size must be divisible by batch_groups")

    return args


def generate_name(base_name, args):
    return (
        f'{base_name}'
        f'{",mask" if args.predict_mask else ""}'
        f'{",color" if args.predict_color else ""}'
        f'{",depth" if args.predict_depth else ""}'
        f'{",disc" if not args.no_discriminator else ""}'
        f'{",occlusion" if args.photographer_occlusion_config else ""}'
        f'{",no_in_color" if args.no_generator_input_color else ""}'
        f'{",in_mask" if args.generator_input_mask else ""}'
        f'{",in_depth" if args.generator_input_depth else ""}'
        f'{",const_cam" if args.use_constrained_cameras else ""}'
        f',mask_noise_p={args.mask_noise_p}'
        f',sm={args.scale_mode}'
        f',fuser={args.fuser_type}'
        # f',isize={args.input_size}'
        # f',csize={args.cube_size}'
        # f',cpool={args.fuser_type}'
        # f',o={args.optimizer}'
        # f',glr={args.generator_lr}'
        # f',dlr={args.discriminator_lr}'
        # f',gan_w={args.g_gan_loss_weight}'
        # f',cr={args.g_color_recon_loss_type}'
        # f',cr_w={args.g_color_recon_loss_weight}'
        # f',mr={args.g_mask_recon_loss_type}'
        # f',mr_w={args.g_mask_recon_loss_weight}'
    ).replace(':', '_')


def _fix_kwarg_paths(kwargs):
    update_kwargs = {}
    for key, value in kwargs.items():
        if key.endswith('_path') and value is not None:
            fast = Path('/local1')
            local1 = Path('/fast')
            local_drive = Path(*value.parts[:2])
            print(key, value, local_drive)
            if not local_drive.exists():
                if fast.exists():
                    update_kwargs[key] = Path(fast, *value.parts[2:])
                elif local1.exists():
                    update_kwargs[key] = Path(local1, *value.parts[2:])

    logger.info('fixed kwargs paths', **update_kwargs)
    kwargs = copy.deepcopy(kwargs)
    kwargs.update(update_kwargs)
    return kwargs


def main():
    multiprocessing.set_start_method('forkserver')

    args = get_args()

    discriminator: Optional[pggan.MultiScaleDiscriminator] = None

    if args.resume:
        logger.info("loading checkpoint", path=args.resume)
        checkpoint = torch.load(args.resume)
        kwargs = trainutils.load_checkpoint_args(checkpoint['args'], args=args)
        kwargs = _fix_kwarg_paths(kwargs)
        name = checkpoint['name']
        epoch = checkpoint['epoch']
        meter_hists = checkpoint['meter_hists']
        sculptor, fuser, photographer, discriminator = recon.models.load_models(checkpoint)

        logger.info("** resuming training **", name=name, epoch=epoch)
        if args.branch:
            now = datetime.now()
            old_name = name
            suffix = f"{args.branch_name}_{now.strftime('%Y%m%d_%Hh%Mm%Ss')}"
            name = f"{name}-{args.branch_name}"
            save_dir = Path(kwargs['save_dir'])
            kwargs['save_dir'] = save_dir.parent / f'{save_dir.name}-{suffix}'
            logger.info("** branching checkpoint **", name=name, branch_point=old_name, epoch=epoch)
    else:
        kwargs = vars(args)
        name = generate_name(args.base_name, args)
        epoch = -1
        meter_hists = None

        discriminator_in_channels = 0
        if args.discriminator_input_color and args.predict_color:
            discriminator_in_channels += 3
        if args.discriminator_input_depth and args.predict_depth:
            discriminator_in_channels += 1
        if args.discriminator_input_mask and args.predict_mask:
            discriminator_in_channels += 1
        if discriminator_in_channels == 0:
            raise ValueError('No inputs to discriminator.')

        if not kwargs['camera_dist']:
            camera_dist = optimal_camera_dist(
                focal_length=max(consts.INTRINSIC[0][0], consts.INTRINSIC[1][1]),
                size=args.input_size,
                radius=args.cube_size / 2.0,
                slack=1.0*128/args.input_size)
            logger.info("camera distance auto-computed", camera_dist=camera_dist)
            kwargs['camera_dist'] = camera_dist

        sculptor = recon.Sculptor(input_color=not args.no_generator_input_color,
                                  input_depth=args.generator_input_depth,
                                  input_mask=args.generator_input_mask,
                                  in_size=args.input_size,
                                  image_config=args.sculptor_image_config,
                                  camera_config=args.sculptor_camera_config,
                                  object_config=args.sculptor_object_config,
                                  cube_size=args.cube_size,
                                  cube_activation_type=args.cube_activation_type,
                                  projection_type=args.sculptor_projection_type,
                                  scale_mode=args.scale_mode)

        if kwargs['fuser_type'] == 'concat':
            in_views = kwargs['num_input_views']
        else:
            in_views = 1
        photographer = recon.Photographer(predict_color=args.predict_color,
                                          predict_depth=args.predict_depth,
                                          predict_mask=args.predict_mask,
                                          in_size=sculptor.out_size,
                                          in_views=in_views,
                                          image_config=args.photographer_image_config,
                                          camera_config=args.photographer_camera_config,
                                          object_config=args.photographer_object_config,
                                          occlusion_config=args.photographer_occlusion_config,
                                          projection_type=args.photographer_projection_type,
                                          cube_size=args.cube_size,
                                          scale_mode=args.scale_mode)
        fuser = fusion.get_fuser(args.fuser_type,
                                 in_channels=sculptor.camera_config[-1],
                                 cube_size=kwargs['cube_size'],
                                 block_config=args.fuser_config)

        if not args.no_discriminator:
            discriminator = pggan.MultiScaleDiscriminator(discriminator_in_channels,
                                                          block_config=args.discriminator_config,
                                                          num_scales=args.discriminator_scales)

    dataset_device = torch.device(args.dataset_gpu_id)
    dataset = trainutils.get_dataset(kwargs['dataset_type'], dataset_device, kwargs)

    logger.info("world configuration",
                cube_size=kwargs['cube_size'],
                camera_dist=kwargs['camera_dist'])
    logger.info("sculptor",
                image_config=sculptor.image_config,
                camera_config=sculptor.camera_config,
                object_config=sculptor.object_config,
                out_size=sculptor.out_size,
                out_channels=sculptor.out_channels,
                image_bottleneck_size=sculptor.image_bottleneck_size,
                image_out_size=sculptor.image_out_size,
                )
    logger.info("photographer",
                image_config=photographer.image_config,
                camera_config=photographer.camera_config,
                occlusion_config=photographer.occlusion_config,
                object_config=photographer.object_config,
                in_size=photographer.in_size,
                out_channels=photographer.out_channels,
                out_size=photographer.out_size,
                image_bottleneck_size=photographer.image_bottleneck_size,
                )
    logger.info("fuser", type=fuser.__class__.__qualname__)

    if photographer.occlusion_module:
        logger.info("occlusion module",
                    block_config=photographer.occlusion_module.block_config,
                    bottleneck_size=photographer.occlusion_module.bottleneck_size(
                        photographer.camera_out_size),
                    out_size=photographer.occlusion_module.output_size(
                        photographer.camera_out_size))

    device = torch.device(args.gpu_id)

    trainer = ReconTrainer(kwargs,
                           name=name,
                           save_dir=kwargs['save_dir'],
                           dataset=dataset,
                           sculptor=sculptor,
                           fuser=fuser,
                           photographer=photographer,
                           discriminator=discriminator,
                           meter_hists=meter_hists,
                           epoch=epoch,
                           device=device)

    trainer.start(train=True)


class ReconTrainer(trainutils.Trainer):

    def __init__(self,
                 kwargs,
                 name,
                 save_dir,
                 *,
                 dataset: Dataset,
                 sculptor: recon.Sculptor,
                 fuser: Optional[fusion.Fuser],
                 photographer: recon.Photographer,
                 discriminator: pggan.MultiScaleDiscriminator,
                 device=torch.device("cuda:0"),
                 meter_hists=None,
                 epoch=0):
        modules = {
            'sculptor': sculptor,
            'fuser': fuser,
            'photographer': photographer,
            'discriminator': discriminator,
        }
        train_modules = {'sculptor', 'fuser', 'photographer', 'discriminator'}
        super().__init__(kwargs,
                         name=name,
                         save_dir=save_dir,
                         modules=modules,
                         train_modules=train_modules,
                         dataset=dataset,
                         device=device,
                         meter_hists=meter_hists,
                         epoch=epoch)
        self._sculptor = sculptor
        self._fuser = fuser
        self._photographer = photographer
        self._discriminator = discriminator

        self._checkpoint_metric_tags.update({
            'error/depth/l1',
        })

        self._optimizers['generator'] = trainutils.get_optimizer(
            itertools.chain(
                self._sculptor.parameters(),
                self._photographer.parameters(),
                self._fuser.parameters(),
            ),
            name=self.optimizer,
            lr=self.generator_lr)

        if kwargs.get('generator_lr_milestones', None):
            self._lr_schedulers['generator'] = lr_scheduler.MultiStepLR(
                self._optimizers['generator'],
                milestones=kwargs['generator_lr_milestones'],
                gamma=kwargs['generator_lr_gamma'])

        if self._discriminator:
            self._optimizers['discriminator'] = trainutils.get_optimizer(
                self._discriminator.parameters(),
                name=self.optimizer,
                lr=self.discriminator_lr)

        self._g_color_recon_criterion = trainutils.get_recon_criterion(
            self.g_color_recon_loss_type,
            self.g_color_recon_loss_k).to(self.device)
        self._g_depth_recon_criterion = trainutils.get_recon_criterion(
            self.g_depth_recon_loss_type,
            self.g_depth_recon_loss_k).to(self.device)
        self._g_depth_recon_k_scheduler = utils.MultiStepMilestoneScheduler(
            self.input_size ** 2, self.g_depth_recon_loss_k_milestones, 0.5)
        self._g_mask_recon_criterion = trainutils.get_recon_criterion(
            self.g_mask_recon_loss_type,
            self.g_mask_recon_loss_k).to(self.device)

        self._input_noise_dist = normal.Normal(self.input_noise_mean, self.input_noise_std)
        self._depth_noise_dist = normal.Normal(self.depth_noise_mean, self.depth_noise_std)

    @property
    def num_recon_views(self):
        num_recon_views = self.num_output_views
        if self.reconstruct_input:
            num_recon_views += self.num_input_views
        return num_recon_views

    @property
    def input_noise_weight(self):
        return 1.0 - self.epoch / self.input_noise_epochs

    def run_iteration(self, batch, train, is_step):
        self.mark_time()
        # Update depth criterion k if applicable.
        if 'hard_' in self.g_depth_recon_loss_type:
            self._g_depth_recon_criterion.k = int(self._g_depth_recon_k_scheduler.get(self.epoch))

        batch = process_batch(batch, self.cube_size, self.camera_dist,
                              self._sculptor.in_size, self.device, self.random_orientation)

        if self.reconstruct_input:
            recon_camera = Camera.vcat((batch['in_gt']['camera'], batch['out_gt']['camera']),
                                       batch_size=self.batch_size)
            recon_mask = torch.cat((batch['in_gt']['mask'], batch['out_gt']['mask']), dim=1)
            recon_image = torch.cat((batch['in_gt']['image'], batch['out_gt']['image']), dim=1)
            recon_depth = torch.cat((batch['in_gt']['depth'], batch['out_gt']['depth']), dim=1)
        else:
            recon_camera = batch['out_gt']['camera']
            recon_mask = batch['out_gt']['mask']
            recon_image = batch['out_gt']['image']
            recon_depth = batch['out_gt']['depth']

        if not self.color_random_background or self.crop_random_background:
            batch['in']['image'] = batch['in']['image'] * batch['in']['mask']

        if not self.depth_random_background or self.crop_random_background:
            batch['in']['depth'] = mask_normalized_depth(batch['in']['depth'], batch['in']['mask'])

        depth_in = None
        if self.generator_input_depth:
            depth_noise = self._depth_noise_dist.sample(batch['in']['depth'].size()).to(self.device)
            depth_in = (batch['in']['depth'] + depth_noise).clamp(-1, 1)

        data_process_time = self.mark_time()

        with autocast():
            # Evaluate generator.
            z_obj, z_extra = self._sculptor.encode(self._fuser,
                                                   camera=batch['in']['camera'],
                                                   color=batch['in']['image'],
                                                   depth=depth_in,
                                                   mask=batch['in']['mask'],
                                                   data_parallel=self.data_parallel)
            fake_image, fake_depth, fake_mask, fake_mask_logits, fake_vox_depth = \
                self._run_photographer(z_obj, recon_camera, recon_mask)

            if 'blend_weights' in z_extra:
                z_weights = z_extra['blend_weights']
            else:
                z_weights = None

            # Train discriminator.
            if self._discriminator:
                d_real, d_fake_d, d_fake_g = self._run_discriminator(
                    fake_image, fake_depth, fake_mask, recon_image, recon_depth, recon_mask)
                loss_d_real = multiscale_lsgan_loss(d_real, 1)
                loss_d_fake = multiscale_lsgan_loss(d_fake_d, 0)
                loss_d = loss_d_real + loss_d_fake
                loss_g_gan = multiscale_lsgan_loss(d_fake_g, 1)

                if train:
                    loss_d.backward()
                    if is_step:
                        self._optimizers['discriminator'].step()

                self.plotter.put_scalar('loss/discriminator/real', loss_d_real)
                self.plotter.put_scalar('loss/discriminator/fake', loss_d_fake)
                self.plotter.put_scalar('loss/discriminator/total', loss_d)
            else:
                loss_g_gan = torch.tensor(0.0, device=self.device)

            # Train generator.
            if self.predict_color:
                loss_g_color_recon = reduce_loss(self._g_color_recon_criterion(fake_image, recon_image))
            else:
                loss_g_color_recon = torch.tensor(0.0, device=self.device)

            if self.predict_depth or self.use_occlusion_depth:
                loss_g_depth_recon = reduce_loss(self._g_depth_recon_criterion(fake_depth, recon_depth))
            else:
                loss_g_depth_recon = torch.tensor(0.0, device=self.device)

            if self.predict_mask:
                if self.g_mask_recon_loss_type == 'binary_cross_entropy':
                    y_mask = fake_mask_logits
                else:
                    y_mask = fake_mask
                loss_g_mask_recon = reduce_loss(self._g_mask_recon_criterion(y_mask, recon_mask))
                loss_g_mask_beta = beta_prior_loss(fake_mask,
                                                   alpha=self.g_mask_beta_loss_param,
                                                   beta=self.g_mask_beta_loss_param)
            else:
                loss_g_mask_recon = torch.tensor(0.0, device=self.device)
                loss_g_mask_beta = torch.tensor(0.0, device=self.device)

            loss_g = (
                    self.g_gan_loss_weight * loss_g_gan
                    + self.g_color_recon_loss_weight * loss_g_color_recon
                    + self.g_depth_recon_loss_weight * loss_g_depth_recon
                    + self.g_mask_recon_loss_weight * loss_g_mask_recon
                    + self.g_mask_beta_loss_weight * loss_g_mask_beta
            ) / self.batch_groups

        if train:
            if self.kwargs.get('use_amp', False):
                self._scaler.scale(loss_g).backward()
            else:
                loss_g.backward()

            if is_step:
                if self.kwargs.get('use_amp', False):
                    self._scaler.step(self._optimizers['generator'])
                    self._scaler.update()
                else:
                    self._optimizers['generator'].step()

        with torch.no_grad():
            if self.predict_depth:
                self.plotter.put_scalar('error/depth/l1', F.l1_loss(fake_depth, recon_depth))
            if self.reconstruct_input:
                self.plotter.put_scalar('error/depth/input_l1',
                                        F.l1_loss(fake_depth[:, :self.num_input_views],
                                                  batch['in_gt']['depth']))
                self.plotter.put_scalar('error/depth/output_l1',
                                        F.l1_loss(fake_depth[:, self.num_input_views:],
                                                  batch['out_gt']['depth']))
            if self.predict_mask:
                self.plotter.put_scalar(
                    'error/mask/cross_entropy',
                    F.binary_cross_entropy_with_logits(fake_mask_logits, recon_mask))
                self.plotter.put_scalar('error/mask/l1', F.l1_loss(fake_mask, recon_mask))

        compute_time = self.mark_time()

        self.plotter.put_scalar('loss/generator/gan', loss_g_gan)
        self.plotter.put_scalar('loss/generator/recon/color', loss_g_color_recon)
        self.plotter.put_scalar('loss/generator/recon/depth', loss_g_depth_recon)
        self.plotter.put_scalar('loss/generator/recon/mask', loss_g_mask_recon)
        self.plotter.put_scalar('loss/generator/recon/mask_beta', loss_g_mask_beta)
        self.plotter.put_scalar('loss/generator/total', loss_g)

        self.plotter.put_scalar('params/input_noise_weight', self.input_noise_weight)
        if hasattr(self._g_depth_recon_criterion, 'k'):
            self.plotter.put_scalar('params/depth_loss_k', self._g_depth_recon_criterion.k)

        self.plotter.put_scalar('time/data_process', data_process_time)
        self.plotter.put_scalar('time/compute', compute_time)
        plot_scalar_time = self.mark_time()
        self.plotter.put_scalar('time/plot/scalars', plot_scalar_time)

        if self.plotter.is_it_time_yet('histogram'):
            if self.predict_color:
                self.plotter.put_histogram('image_fake', fake_image)
                self.plotter.put_histogram('image_real', recon_image)
            if self.predict_mask:
                self.plotter.put_histogram('mask_fake', fake_mask)
            self.plotter.put_histogram('z_obj', z_obj)
            if z_weights is not None:
                self.plotter.put_histogram('z_weights', z_weights)
        plot_histogram_time = self.mark_time()
        self.plotter.put_scalar('time/plot/histogram', plot_histogram_time)

        if self.plotter.is_it_time_yet('show'):
            self.plotter.put_image(
                'inputs',
                viz.make_grid([
                    gan_denormalize(batch['in']['image']),
                    viz.colorize_depth(batch['in']['depth']) if self.generator_input_depth else None,
                    viz.colorize_tensor(batch['in']['mask']) if self.generator_input_mask else None,
                ], row_size=4, stride=2, output_size=64))
            with torch.no_grad():
                self.plotter.put_image(
                    'reconstruction',
                    viz.make_grid([
                        gan_denormalize(recon_image),
                        gan_denormalize(fake_image) if (fake_image is not None) else None,
                        viz.colorize_depth(recon_depth),
                        viz.colorize_depth(fake_depth) if (fake_depth is not None) else None,
                        viz.colorize_tensor((recon_depth.cpu() - fake_depth.cpu()).abs()) if (fake_depth is not None) else None,
                        viz.colorize_tensor(recon_mask),
                        viz.colorize_tensor(fake_mask) if (fake_mask is not None) else None,
                        viz.colorize_tensor((recon_mask.cpu() - fake_mask.cpu()).abs()) if (fake_mask is not None) else None,
                    ], stride=8))
        plot_images_time = self.mark_time()
        self.plotter.put_scalar('time/plot/images', plot_images_time)

    def _run_discriminator(self, image_fake, depth_fake, mask_fake, image_real, depth_real,
                           mask_real):
        discriminator = self._discriminator.to(self.device)
        if self.data_parallel:
            discriminator = nn.DataParallel(discriminator)

        y_fake = []
        y_real = []
        if self.discriminator_input_color:
            y_fake.append(image_fake)
            y_real.append(image_real)
        if self.discriminator_input_depth:
            y_fake.append(depth_fake)
            y_real.append(depth_real)
        if self.discriminator_input_mask:
            y_fake.append(mask_fake)
            y_real.append(mask_real)

        y_fake = torch.cat(bv2b(y_fake), dim=1)
        y_real = torch.cat(bv2b(y_real), dim=1)
        mask_real = bv2b(mask_real)

        image_real_noise = (
                self.input_noise_weight
                * self._input_noise_dist.sample(y_real.size())).to(self.device)
        image_fake_noise = (
                self.input_noise_weight
                * self._input_noise_dist.sample(y_fake.size())).to(self.device)

        # Train discriminator.
        d_real = discriminator(y_real + image_real_noise, mask=mask_real)
        d_fake_d = discriminator(y_fake.detach() + image_fake_noise, mask=mask_real)

        d_fake_g = discriminator(y_fake + image_fake_noise, mask=mask_real)

        return d_real, d_fake_d, d_fake_g

    def _run_photographer(self, z_obj, camera_recon, mask_real_recon):
        y, z, y_vox_depth = self._photographer.decode(z_obj, camera_recon,
                                                      interpret_logits=True,
                                                      data_parallel=self.data_parallel)

        # Crop image with mask.
        if self.predict_mask and self.predict_color:
            if self.crop_predicted_mask:
                y['color'] = y['color'] * y['mask']
            else:
                y['color'] = y['color'] * mask_real_recon

        return y.get('color'), y.get('depth'), y.get('mask'), y.get('mask_logits'), y_vox_depth


if __name__ == '__main__':
    main()
