import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Optional

import structlog
import torch
from torch import nn
from torch.backends import cudnn
from torch.distributions import normal
from torch.nn import functional as F
from torch.utils.data import Dataset

from latentfusion import pggan, ibr
from latentfusion import recon
from latentfusion import trainutils
from latentfusion import utils
from latentfusion import visualization as viz
from latentfusion.augment import gan_denormalize
from latentfusion.losses import (multiscale_lsgan_loss, reduce_loss, beta_prior_loss)
from latentfusion.modules import unet
from latentfusion.modules.geometry import Camera
from latentfusion.recon import fusion
from latentfusion.recon.utils import process_batch, mask_normalized_depth
from latentfusion.style import StyleEncoder
from latentfusion.three.batchview import b2bv, bv2b
from latentfusion.utils import list_arg, block_config_arg

logger = structlog.get_logger(__name__)
cudnn.benchmark = True


def get_args():
    is_resume = '--resume' in sys.argv
    parser = argparse.ArgumentParser()
    parser = trainutils.add_common_args(parser)
    parser = trainutils.add_dataset_args(parser)

    # Architecture config.
    parser.add_argument('--generator-config',
                        default='32,D,64,D,128,D,256,D,512:512,U,256,U,128,U,64,U,32',
                        type=block_config_arg())
    parser.add_argument('--discriminator-config', default='64,128,256', type=list_arg(int))
    parser.add_argument('--discriminator-scales', default=3, type=int)
    parser.add_argument('--style-size', default=0, type=int)
    parser.add_argument('--style-normalize', action='store_true')
    parser.add_argument('--ibr-type', choices=['regress', 'blend', 'blend_flow'], default='regress')
    parser.add_argument('--flow-size', default=5, type=float)

    parser.add_argument('--save-dir', type=Path, required=not is_resume)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--override', type=list_arg(str))
    parser.add_argument('--branch', action='store_true')
    parser.add_argument('--base-name', type=str, required=not is_resume)
    parser.add_argument('--recon-checkpoint', type=Path, required=not is_resume)

    # Training parameters.
    parser.add_argument('--num-input-views', default=16, type=int)
    parser.add_argument('--num-output-views', default=8, type=int)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--generator-lr', default=0.001, type=float)
    parser.add_argument('--discriminator-lr', default=0.001, type=float)
    parser.add_argument('--recon-lr', default=0.0001, type=float)
    parser.add_argument('--train-recon', action='store_true')

    parser.add_argument('--g-gan-loss-weight', default=1.0, type=float)
    parser.add_argument('--g-color-recon-loss-weight', default=50.0, type=float)
    parser.add_argument('--g-color-recon-loss-type',
                        choices=['l1', 'smooth_l1', 'hard_l1', 'hard_smooth_l1',
                                 'perceptual_vgg16'],
                        default='l1')
    parser.add_argument('--g-color-recon-loss-k', type=int, default=2000)
    parser.add_argument('--g-color-recon-loss-k-milestones', type=list_arg(int))
    parser.add_argument('--g-depth-recon-loss-weight', default=50.0, type=float)
    parser.add_argument('--g-depth-recon-loss-type',
                        choices=['l1', 'smooth_l1', 'hard_l1', 'hard_smooth_l1'],
                        default='l1')
    parser.add_argument('--g-depth-recon-loss-k', type=int, default=2000)
    parser.add_argument('--g-mask-recon-loss-weight', default=50.0, type=float)
    parser.add_argument('--g-mask-beta-loss-weight', default=1.0, type=float)
    parser.add_argument('--g-mask-beta-loss-param', default=0.01, type=float)
    parser.add_argument('--g-mask-recon-loss-type',
                        choices=['l1', 'smooth_l1', 'hard_l1', 'hard_smooth_l1',
                                 'binary_cross_entropy'],
                        default='binary_cross_entropy')
    parser.add_argument('--g-mask-recon-loss-k', type=int, default=2000)
    parser.add_argument('--reconstruct-input', action='store_true')
    parser.add_argument('--no-apply-mask', action='store_true')

    parser.add_argument('--input-noise-mean', default=0.0, type=float)
    parser.add_argument('--input-noise-std', default=0.2, type=float)
    parser.add_argument('--input-noise-epochs', default=1000, type=float)
    parser.add_argument('--depth-noise-mean', default=0.0, type=float)
    parser.add_argument('--depth-noise-std', default=0.25, type=float)
    parser.add_argument('--generator-input-mask', action='store_true')
    parser.add_argument('--generator-input-depth', action='store_true')
    parser.add_argument('--no-discriminator', action='store_true')
    parser.add_argument('--discriminator-input-color', action='store_true')
    parser.add_argument('--discriminator-input-depth', action='store_true')
    parser.add_argument('--discriminator-input-mask', action='store_true')

    args = parser.parse_args()

    if args.batch_size % args.batch_groups != 0:
        raise ValueError("batch_size must be divisible by batch_groups")

    return args


def generate_name(base_name, args):
    return (
        f'{base_name}'
        f',{args.ibr_type}'
        f'{",no_disc" if args.no_discriminator else ""}'
        f'{",no_mask" if args.no_apply_mask else ""}'
        f',cr={args.g_color_recon_loss_type}'
    )


def load_recon_checkpoint(path):
    checkpoint = torch.load(path)
    params = checkpoint['args']
    name = checkpoint['name']
    epoch = checkpoint['epoch']
    logger.info("loading recon checkpoint", path=path, epoch=epoch, name=name)
    sculptor, fuser, photographer, discriminator = recon.models.load_models(checkpoint)
    return params, sculptor, fuser, photographer


def main():
    multiprocessing.set_start_method('forkserver')

    args = get_args()

    discriminator = None

    if args.resume:
        logger.info("loading checkpoint", path=args.resume)
        checkpoint = torch.load(args.resume)
        kwargs = trainutils.load_checkpoint_args(checkpoint['args'], args=args)
        name = checkpoint['name']
        epoch = checkpoint['epoch']
        meter_hists = checkpoint.get('meter_hists', None)
        recon_params = kwargs['recon_params']

        sculptor, fuser, photographer, discriminator = recon.models.load_models(checkpoint, kwargs=recon_params)
        generator = unet.UNet2d.from_checkpoint(checkpoint['modules']['generator'])
        if not kwargs.get('no_discriminator', False):
            discriminator = pggan.MultiScaleDiscriminator.from_checkpoint(checkpoint['modules']['discriminator'])
        style_encoder = None

        logger.info("** resuming training **", name=name, epoch=epoch)
    else:
        recon_params, sculptor, fuser, photographer = load_recon_checkpoint(args.recon_checkpoint)
        kwargs = vars(args)
        name = generate_name(args.base_name, args)
        epoch = -1
        meter_hists = None
        # Output depth + (cam_dists + input depth + input color) * num views.
        in_channels = 1 + (2 + 1 + 3) * kwargs['num_input_views']
        if kwargs['ibr_type'] == 'regress':
            out_channels = (3,)
        elif kwargs['ibr_type'] == 'blend':
            out_channels = (kwargs['num_input_views'],)
        elif kwargs['ibr_type'] == 'blend_flow':
            # blend weights + flow x + flow y.
            out_channels = (kwargs['num_input_views'],
                            kwargs['num_input_views'],
                            kwargs['num_input_views'])
        else:
            raise ValueError('Unknown ibr_type')

        generator = unet.UNet2d(in_channels=in_channels,
                                out_channels=out_channels,
                                block_config=args.generator_config)
        if not kwargs.get('no_discriminator', False):
            discriminator = pggan.MultiScaleDiscriminator(
                in_channels=3,
                block_config=args.discriminator_config,
                num_scales=args.discriminator_scales)
        style_encoder = None

    dataset_device = torch.device(args.dataset_gpu_id)
    dataset = trainutils.get_dataset(kwargs['dataset_type'], dataset_device, kwargs)
    device = torch.device(args.gpu_id)

    trainer = IBRTrainer(kwargs,
                         name=name,
                         save_dir=kwargs['save_dir'],
                         dataset=dataset,
                         recon_params=recon_params,
                         sculptor=sculptor,
                         fuser=fuser,
                         photographer=photographer,
                         generator=generator,
                         discriminator=discriminator,
                         style_encoder=style_encoder,
                         meter_hists=meter_hists,
                         epoch=epoch,
                         device=device)

    trainer.start(train=True)


class IBRTrainer(trainutils.Trainer):

    def __init__(self,
                 kwargs,
                 name,
                 save_dir,
                 *,
                 dataset: Dataset,
                 recon_params,
                 sculptor: recon.Sculptor,
                 fuser: Optional[fusion.Fuser],
                 photographer: recon.Photographer,
                 generator: unet.UNet2d,
                 discriminator: pggan.MultiScaleDiscriminator,
                 style_encoder: Optional[StyleEncoder],
                 device=torch.device("cuda:0"),
                 meter_hists=None,
                 epoch=0):
        modules = {
            'sculptor': sculptor,
            'fuser': fuser,
            'photographer': photographer,
            'generator': generator,
            'discriminator': discriminator,
            'style_encoder': style_encoder,
        }
        train_modules = {'generator', 'discriminator', 'style_encoder'}
        super().__init__(kwargs,
                         name=name,
                         save_dir=save_dir,
                         modules=modules,
                         train_modules=train_modules,
                         dataset=dataset,
                         device=device,
                         meter_hists=meter_hists,
                         epoch=epoch)
        self._recon_params = recon_params
        self._sculptor = sculptor
        self._photographer = photographer
        self._fuser = fuser
        self._generator = generator
        self._discriminator: pggan.MultiScaleDiscriminator = discriminator
        # self._style_encoder = style_encoder.to(device)

        self.kwargs['cube_size'] = self._recon_params['cube_size']
        self.kwargs['camera_dist'] = self._recon_params['camera_dist']
        self.kwargs['recon_params'] = self._recon_params

        self._checkpoint_metric_tags.update({
            'loss/generator/recon/color',
        })

        self._optimizers['generator'] = trainutils.get_optimizer(
            self._generator.parameters(),
            name=self.optimizer,
            lr=self.generator_lr)
        if self._discriminator:
            self._optimizers['discriminator'] = trainutils.get_optimizer(
                self._discriminator.parameters(),
                name=self.optimizer,
                lr=self.discriminator_lr)

        if self.train_recon:
            self._optimizers['sculptor'] = trainutils.get_optimizer(
                self._sculptor.parameters(),
                name=self.optimizer,
                lr=self.recon_lr)
            self._optimizers['photographer'] = trainutils.get_optimizer(
                self._photographer.parameters(),
                name=self.optimizer,
                lr=self.recon_lr)
            if len(list(self._fuser.parameters())) > 0:
                self._optimizers['fuser'] = trainutils.get_optimizer(
                    self._fuser.parameters(),
                    name=self.optimizer,
                    lr=self.recon_lr)

        self._g_color_recon_criterion = trainutils.get_recon_criterion(
            self.g_color_recon_loss_type,
            self.g_color_recon_loss_k).to(self.device)
        self._g_color_recon_k_scheduler = utils.MultiStepMilestoneScheduler(
            self.input_size ** 2, self.g_color_recon_loss_k_milestones, 0.5)
        self._g_depth_recon_criterion = trainutils.get_recon_criterion(
            self.g_depth_recon_loss_type,
            self.g_depth_recon_loss_k).to(self.device)
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

    def _render_reprojections(self, batch):
        recon_camera = Camera.vcat((batch['in_gt']['camera'], batch['out_gt']['camera']),
                                   batch_size=self.batch_size)

        if self._recon_params['generator_input_depth']:
            depth_noise = self._depth_noise_dist.sample(batch['in']['depth'].size()).to(self.device)
            depth_real_in = (batch['in']['depth'] + depth_noise).clamp(-1, 1)
        else:
            depth_real_in = None

        # Create IBR images.
        with torch.set_grad_enabled(self.train_recon):
            z_obj, z_extra = self._sculptor.encode(self._fuser,
                                                   camera=batch['in']['camera'],
                                                   color=batch['in']['image'],
                                                   depth=depth_real_in,
                                                   mask=batch['in']['mask'],
                                                   data_parallel=self.data_parallel)

            fake, _, _ = self._photographer.decode(z_obj, recon_camera, data_parallel=self.data_parallel)

            # Reshape things to BVCHW.
            sections = (self.num_input_views, self.num_output_views)
            depth_fake_in, depth_fake_out = torch.split(fake['depth'], sections, dim=1)
            mask_fake_in, mask_fake_out = torch.split(fake['mask'], sections, dim=1)

            image_reproj, depth_reproj, cam_dists_r, cam_dists_t = ibr.reproject_views_batch(
                image_in=batch['in']['image'],
                depth_in=depth_fake_in,
                depth_out=depth_fake_out,
                camera_in=batch['in']['camera'],
                camera_out=batch['out_gt']['camera'])
            image_reproj = image_reproj * mask_fake_out.unsqueeze(2)
            depth_reproj = (depth_reproj + 1.0) * mask_fake_out.unsqueeze(2) - 1.0

        return (
            image_reproj,
            depth_reproj,
            mask_fake_out.contiguous(),
            depth_fake_out.contiguous(),
            cam_dists_r,
            cam_dists_t,
        )

    def run_iteration(self, batch, train, is_step):
        if 'hard_' in self.g_depth_recon_loss_type:
            self._g_color_recon_criterion.k = int(self._g_color_recon_k_scheduler.get(self.epoch))

        batch = process_batch(batch, self.cube_size, self.camera_dist,
                              self._sculptor.in_size, self.device, random_orientation=False)

        if not self.color_random_background or self.crop_random_background:
            batch['in']['image'] = batch['in']['image'] * batch['in']['mask']

        if not self.depth_random_background or self.crop_random_background:
            batch['in']['depth'] = mask_normalized_depth(batch['in']['depth'], batch['in']['mask'])

        data_process_time = self.mark_time()

        generator = self._generator
        if self.data_parallel:
            generator = nn.DataParallel(generator)

        image_reproj, depth_reproj, mask_ibr_out, depth_ibr_out, cam_dists_r, cam_dists_t = \
            self._render_reprojections(batch)
        ibr_time = self.mark_time()

        # Add dists as another channel to image
        x = torch.cat((
            image_reproj,
            depth_reproj,
            cam_dists_r[:, :, :, None, None, None].expand(-1, -1, -1, -1, *image_reproj.shape[-2:]),
            cam_dists_t[:, :, :, None, None, None].expand(-1, -1, -1, -1, *image_reproj.shape[-2:]),
        ), dim=3)
        # Factor input views into channels and batch/output_views into batch dim.
        x = x.view(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3], x.shape[4], x.shape[5])

        # Add output predicted depth to input.
        x = torch.cat((bv2b(depth_ibr_out), x), dim=1)
        blend_weights = None
        flow_dx = None
        flow_dy = None
        logits = generator(x, z_inject=None)
        if self.ibr_type == 'regress':
            image_ibr_out = torch.tanh(logits)
        elif self.ibr_type == 'blend':
            image_ibr_out, blend_weights = ibr.blend_logits(logits, bv2b(image_reproj))
        else:
            image_ibr_out, blend_weights, flow_dx, flow_dy = ibr.warp_blend_logits(
                logits, bv2b(image_reproj), self.flow_size)

        image_ibr_out = b2bv(image_ibr_out, self.num_output_views)

        if not self.no_apply_mask:
            image_ibr_out = image_ibr_out * mask_ibr_out
            depth_ibr_out = mask_normalized_depth(depth_ibr_out, mask_ibr_out)

        if self._discriminator:
            image_real_noise = (self.input_noise_weight
                                * self._input_noise_dist.sample(batch['out_gt']['image'].size()))
            image_fake_noise = (self.input_noise_weight
                                * self._input_noise_dist.sample(image_ibr_out.size()))

            discriminator = self._discriminator
            if self.data_parallel:
                discriminator = nn.DataParallel(discriminator)
            # Train discriminator.
            d_real = discriminator(
                bv2b(batch['out_gt']['image'] + image_real_noise.to(self.device)),
                mask=bv2b(batch['out_gt']['mask']))
            d_fake_d = discriminator(image_ibr_out.detach() + image_fake_noise.to(self.device),
                                     mask=mask_ibr_out)

            loss_d_real = multiscale_lsgan_loss(d_real, 1)
            loss_d_fake = multiscale_lsgan_loss(d_fake_d, 0)
            loss_d = loss_d_real + loss_d_fake

            d_fake_g = discriminator(image_ibr_out + image_fake_noise, mask=mask_ibr_out)
            loss_g_gan = self.g_gan_loss_weight * multiscale_lsgan_loss(d_fake_g, 1)

            if train:
                loss_d.backward()
                if is_step:
                    self._optimizers['discriminator'].step()

            self.plotter.put_scalar('loss/discriminator/real', loss_d_real)
            self.plotter.put_scalar('loss/discriminator/fake', loss_d_fake)
            self.plotter.put_scalar('loss/discriminator/total', loss_d)
        else:
            d_real, d_fake_d, d_fake_g = None, None, None
            loss_g_gan = torch.tensor(0.0, device=self.device)

        # Train generator. Must re-evaluate discriminator to propagate gradients down the
        # generator.
        loss_g_color_recon = self.g_color_recon_loss_weight * reduce_loss(
            self._g_color_recon_criterion(image_ibr_out, batch['out_gt']['image']), reduction='mean')
        loss_g_depth_recon = self.g_depth_recon_loss_weight * reduce_loss(
            self._g_depth_recon_criterion(depth_ibr_out, batch['out_gt']['depth']), reduction='mean')
        loss_g_mask_recon = self.g_depth_recon_loss_weight * reduce_loss(
            self._g_depth_recon_criterion(mask_ibr_out, batch['out_gt']['mask']), reduction='mean')
        loss_g_mask_beta = beta_prior_loss(mask_ibr_out,
                                           alpha=self.g_mask_beta_loss_param,
                                           beta=self.g_mask_beta_loss_param)
        loss_g = (loss_g_gan
                  + loss_g_color_recon
                  + loss_g_depth_recon
                  + loss_g_mask_recon
                  + loss_g_mask_beta)
        if train:
            loss_g.backward()
            if is_step:
                self._optimizers['generator'].step()
                if self.train_recon:
                    self._optimizers['sculptor'].step()
                    self._optimizers['photographer'].step()
                    if 'fuser' in self._optimizers:
                        self._optimizers['fuser'].step()

        compute_time = self.mark_time()

        with torch.no_grad():
            self.plotter.put_scalar('error/color/l1',
                                    F.l1_loss(image_ibr_out, batch['out_gt']['image']))
            self.plotter.put_scalar('error/depth/l1',
                                    F.l1_loss(depth_ibr_out, batch['out_gt']['depth']))
            self.plotter.put_scalar('error/mask/l1',
                                    F.l1_loss(mask_ibr_out, batch['out_gt']['mask']))
            self.plotter.put_scalar(
                'error/mask/cross_entropy',
                F.binary_cross_entropy_with_logits(mask_ibr_out, batch['out_gt']['mask']))

        self.plotter.put_scalar('loss/generator/gan', loss_g_gan)
        self.plotter.put_scalar('loss/generator/recon/color', loss_g_color_recon)
        self.plotter.put_scalar('loss/generator/recon/depth', loss_g_depth_recon)
        self.plotter.put_scalar('loss/generator/recon/mask', loss_g_mask_recon)
        self.plotter.put_scalar('loss/generator/total', loss_g)

        self.plotter.put_scalar('params/input_noise_weight', self.input_noise_weight)

        self.plotter.put_scalar('time/data_process', data_process_time)
        self.plotter.put_scalar('time/ibr', ibr_time)
        self.plotter.put_scalar('time/compute', compute_time)
        plot_scalar_time = self.mark_time()
        self.plotter.put_scalar('time/plot/scalars', plot_scalar_time)

        if self.plotter.is_it_time_yet('show'):
            self.plotter.put_image(
                'results',
                viz.make_grid([
                    gan_denormalize(batch['out_gt']['image']),
                    gan_denormalize(image_ibr_out),
                    viz.colorize_tensor(batch['out_gt']['depth'] / 2.0 + 0.5),
                    viz.colorize_tensor(depth_ibr_out / 2.0 + 0.5),
                    viz.colorize_tensor(batch['out_gt']['mask']),
                    viz.colorize_tensor(mask_ibr_out),
                ], row_size=1, output_size=128, d_real=d_real, d_fake=d_fake_d))

            n = self.num_output_views
            images = [
                gan_denormalize(image_reproj[0].view(-1, *image_reproj.shape[-3:])),
                viz.colorize_tensor(
                    depth_reproj[0].view(-1, *depth_reproj.shape[-3:]) / 2.0 + 0.5)
            ]
            if blend_weights is not None:
                images.append(
                    viz.colorize_tensor(
                        blend_weights[:n].view(-1, *blend_weights.shape[-2:])))
            if flow_dx is not None:
                flow_range = self.flow_size / self.input_size
                images.append(
                    viz.colorize_tensor(
                        flow_dx[:n].reshape(-1, *flow_dx.shape[-2:]),
                        cmap='coolwarm', cmin=-flow_range, cmax=flow_range))
                images.append(
                    viz.colorize_tensor(
                        flow_dy[:n].reshape(-1, *flow_dy.shape[-2:]),
                        cmap='coolwarm', cmin=-flow_range, cmax=flow_range))
            self.plotter.put_image(
                'ibr_components', viz.make_grid(images, row_size=4, stride=4, output_size=64))

        if self.plotter.is_it_time_yet('histogram'):
            if flow_dx is not None:
                self.plotter.put_histogram('flow/dx', flow_dx)
                self.plotter.put_histogram('flow/dy', flow_dy)

        plot_images_time = self.mark_time()
        self.plotter.put_scalar('time/plot/images', plot_images_time)


if __name__ == '__main__':
    main()
