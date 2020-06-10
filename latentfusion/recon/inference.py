from pathlib import Path

import structlog
import torch

from latentfusion import ibr
from latentfusion import recon
from latentfusion.observation import Observation
from latentfusion.three import bv2b, b2bv

logger = structlog.get_logger(__name__)


class LatentFusionModel(object):

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        if isinstance(checkpoint, Path) or isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)
        kwargs = checkpoint['args']
        name = checkpoint['name']
        epoch = checkpoint['epoch'] + 1
        sculptor, fuser, photographer, discriminator, generator = recon.models.load_models(
            checkpoint, device=device,
            return_generator=True)
        model = cls(sculptor, fuser, photographer, kwargs['camera_dist'], device, generator=generator)
        logger.info("loaded model", name=name, epoch=epoch)

        return model

    def __init__(self, sculptor, fuser, photographer, camera_dist, device, generator=None):
        self.device = device

        self.sculptor = sculptor.to(device)
        self.fuser = fuser.to(device)
        self.photographer = photographer.to(device)
        if generator is not None:
            self.generator = generator.to(device)
        else:
            self.generator = None

        self.camera_dist = camera_dist
        self.input_size = sculptor.in_size
        self.eval()

    def eval(self):
        self.train(False)
        return self

    def train(self, train):
        self.sculptor.train(train)
        self.photographer.train(train)
        self.fuser.train(train)
        if self.generator:
            self.generator.train(train)
        return self

    def zoom_observation(self, observation):
        if not observation.meta['is_zoomed']:
            return observation.zoom(self.camera_dist, self.input_size)
        return observation

    def preprocess_observation(self, observation):
        if not observation.meta['is_zoomed']:
            observation = observation.zoom(self.camera_dist, self.input_size)
        if not observation.meta['is_prepared']:
            observation = observation.prepare()
        if not observation.meta['is_normalized']:
            observation = observation.normalize()

        return observation

    def build_latent_object(self, observation: Observation):
        observation = self.preprocess_observation(observation).to(self.device)

        # Create object representation.
        with torch.no_grad():
            z_obj, _ = self.sculptor.encode(self.fuser,
                                            camera=observation.camera,
                                            color=observation.color.unsqueeze(0),
                                            depth=observation.depth.unsqueeze(0),
                                            mask=observation.mask.unsqueeze(0))

        return z_obj

    def compute_latent_code(self, observation, camera):
        observation = self.preprocess_observation(observation)

        num_batch = len(camera)
        if len(observation) == 1:
            observation = observation.expand(num_batch)

        _, feats_tar = recon.models.autoencode(self.sculptor, self.fuser, self.photographer,
                                               camera=camera,
                                               color=observation.color.unsqueeze(1),
                                               depth=observation.depth.unsqueeze(1),
                                               mask=observation.mask.unsqueeze(1))

        return feats_tar

    def render_full(self, z_obj, camera, input_obs=None, p=0.5):
        camera_zoom = camera.zoom(None, self.camera_dist, self.input_size).to(self.device)
        if input_obs is None:
            pred_y, _ = self.render_latent_object(z_obj, camera_zoom, apply_mask=True, return_latent=False)
        else:
            pred_y, _ = self.render_ibr_basic(z_obj, input_obs, camera_zoom, apply_mask=True, return_latent=False,
                                              p=p)

        out = {}
        mask = pred_y['mask']
        depth = pred_y['depth']
        depth = camera_zoom.denormalize_depth(depth) * mask
        out['depth'], _ = camera_zoom.uncrop(depth)
        out['mask'], _ = camera_zoom.uncrop(mask)

        if 'color' in pred_y:
            color = pred_y['color'] / 2 + 0.5
            out['color'], _ = camera_zoom.uncrop(color)

        return out

    def render_latent_object(self, z_obj, camera, return_latent=True, apply_mask=True):
        y_opt, z_opt, _ = self.photographer.decode(z_obj, camera, return_latent=return_latent,
                                                   apply_mask=apply_mask)
        if return_latent:
            z_opt = z_opt.squeeze(0)  # We're only decoding one object.

        return y_opt, z_opt

    def render_ibr_basic(self, z_obj, input_obs, camera_out, return_latent=True, apply_mask=True, p=0.5):
        input_obs = self.preprocess_observation(input_obs)
        color_in = input_obs.color
        camera_in = input_obs.camera
        y_ibr, z_ibr = ibr.render_latent_ibr2(
            self.photographer, z_obj,
            camera_in.clone().to(self.device),
            camera_out.clone().to(self.device),
            b2bv(color_in, batch_size=1).to(self.device),
            p=p,
            weight_type='cam_dist',
            return_latent=return_latent,
            apply_mask=apply_mask)
        if return_latent:
            z_ibr = z_ibr.squeeze(0)  # We're only decoding one object.
        y_ibr = {
            k: v.squeeze(0) for k, v in y_ibr.items()
        }

        return y_ibr, z_ibr

    def render_ibr(self, z_obj, input_obs, camera_out, return_latent=True):
        input_obs = self.preprocess_observation(input_obs)
        color_in = input_obs.color
        camera_in = input_obs.camera

        y_out, z_out, image_reproj, depth_reproj, mask_ibr_out, depth_ibr_out, cam_dist_r, cam_dist_t = \
            self._render_reprojections(z_obj, color_in, camera_in, camera_out)
        if return_latent:
            z_out = z_out.squeeze(0)  # We're only decoding one object.

        # TODO: switch this out once new model is trained.
        cam_sims = 1.0 - cam_dist_t * 2

        # Add dists as another channel to image
        x = torch.cat((
            image_reproj,
            depth_reproj,
            cam_sims[:, :, None, None, None].expand(-1, -1, -1, *image_reproj.shape[-2:]),
        ), dim=2)

        # # Add dists as another channel to image
        # x = torch.cat((
        #     image_reproj,
        #     depth_reproj,
        #     cam_dist_r[:, :, None, None, None].expand(-1, -1, -1, *image_reproj.shape[-2:]),
        #     cam_dist_t[:, :, None, None, None].expand(-1, -1, -1, *image_reproj.shape[-2:]),
        # ), dim=2)

        # Factor views into channels.
        x = x.view(-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

        # Add output predicted depth to input.
        x = torch.cat((depth_ibr_out, x), dim=1)
        logits = self.generator(x)
        color_ibr, _, _, _ = ibr.warp_blend_logits(logits, image_reproj, 5)
        # color_ibr, _ = ibr.blend_logits(logits, image_reproj)
        y_out['color'] = color_ibr
        y_out = {
            k: v.squeeze(0) for k, v in y_out.items()
        }

        return y_out, z_out

    def _render_reprojections(self, z_obj, color_in, camera_in, camera_out, return_latent=True):
        # Create IBR images.
        y_in, _, _ = self.photographer.decode(z_obj, camera_in)
        y_out, z_out, _ = self.photographer.decode(z_obj, camera_out, return_latent=return_latent)
        mask_fake_out = y_out['mask']
        depth_fake_out = y_out['depth']

        image_reproj, depth_reproj, cam_dist_r, cam_dist_t = ibr.reproject_views_batch(
            color_in.unsqueeze(0), y_in['depth'], y_out['depth'], camera_in, camera_out)
        image_reproj = image_reproj * mask_fake_out.unsqueeze(2)
        depth_reproj = (depth_reproj + 1.0) * mask_fake_out.unsqueeze(2) - 1.0

        return (
            y_out,
            z_out,
            bv2b(image_reproj),
            bv2b(depth_reproj),
            bv2b(mask_fake_out),
            bv2b(depth_fake_out),
            bv2b(cam_dist_r),
            bv2b(cam_dist_t),
        )


