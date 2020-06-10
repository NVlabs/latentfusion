import abc
import copy
from collections import defaultdict
from pathlib import Path

import math
import numpy as np
import sklearn.mixture
import structlog
import toml
import torch
from torch import optim
from torch.nn import functional as F

from latentfusion import three, utils, distances
from latentfusion.modules.geometry import Camera
from latentfusion.observation import Observation
from latentfusion.pose import initialization
from latentfusion.pose import utils as pu
from latentfusion.recon.inference import LatentFusionModel
from latentfusion.utils import ExponentialScheduler, LinearScheduler

DEFAULT_TRANSLATION_STD = 0.01
DEFAULT_QUATERION_STD = 10.0 / 180.0 * math.pi

logger = structlog.get_logger(__name__)


def load_from_config(config, model, **kwargs):
    if isinstance(config, Path) or isinstance(config, str):
        config = toml.load(config)

    params = config['args']
    params.update(kwargs)

    logger.info('loading pose estimator from config',
                type=config['type'],
                **params,
                loss_weights=config['loss_weights'])

    if config['type'] == 'metropolis':
        return MetropolisPoseEstimator(model=model,
                                       **params,
                                       loss_weights=config['loss_weights'])
    elif config['type'] == 'cross_entropy':
        return CrossEntropyPoseEstimator(model=model,
                                         **params,
                                         loss_weights=config['loss_weights'])
    elif config['type'] == 'gradient':
        loss_schedules = {
            k: load_schedules_from_config(v)
            for k, v in config.get('loss_schedules', {}).items()
        }
        return GradientPoseEstimator(model=model,
                                     **params,
                                     loss_weights=config['loss_weights'],
                                     loss_schedules=loss_schedules)
    else:
        raise ValueError(f"Unknown estimator type {config['type']}")


def load_schedules_from_config(config):
    config = copy.copy(config)
    if config.pop('type') == 'exponential':
        return ExponentialScheduler(**config)
    if config.pop('type') == 'linear':
        return LinearScheduler(**config)


def default_pose_loss(target, z_pred_depth, z_pred_mask_logits, z_pred_camera,
                      z_pred_latent=None,
                      z_target_latent=None):
    """
    Computes a loss that measures the fitness of a pose.

    We pass the mask logits and scale here so that we can use the unscaled mask for determining
    the valid area of the depth. The depth prediction may not be valid for pixels outside of
    the network's predicted mask so we only use the potentially expanded mask for the IOU
    and shape loss.
    """

    pred_depth, pred_camera = z_pred_camera.uncrop(z_pred_depth, scale_mode='nearest')
    pred_mask_logits, _ = z_pred_camera.uncrop(z_pred_mask_logits, scale_mode='bilinear')
    pred_mask = torch.sigmoid(pred_mask_logits)
    pred_depth = pred_depth * pred_mask
    invalid_mask = (target.depth == 0) & (target.mask > 0.1)

    target = target.prepare()
    target_mask = target.mask
    target_depth = target.depth

    loss_dict = {}

    overlap_mask = pred_mask * target_mask
    depth_loss = F.l1_loss(pred_depth, target_depth.expand_as(pred_depth), reduction='none')
    depth_loss = pu.zero_invalid_pixels(depth_loss, invalid_mask)
    ov_depth_loss = pu.reduce_loss_mask(depth_loss, overlap_mask)
    loss_dict['ov_depth'] = ov_depth_loss
    loss_dict['depth'] = depth_loss.mean(dim=(1, 2, 3))

    # IOU Loss.
    iou_loss = pu.iou_loss(pred_mask, pu.zero_invalid_pixels(target.mask, invalid_mask))
    loss_dict['iou'] = iou_loss

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, target_mask.expand_as(pred_mask), reduction='none')
    # mask_loss = pu.zero_invalid_pixels(mask_loss, target_ring_mask)
    mask_loss = mask_loss.mean(dim=(1, 2, 3))
    loss_dict['mask'] = mask_loss

    if z_pred_latent is not None and z_target_latent is not None:
        z_pred_latent = z_pred_latent.view(z_pred_latent.shape[0], -1)
        z_target_latent = z_target_latent.view(z_target_latent.shape[0], -1)
        loss_dict['latent'] = distances.cosine_distance(z_pred_latent, z_target_latent.expand_as(z_pred_latent))
        # loss_dict['latent'] = F.l1_loss(z_pred_latent, z_target_latent.expand_as(z_pred_latent),
        #                                 reduction='none').mean(dim=(1, 2, 3))

    return loss_dict


def weigh_losses(loss_dict, weight_dict):
    weighted_losses = {}
    for k, v in loss_dict.items():
        weighted_losses[k] = weight_dict.get(k, 0.0) * v

    return weighted_losses


class PoseEstimator(abc.ABC):

    def __init__(self, *, model: LatentFusionModel, ranking_size, loss_weights, loss_func=None,
                 return_camera_history=False, verbose=False):
        self.model = model
        self.ranking_size = ranking_size
        if loss_func is None:
            loss_func = default_pose_loss
        self.loss_func = loss_func
        self.loss_weights = defaultdict(float)
        self.loss_weights.update(loss_weights)

        self.return_camera_history = return_camera_history
        self.verbose = verbose

    @property
    def device(self):
        return self.model.device

    @classmethod
    def initial_pose(cls, target_obs):
        """
        Computes the initial pose based on the depth and foreground mask. This can only infer
        the translation and the rotation is set to the identity.

        Args:
            target_obs (Observation): the target observation to estimate from.

        Returns:
            (Camera): a camera with estimated translation.
        """
        return initialization.estimate_initial_pose(target_obs.depth,
                                                    target_obs.mask,
                                                    target_obs.camera.intrinsic,
                                                    target_obs.camera.width,
                                                    target_obs.camera.height)

    def estimate(self, z_obj, target_obs, **kwargs):
        """
        Estimates the pose given a latent object representation and a target observation.

        Args:
            z_obj (torch.tensor): A latent object represention.
            target_obs (Observation): an observation to estimate the pose for.

        Returns:
            (Camera): One or multiple cameras representing the estimated pose.
        """
        if len(target_obs) > 1:
            raise ValueError(f"The pose can only be estiamted for one observation at a time.")

        return self._estimate(z_obj, target_obs, **kwargs)

    @abc.abstractmethod
    def _estimate(self, z_obj, target_obs, **kwargs):
        """Implementation of the pose estimation."""
        raise NotImplementedError()

    def _track_best_items(self, ranking, step, items, loss):
        loss = loss.cpu()
        # Keep track of best poses.
        prev_best_error = ranking[0][1] if len(ranking) > 0 else float('inf')
        ranking.extend((c, e.item(), step) for c, e in zip(items, loss))
        ranking.sort(key=lambda x: x[1])
        del ranking[self.ranking_size:]
        best_camera, best_error, _ = ranking[0]
        delta = 0.0
        if best_error < prev_best_error:
            delta = prev_best_error - best_error
            if self.verbose:
                logger.info('better camera found',
                            error=best_error,
                            last_error=prev_best_error,
                            delta=delta,
                            step=step)

        return delta

    def _render_observation(self, z_obj, camera, **kwargs):
        z_camera = camera.zoom(None, self.model.input_size, self.model.camera_dist)
        with torch.set_grad_enabled(kwargs.get('grad_enabled', False)):
            pred_dict, z_latent = self.model.render_latent_object(
                z_obj, z_camera.to(self.device), return_latent=True)
            z_mask = pred_dict['mask'].squeeze(0)
            z_mask_logits = pred_dict['mask_logits'].squeeze(0)
            z_depth = camera.denormalize_depth(pred_dict['depth'].squeeze(0)) * z_mask

        return z_depth, z_mask_logits, z_latent, z_camera


class MetropolisPoseEstimator(PoseEstimator):
    """
    Uses the Metropolis-Hastings algorithm with simulated annealing to optimize the pose.

    References:
        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        https://en.wikipedia.org/wiki/Simulated_annealing
    """

    def __init__(self, *, num_samples, num_iters,
                 translation_std=DEFAULT_TRANSLATION_STD,
                 quaternion_std=DEFAULT_QUATERION_STD,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_iters = num_iters
        self.translation_std = translation_std
        self.quaternion_std = quaternion_std

    def _estimate(self, z_obj, target_obs, **kwargs):
        camera_init = self.initial_pose(target_obs)
        camera = pu.sample_cameras_with_estimate(self.num_samples, camera_init).to(
            self.device)
        error = torch.full((self.num_samples,), fill_value=100.0, device=self.device)
        ranking = []

        temp_weight = 1.0 / camera_init.translation[:, -1].mean().item()
        temp_sched = ExponentialScheduler(temp_weight * 0.1,
                                          temp_weight * 0.005, num_steps=self.num_iters)
        logger.info("simulated annealing",
                    temp_weight=temp_weight,
                    temp_sched_range=[temp_sched.initial_value, temp_sched.final_value],
                    n_iters=self.num_iters,
                    n_samples=self.num_samples)

        target_obs = target_obs.to(self.device)

        camera_history = []
        pbar = utils.trange(self.num_iters)
        for step in pbar:
            temperature = temp_sched.get(step)
            camera, error, num_accepted = self._refine_pose(z_obj,
                                                            camera.clone(),
                                                            error.clone(),
                                                            target_obs=target_obs,
                                                            temperature=temperature)
            delta = self._track_best_items(ranking, step, camera, error)
            if delta > 0:
                camera_history.append((error, camera.clone().cpu()))

            pbar.set_description(
                f"E={ranking[0][1]:.05f}, "
                f"T={temperature:.04f}, "
                f"N={num_accepted}/{self.num_samples}")

        cameras = Camera.cat([c for c, e, step in ranking])
        if self.return_camera_history:
            return cameras, camera_history
        else:
            return cameras

    def _refine_pose(self, z_obj, prev_camera: Camera, prev_error, target_obs, temperature=1.0):
        camera = pu.perturb_camera(prev_camera, self.translation_std, self.quaternion_std)
        z_target_latent = self.model.compute_latent_code(target_obs, camera)
        z_pred_depth, z_pred_mask_logits, z_pred_latent, z_camera = self._render_observation(z_obj, camera)
        loss_dict = self.loss_func(target_obs, z_pred_depth, z_pred_mask_logits, z_camera,
                                   z_pred_latent=z_pred_latent,
                                   z_target_latent=z_target_latent)
        loss = sum(weigh_losses(loss_dict, self.loss_weights).values())
        transition_prob = torch.exp((prev_error - loss) / temperature)
        thres = torch.rand_like(transition_prob)
        accept = transition_prob > thres

        camera[~accept] = prev_camera[~accept]
        loss[~accept] = prev_error[~accept]

        return camera, loss, accept.sum().item()


class CrossEntropyPoseEstimator(PoseEstimator):
    """
    Uses the Cross Entropy method to optimize the pose.

    References:
        https://en.wikipedia.org/wiki/Cross-entropy_method
        http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf
    """

    def __init__(self, *, num_samples, num_elites, num_iters,
                 num_gmm_components, learning_rate,
                 sample_flipped=False,
                 init_hemisphere=False,
                 init_upright=False,
                 translation_std=DEFAULT_TRANSLATION_STD,
                 quaternion_std=DEFAULT_QUATERION_STD,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.num_gmm_components = num_gmm_components
        self.sample_flipped = sample_flipped
        self.init_upright = init_upright
        self.init_hemisphere = init_hemisphere
        self.learning_rate = learning_rate
        self.translation_std = translation_std
        self.quaternion_std = quaternion_std
        self.elite_sched = ExponentialScheduler(num_samples, num_elites, num_iters)

    def _estimate(self, z_obj, target_obs, **kwargs):
        if kwargs.get('cameras', None):
            cameras = kwargs['cameras']
            camera_init = kwargs['cameras'][0]
        else:
            camera_init = self.initial_pose(target_obs)
            cameras = pu.sample_cameras_with_estimate(
                n=self.num_gmm_components * self.num_samples,
                camera_est=camera_init,
                upright=self.init_upright,
                hemisphere=self.init_hemisphere)

        gmm = self._create_gmm(self._camera_to_params(cameras))
        target_obs = target_obs.to(self.device)
        camera_history = []

        prev_gmm = None
        ranking = []
        pbar = utils.trange(self.num_iters)
        for step in pbar:
            # Refine pose.
            _num_elites = int(self.elite_sched.get(step))
            cameras, losses = self._refine_pose(z_obj, target_obs, prev_gmm, gmm,
                                                num_elites=_num_elites,
                                                camera_init=camera_init)
            prev_gmm = gmm
            gmm = self._create_gmm(self._camera_to_params(cameras).cpu())
            delta = self._track_best_items(ranking, step, cameras, losses)
            if delta > 0:
                camera_history.append((losses, Camera.cat([c for c, e, step in ranking])))
            pbar.set_description(f"best_error={ranking[0][1]:.05f}, num_elite={_num_elites}")

        # gmm_camera = self._params_to_camera(torch.tensor(gmm.means_, dtype=torch.float32),
        #                                     camera_init=camera_init)

        logger.info('best camera', step=ranking[0][2], loss=ranking[0][1])

        cameras = Camera.cat([c for c, e, step in ranking])
        if self.return_camera_history:
            return cameras, camera_history
        else:
            return cameras

    def _refine_pose(self, z_obj, target_obs, prev_gmm, gmm, num_elites, camera_init):
        # Sample from blended distribution and then set current distribution to
        # new distribution.
        if prev_gmm is not None:
            sample_gmm = self._combined_gmm(prev_gmm, gmm, self.learning_rate)
        else:
            sample_gmm = gmm

        num_samples = self.num_samples // 4 if self.sample_flipped else self.num_samples
        params = self._sample_poses(sample_gmm, num_samples)
        cameras = self._params_to_camera(params, camera_init=camera_init, device=self.device)

        if self.sample_flipped:
            cameras = Camera.cat([
                cameras,
                pu.flip_camera(cameras, axis=(0.0, 0.0, 1.0)),
                pu.flip_camera(cameras, axis=(0.0, 1.0, 0.0)),
                pu.flip_camera(cameras, axis=(1.0, 0.0, 0.0)),
            ])

        if self.loss_weights.get('latent', 0.0) > 0.0:
            with torch.no_grad():
                z_target_latent = self.model.compute_latent_code(target_obs, cameras[0])
        else:
            z_target_latent = None

        z_pred_depth, z_pred_mask_logits, z_pred_latent, z_camera = self._render_observation(z_obj, cameras)
        loss_dict = self.loss_func(target_obs, z_pred_depth, z_pred_mask_logits, z_camera,
                                   z_pred_latent=z_pred_latent,
                                   z_target_latent=z_target_latent)
        loss = sum(weigh_losses(loss_dict, self.loss_weights).values())
        sorted_inds = torch.argsort(loss)
        elite_inds = sorted_inds[:num_elites]
        if self.verbose:
            logger.info('pose error', **{k: v[elite_inds[0]].item() for k, v in loss_dict.items()})

        elite_losses = loss[elite_inds]
        elite_cameras = cameras[elite_inds]

        return elite_cameras, elite_losses

    def _sample_poses(self, gmm, n):
        """
        Sample poses from a distribution.
        Args:
            gmm: the distribution to sample from.
            n (int): the number of samples.

        Returns:
            torch.Tensor: an (n, 6) tensor containing camera parameters.
        """
        params, _ = gmm.sample(n)
        params = torch.tensor(params, dtype=torch.float32, device=self.device)
        # TODO: Think about this more. It seems to help.
        params[:, :3] += torch.randn_like(params[:, :3]) * self.translation_std
        params[:, 3:] += torch.randn_like(params[:, 3:]) * self.quaternion_std
        return params

    def _create_gmm(self, params=None):
        """
        Creates a Gaussian mixture model instance.

        Args:
            params (Optional[torch.Tensor]): if not None the GMM will be fit using these parameters.

        Returns:
            sklearn.mixture.GaussianMixture: a Gaussian mixture model.
        """
        gmm = sklearn.mixture.GaussianMixture(covariance_type='diag',
                                              n_components=self.num_gmm_components,
                                              reg_covar=1e-5)
        if params is not None:
            if torch.is_tensor(params):
                params = params.numpy()
            gmm.fit(params)

        return gmm

    def _combined_gmm(self, old_gmm, new_gmm, alpha):
        """
        Blends two GMMs together by sampling from both of their components.

        Args:
            old_gmm: the old distribution to add.
            new_gmm:  the new distribution to add.
            alpha: the weight of the new distribution.

        Returns:
            the weighted combination of the two distributions.
        """
        if alpha > 1.0 or alpha < 0.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        out_gmm = self._create_gmm()
        out_gmm.weights_ = np.concatenate([(1.0 - alpha) * old_gmm.weights_,
                                           alpha * new_gmm.weights_], axis=0)
        out_gmm.means_ = np.concatenate([old_gmm.means_,
                                         new_gmm.means_], axis=0)
        out_gmm.covariances_ = np.concatenate([old_gmm.covariances_,
                                               new_gmm.covariances_], axis=0)
        out_gmm.precisions_cholesky_ = np.concatenate([old_gmm.precisions_cholesky_,
                                                       new_gmm.precisions_cholesky_], axis=0)
        return out_gmm

    @classmethod
    def _camera_to_params(cls, camera):
        return torch.cat([
            camera.translation,
            camera.log_quaternion,
        ], dim=-1)

    @classmethod
    def _params_to_camera(cls, params, camera_init, device='cpu'):
        if len(params.shape) == 1:
            params = params.unsqueeze(0)

        intrinsic = camera_init.intrinsic.expand(params.shape[0], -1, -1).to(device)
        translations = params[:, :3].to(device)
        log_quaternions = params[:, 3:].to(device)
        cameras = Camera(intrinsic=intrinsic,
                         extrinsic=None,
                         translation=translations,
                         log_quaternion=log_quaternions,
                         width=camera_init.width,
                         height=camera_init.height,
                         z_span=camera_init.z_span).to(device)
        return cameras


class GradientPoseEstimator(PoseEstimator):
    """
    Optimizes the pose using gradient updates.
    """

    def __init__(self, *,
                 learning_rate,
                 num_samples,
                 num_iters,
                 converge_threshold,
                 converge_patience,
                 lr_reduce_patience=25,
                 lr_reduce_threshold=1e-5,
                 lr_reduce_factor=0.5,
                 track_stats=False,
                 loss_schedules=None,
                 optimizer='adamw',
                 **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.num_iters = num_iters
        self.optimizer = optimizer
        self.lr_reduce_patience = lr_reduce_patience
        self.lr_reduce_threshold = lr_reduce_threshold
        self.lr_reduce_factor = lr_reduce_factor
        self.converge_threshold = converge_threshold
        self.converge_patience = converge_patience
        self.loss_schedules = {}
        if loss_schedules is not None:
            self.loss_schedules.update(loss_schedules)

        self.track_stats = track_stats

    def _estimate(self, z_obj, target_obs, **kwargs):
        if 'camera' in kwargs:
            camera = kwargs['camera']
        else:
            camera = self.initial_pose(target_obs)
            camera = pu.sample_cameras_with_estimate(
                n=self.num_samples, camera_est=camera)
        target_obs = target_obs.to(self.device)

        # Mask parameters.

        # Optimize the 'zoomed' camera.
        camera = camera.zoom(None, self.model.input_size, self.model.camera_dist).to(self.device)

        ranking = []
        stat_history, camera_history = self._optimize_camera(
            z_obj, target_obs, camera,
            iters=self.num_iters,
            ranking=ranking)

        logger.info('best camera', step=ranking[0][2], loss=ranking[0][1])
        best_cameras = Camera.cat([c for c, loss, step in ranking])

        if self.track_stats and self.return_camera_history:
            return best_cameras, stat_history, camera_history
        elif self.track_stats:
            return best_cameras, stat_history
        elif self.return_camera_history:
            return best_cameras, camera_history
        else:
            return best_cameras

    @classmethod
    def get_optimizer(cls, name, *args, **kwargs):
        if name == 'adamw':
            return optim.AdamW(*args, **kwargs)
        elif name == 'adam':
            return optim.Adam(*args, **kwargs)
        elif name == 'sgd':
            return optim.SGD(*args, **kwargs)
        elif name == 'adagrad':
            return optim.Adagrad(*args, **kwargs)
        else:
            raise ValueError(f"Unknow optimizer {name!r}")

    def _optimize_camera(self, z_obj, target_obs, cameras, iters, ranking):
        optimizers = []
        schedulers = []
        param_cameras = [pu.parameterize_camera(camera, optimize_viewport=True) for camera in
                         cameras]
        for camera in param_cameras:
            parameters = [camera.log_quaternion, camera.translation, camera.viewport]
            optimizer = self.get_optimizer(self.optimizer, parameters, lr=self.learning_rate)
            optimizers.append(optimizer)
            schedulers.append(
                optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.lr_reduce_patience,
                    threshold=self.lr_reduce_threshold,
                    factor=self.lr_reduce_factor,
                    verbose=self.verbose))

        pbar = utils.trange(iters)
        stat_history = {}
        converge_count = 0
        camera_history = []

        for step in pbar:
            for optimizer in optimizers:
                optimizer.zero_grad()
            cameras = Camera.cat(param_cameras)
            if self.loss_weights.get('latent', 0.0) > 0.0:
                with torch.no_grad():
                    z_target_latent = self.model.compute_latent_code(target_obs, cameras)
            else:
                z_target_latent = None
            z_depth, z_mask, z_mask_logits, z_pred_latent = self._render_observation(z_obj, cameras)
            optim_weights = copy.copy(self.loss_weights)
            optim_weights.update({k: v.get(step) for k, v in self.loss_schedules.items()})

            loss_dict = self.loss_func(target_obs, z_depth, z_mask_logits, cameras,
                                       z_pred_latent=z_pred_latent, z_target_latent=z_target_latent)
            optim_loss = sum(weigh_losses(loss_dict, optim_weights).values())
            optim_loss.mean().backward()
            rank_loss = sum(weigh_losses(loss_dict, self.loss_weights).values())

            best_idx = torch.argmin(rank_loss)
            detached_cameras = pu.deparameterize_camera(cameras.uncrop()).clone()
            angle_dists = three.quaternion.angular_distance(
                detached_cameras.quaternion, target_obs.camera.quaternion).squeeze()
            translation_dists = torch.norm(detached_cameras.translation
                                           - target_obs.camera.translation, dim=1).squeeze()
            if self.return_camera_history:
                camera_history.append((rank_loss.detach().cpu(), detached_cameras.cpu()))

            # Save best cameras in ranking list.
            delta = self._track_best_items(ranking, step,
                                           items=detached_cameras.cpu(),
                                           loss=rank_loss)

            pbar.set_description(f"idx={best_idx}, loss={rank_loss[best_idx].item():.04f}"
                                 f", depth={loss_dict['depth'][best_idx].item():.04f}"
                                 f", ov_depth={loss_dict['ov_depth'][best_idx].item():.04f}"
                                 f", mask={loss_dict['mask'][best_idx].item():.04f}"
                                 f", iou={loss_dict['iou'][best_idx].item():.04f}"
                                 f", latent={loss_dict.get('latent', [0.0]*len(cameras))[best_idx]:.04f}"
                                 f", converge={converge_count}"
                                 f", angle={angle_dists[best_idx].item() / math.pi * 180:.02f}°"
                                 f", trans={translation_dists[best_idx].item():.04f}"
                                 f"")

            if self.track_stats:
                self._record_stat_dict(stat_history, {
                    **{f'{k}_loss': v.detach().cpu() for k, v in loss_dict.items()},
                    **{f'{k}_weight': v for k, v in optim_weights.items()},
                    'delta': delta,
                    'converge_count': converge_count,
                    'angle_dist': angle_dists.cpu(),
                    'trans_dist': translation_dists.cpu(),
                    'optim_loss': optim_loss.detach().cpu(),
                    'rank_loss': rank_loss.detach().cpu(),
                    # 'translation_grad': (cameras.translation.grad
                    #                      if cameras.translation.grad is not None
                    #                      else torch.zeros_like(cameras.translation)),
                    # 'rotation_grad': (cameras.log_quaternion.grad
                    #                   if cameras.log_quaternion.grad is not None
                    #                   else torch.zeros_like(cameras.log_quaternion)),
                    # 'viewport_grad': cameras.viewport.grad,
                })

            for i, (optimizer, scheduler) in enumerate(zip(optimizers, schedulers)):
                optimizer.step()
                scheduler.step(rank_loss[i])

            if delta < self.converge_threshold:
                converge_count += 1
            elif delta > self.converge_threshold:
                converge_count = 0

            if converge_count >= self.converge_patience:
                logger.info("convergence threshold reached", step=step, delta=delta,
                            count=converge_count)
                pbar.close()
                break

        return stat_history, camera_history

    @classmethod
    def _record_stat(cls, history, key, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        else:
            value = value.detach().cpu()
        # Make time dimension.
        value = value.squeeze().unsqueeze(0)
        if len(value.shape) > 2:
            for i in range(value.shape[-1]):
                cls._record_stat(history, f'{key}[{i}]', value[..., i])
        else:
            if key in history:
                history[key] = torch.cat((history[key], value), dim=0)
            else:
                history[key] = value

    @classmethod
    def _record_stat_dict(cls, history, d):
        for key, value in d.items():
            cls._record_stat(history, key, value)

    def _render_observation(self, z_obj, camera, **kwargs):
        """Override rendering function since we're directly optimizing the zoomed camera."""
        # camera = camera.zoom(None, self.model.input_size, self.model.camera_dist)
        pred_dict, z_latent = self.model.render_latent_object(
            z_obj, camera.to(self.model.device), return_latent=True)
        z_mask = pred_dict['mask'].squeeze(0)
        z_mask_logits = pred_dict['mask_logits'].squeeze(0)
        # Adjust sigmoid scale and bias.
        z_depth = camera.denormalize_depth(pred_dict['depth'].squeeze(0))

        return z_depth, z_mask, z_mask_logits, z_latent
