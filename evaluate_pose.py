import argparse 
import json
import time
from pathlib import Path

import shutil
from typing import Optional

import toml

import math
import structlog
import torch
from torch.backends import cudnn

import latentfusion.pose.estimation as pe
from latentfusion import utils, meshutils
from latentfusion import visualization as viz
from latentfusion.augment import gan_denormalize, gan_normalize
from latentfusion.datasets.bop_new import BOPDataset
from latentfusion.datasets.realsense import RealsenseDataset
from latentfusion.modules.geometry import Camera
from latentfusion.observation import Observation
from latentfusion.pointcloud import load_ply
from latentfusion.pose import camera_metrics, metrics_table_multiple, three
from latentfusion.recon.inference import LatentFusionModel
from latentfusion import pointcloud as pc
from matplotlib import pyplot as plt

logger = structlog.get_logger(__name__)
cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--dataset-type', choices=['lm', 'tless', 'lmo', 'moped', 'lm_format'], required=True)
    parser.add_argument('--input-scene-dir', type=Path, required=True)
    parser.add_argument('--target-scene-dir', type=Path, required=True)
    parser.add_argument('--object-id', type=str, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    parser.add_argument('--base-name', type=str, required=True)

    parser.add_argument('--gpu-id', type=int, default=0)

    parser.add_argument('--num-input-views', type=int, default=16)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int)

    parser.add_argument('--coarse-config', type=Path, required=True)
    parser.add_argument('--refine-config', type=Path, required=True)

    parser.add_argument('--ranking-size', type=int, default=16)
    parser.add_argument('--method', type=str, default='cross_entropy')
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--save-ply', action='store_true')
    parser.add_argument('--save-stats', action='store_true')
    args = parser.parse_args()

    return args


def get_run_name(args):
    checkpoint_name = args.checkpoint_path.parent.name
    return (
        f"{checkpoint_name}"
        f",{args.base_name}"
        f",coarse={args.coarse_config.stem if args.coarse_config else 'none'}"
        f",refine={args.refine_config.stem}"
    )


def load_dataset(args):
    if args.dataset_type in {'lm', 'lmo', 'tless', 'lm_format'}:
        input_dataset = BOPDataset(args.dataset_dir, args.input_scene_dir, object_id=int(args.object_id))
        target_dataset = BOPDataset(args.dataset_dir, args.target_scene_dir, object_id=int(args.object_id))
        pointcloud = input_dataset.load_pointcloud()
        object_scale_to_meters = 1.0 / (1000.0 * target_dataset.object_scale)
    elif args.dataset_type == 'moped':
        pointcloud_path = args.input_scene_dir / 'pointcloud_eval.ply'
        pointcloud = load_ply(pointcloud_path)
        diameter = three.points_bounding_size(pointcloud)
        object_scale = 1.0 / diameter
        pointcloud = pointcloud * object_scale

        input_paths = sorted([x for x in args.input_scene_dir.iterdir() if x.is_dir()])
        input_dataset = RealsenseDataset(input_paths,
                                         image_scale=1.0,
                                         object_scale=object_scale,
                                         odometry_type='open3d')
        target_paths = sorted([x for x in args.target_scene_dir.iterdir() if x.is_dir()])
        target_dataset = RealsenseDataset(target_paths,
                                          image_scale=1.0,
                                          object_scale=object_scale,
                                          odometry_type='open3d',
                                          use_registration=True)
        object_scale_to_meters = 1.0 / target_dataset.object_scale
    else:
        raise ValueError(f"Unknown dataset type {args.dataset_type}")

    return input_dataset, target_dataset, pointcloud, object_scale_to_meters


def main():
    args = get_args()
    device = torch.device(args.gpu_id)
    logger.info("pose evaluation", **vars(args))

    # Load model.
    model = LatentFusionModel.from_checkpoint(args.checkpoint_path, device=device)
    input_dataset, target_dataset, pointcloud, object_scale_to_meters = load_dataset(args)

    run_name = get_run_name(args)
    out_dir = args.out_dir / args.dataset_type / run_name / args.scene_name

    # Create output path.
    logger.info("creating output directory", path=out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save args.
    with open(out_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2, cls=utils.MyEncoder)

    shutil.copy(args.coarse_config, out_dir / 'coarse_config.toml')
    shutil.copy(args.refine_config, out_dir / 'refine_config.toml')

    input_obs = Observation.from_dataset(input_dataset, inds=input_dataset.sample_evenly(16))
    input_obs_pp = model.preprocess_observation(input_obs)

    # Construct latent object.
    with torch.no_grad():
        z_obj = model.build_latent_object(input_obs_pp)
        y, z = model.render_latent_object(z_obj, input_obs_pp.camera.to(device),
                                          return_latent=False)

    recon_error = (y['depth'].detach().cpu() - input_obs_pp.depth).abs()
    logger.info("reconstructed object", recon_error=recon_error.mean().item())
    viz.plot_image_batches(out_dir / 'input.png', [
        ('Ground Truth Color', gan_denormalize(input_obs_pp.color)),
        ('Ground Truth Depth', viz.colorize_depth(input_obs_pp.depth)),
        ('Ground Truth Mask', viz.colorize_tensor(input_obs_pp.mask)),
        ('Dilated Color', gan_denormalize(input_obs_pp.color)),
        ('Dilated Depth', viz.colorize_depth(input_obs_pp.depth)),
        ('Dilated Mask', viz.colorize_tensor(input_obs_pp.mask)),
        ('Predicted Mask', viz.colorize_tensor(y['mask'].detach().cpu())),
        ('Predicted Depth', viz.colorize_depth(y['depth'].detach().cpu())),
        ('Depth Error', viz.colorize_tensor(recon_error)),
    ], num_cols=3)
    del y, z

    coarse_estimator = pe.load_from_config(args.coarse_config, model)
    refine_estimator = pe.load_from_config(args.refine_config, model, track_stats=True)

    for i, item in enumerate(target_dataset):
        if i < args.start_frame:
            continue
        if args.end_frame and i > args.end_frame:
            logger.info('end frame reached', end_frame=args.end_frame, i=i)
            break

        prefix = f'pose_{i:04d}'
        out_table_path = out_dir / f'{prefix}.txt'
        out_json_path = out_dir / f'{prefix}.json'
        if out_json_path.exists():
            continue
        logger.info("estimating pose", frame=i, total_frames=len(target_dataset))
        target_obs = Observation.from_dict(item)

        tic = time.time()
        coarse_camera = coarse_estimator.estimate(z_obj, target_obs)
        coarse_time = time.time() - tic
        _visualize_pose(model, z_obj, input_obs, target_obs, coarse_camera,
                        out_path=out_dir / f'{prefix}_coarse.png')
        logger.info('estimated coarse pose', time=coarse_time)

        tic = time.time()
        refined_camera, stat_history = refine_estimator.estimate(z_obj, target_obs, camera=coarse_camera)
        refine_time = time.time() - tic
        _visualize_pose(model, z_obj, input_obs, target_obs, refined_camera,
                        out_path=out_dir / f'{prefix}_refined.png')
        logger.info('refined pose', time=refine_time)

        coarse_metrics = camera_metrics(target_obs.camera, coarse_camera[0], pointcloud,
                                        object_scale_to_meters)
        refined_metrics = camera_metrics(target_obs.camera, refined_camera[0], pointcloud,
                                         object_scale_to_meters)
        if args.save_ply:
            pc_gt = three.transform_coords(pointcloud, target_obs.camera.extrinsic)
            pc_coarse = three.transform_coords(pointcloud, coarse_camera[0].extrinsic.detach().cpu())
            pc_refined = three.transform_coords(pointcloud, refined_camera[0].extrinsic.detach().cpu())
            _save_pointclouds(out_dir / f'{prefix}_coarse.ply', pc_gt, pc_coarse)
            _save_pointclouds(out_dir / f'{prefix}_refined.ply', pc_gt, pc_refined)
        if args.save_stats:
            torch.save(stat_history, out_dir / f'{prefix}_stats.pth')
            _plot_stats(out_dir / f'{prefix}_stats.png', stat_history, object_scale_to_meters)

        table = metrics_table_multiple([
            coarse_metrics,
            refined_metrics
        ], ['coarse', 'refined'], tablefmt='github')

        print(f"*** Frame {i} Metrics ***")
        print(table)

        # Save results.
        with open(out_table_path, 'w') as f:
            print(table, file=f)

        with open(out_json_path, 'w') as f:
            json.dump({
                'gt_camera': target_obs.camera.to_kwargs(),
                'coarse': {
                    'metrics': coarse_metrics,
                    'camera': coarse_camera.to_kwargs(),
                    'time': coarse_time,
                },
                'refined': {
                    'metrics': refined_metrics,
                    'camera': refined_camera.to_kwargs(),
                    'time': refine_time,
                }
            }, f, indent=2, cls=utils.MyEncoder)


def _visualize_pose(model, z_obj, input_obs, target_obs, camera, out_path, count=4):
    cameras_zoom = camera.zoom(None, model.camera_dist, model.input_size)
    cameras_zoom = cameras_zoom[:count]
    # pred_y, pred_z = model.render_latent_object(z_obj, cameras_zoom.to(model.device))
    pred_y, _ = model.render_ibr_basic(z_obj, input_obs.to(model.device),
                                       cameras_zoom.to(model.device),
                                       return_latent=False, p=4.0)

    pred_mask = pred_y['mask']
    pred_depth = pred_y['depth']
    pred_color = pred_y['color']
    pred_color = gan_denormalize(pred_color * pred_mask)
    pred_color, _ = cameras_zoom.uncrop(pred_color)
    pred_depth = cameras_zoom.denormalize_depth(pred_depth) * pred_mask
    pred_depth, _ = cameras_zoom.uncrop(pred_depth)
    pred_mask, _ = cameras_zoom.uncrop(pred_mask)
    pred_depth_error = (target_obs.prepare().depth - pred_depth.detach().cpu()).abs()
    pred_mask_error = (target_obs.prepare().mask - pred_mask.detach().cpu()).abs()

    dist = target_obs.camera.translation[0, -1]
    cmin = dist - 1.0
    cmax = dist + 1.0
    viz.plot_image_batches(out_path, (
        ('Ground Truth Color', target_obs.color),
        ('IBR Rerendered Color', pred_color),
        (None, None),
        ('Ground Truth Depth', viz.colorize_tensor(target_obs.depth,
                                                   cmin=cmin, cmax=cmax)),
        ('Estimated Depth', viz.colorize_tensor(pred_depth.detach().cpu(),
                                                cmin=cmin, cmax=cmax)),
        ('Estimated Depth Error', viz.colorize_tensor(pred_depth_error)),
        ('Ground Truth Mask', viz.colorize_tensor(target_obs.mask)),
        ('Estimated Mask', viz.colorize_tensor(pred_mask.detach().cpu())),
        ('Estimated Mask Error', viz.colorize_tensor(pred_mask_error)),
    ), num_cols=3, size=10)


def _plot_gradients(path, stat_history):
    logger.info('saving gradient plots', path=path)
    fig = viz.plot_grid(4, figsize=(20, 10), plots=[
        *[
            viz.Plot(title=f'translation[{i}].grad',
                     args=[stat_history[f'translation_grad[{i}]'].numpy()],
                     func='plot')
            for i in range(3)
        ],
        None,
        *[
            viz.Plot(title=f'rotation_grad[{i}].grad',
                     args=[stat_history[f'rotation_grad[{i}]'].numpy()],
                     func='plot')
            for i in range(3)
        ],
        None,
        *[
            viz.Plot(title=f'viewport_grad[{i}].grad',
                     args=[stat_history[f'viewport_grad[{i}]'].numpy()],
                     func='plot')
            for i in range(4)
        ],
    ])
    fig.savefig(path)
    plt.close('all')


def _plot_stats(path, stat_history, object_scale_to_meters):
    logger.info('saving stat plots', path=path)
    under_5deg_5cm = ((stat_history['trans_dist']*object_scale_to_meters < 0.05)
                      & (stat_history['angle_dist']/math.pi*180 < 5))
    fig = viz.plot_grid(4, figsize=(30, 15), plots=[
        viz.Plot('Angular Error', [stat_history['angle_dist']/math.pi*180],
                 params={'ylabel': 'Error (deg)', 'xlabel': 'Iteration'}),
        viz.Plot('Translation Error', [stat_history['trans_dist']*object_scale_to_meters],
                 params={'ylabel': 'Error (m)', 'xlabel': 'Iteration'}),
        viz.Plot('Rank Loss', [stat_history['rank_loss']],
                 params={'ylabel': 'Loss', 'xlabel': 'Iteration'}),
        viz.Plot('Optim Loss', [stat_history['optim_loss']],
                 params={'ylabel': 'Loss', 'xlabel': 'Iteration'}),

        viz.Plot('Depth Loss', [stat_history['depth_loss']],
                 params={'ylabel': 'Loss', 'xlabel': 'Iteration'}),
        viz.Plot('Overlap Depth Loss', [stat_history['ov_depth_loss']],
                 params={'ylabel': 'Loss', 'xlabel': 'Iteration'}),
        viz.Plot('Mask Loss', [stat_history['mask_loss']],
                 params={'ylabel': 'Loss', 'xlabel': 'Iteration'}),
        viz.Plot('IOU Loss', [stat_history['iou_loss']],
                 params={'ylabel': 'Loss', 'xlabel': 'Iteration'}),

        viz.Plot('<5 deg <5 cm', [under_5deg_5cm],
                 params={'ylabel': 'Success', 'xlabel': 'Iteration'}),
        None,
        None,

        viz.Plot('Depth Weight', [stat_history['depth_weight']], params={'ylabel': 'Weight', 'xlabel': 'Iteration'}),
        viz.Plot('Overlap Depth Weight', [stat_history['ov_depth_weight']], params={'ylabel': 'Weight', 'xlabel': 'Iteration'}),
        viz.Plot('Mask Weight', [stat_history['mask_weight']], params={'ylabel': 'Weight', 'xlabel': 'Iteration'}),
    ])
    fig.savefig(path)
    plt.close('all')


def _save_pointclouds(path, pc_gt, pc_eval):
    green = torch.tensor([[0.427, 0.639, 0.302]]).expand_as(pc_gt)
    yellow = torch.tensor([[0.925, 0.643, 0.000]]).expand_as(pc_gt)
    pc.save_ply(path,
                points=torch.cat((pc_gt, pc_eval)),
                colors=torch.cat((green, yellow)))


if __name__ == '__main__':
    main()
