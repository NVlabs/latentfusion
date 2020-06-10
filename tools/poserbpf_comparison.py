import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib import font_manager as fm
from scipy.interpolate import UnivariateSpline
from scipy.io import loadmat
from skimage import measure
from tqdm.auto import tqdm

import latentfusion
from latentfusion import three
from latentfusion import visualization as viz
from latentfusion.datasets.realsense import RealsenseDataset
from latentfusion.meshutils import Object3D
from latentfusion.modules.geometry import Camera
from latentfusion.observation import Observation
from latentfusion.pointcloud import load_ply
from latentfusion.pose import estimation as pe
from latentfusion.recon.inference import LatentFusionModel
from latentfusion.rendering import Renderer, SceneContext
from latentfusion.videos import PyAVWriter

prop = fm.FontProperties(fname=str(latentfusion.resource_dir / 'fonts/Manrope-Bold.ttf'),
                         weight='bold', size=32)
text_bbox = dict(boxstyle='round', facecolor='black', alpha=0.8, pad=0.5)

local_dir = Path('/fast')
if not local_dir.exists():
    local_dir = Path('/local1')

if not local_dir.exists():
    print('ERROR NO LOCAL DIR')
    exit(1)


def smooth_curve(points, num_samples=75):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]
    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, num_samples)
    return np.vstack([spl(alpha) for spl in splines]).T


def composite_bg(color, mask, mask_gt, bg, opacity=0.7, title=None, title_color='#fff'):
    frame = (mask * (opacity * color + (1.0 - opacity) * bg)
             + (1.0 - mask) * bg * 0.8)

    contour = smooth_curve(measure.find_contours(mask.squeeze().numpy(), 0.8)[0])
    gt_contour = smooth_curve(measure.find_contours(mask_gt.squeeze().numpy(), 0.8)[0])

    height, width = frame.shape[:2]
    with viz.plot_to_array(height * 2, width * 2) as (fig, ax, array):
        ax.axis('off')
        ax.imshow(frame)
        if title:
            ax.text(25.0, 40.0, title, color=title_color, fontproperties=prop,
                    bbox=text_bbox)
        ax.plot(gt_contour[:, 1], gt_contour[:, 0], linewidth=3, color='#0ead69', alpha=0.7)
        ax.plot(contour[:, 1], contour[:, 0], linewidth=4, color='#ee4266', alpha=0.8)
        fig.tight_layout(pad=0)

    return array


def load_poserbpf_camera(mat_path, key='poses'):
    mat = loadmat(mat_path)
    intrinsic = torch.tensor(mat['intrinsic_matrix']).float()
    pose = torch.tensor(mat[key]).squeeze().float()
    quat = pose[:4]
    translation = pose[4:]
    extrinsic = three.to_extrinsic_matrix(translation, quat)
    camera = Camera(intrinsic=intrinsic, extrinsic=extrinsic)
    return camera


def parse_poserbpf_cameras(seq_path):
    mat_paths = sorted(seq_path.glob('*.mat'))
    #     print(mat_paths)
    inds = [int(x.name.split('.')[0]) for x in mat_paths]
    cameras = []
    for mat in mat_paths:
        cameras.append(load_poserbpf_camera(mat))
    return Camera.cat(cameras)


def get_seq_dataset(path, pointcloud, object_scale=1.0):
    dataset = RealsenseDataset([path],
                               image_scale=1.0,
                               object_scale=object_scale,
                               odometry_type='open3d',
                               center_object=True,
                               ref_points=pointcloud,
                               use_registration=True)
    return dataset


def process_seq(renderer, model, coarse_estimator, refine_estimator, z_obj, out_dir, poserbpf_seq_path, moped_seq_path, pointcloud,
                object_scale):
    object_id = moped_seq_path.parent.parent.name
    prefix = f"{object_id}_seq{moped_seq_path.name}"
    print(prefix)
    video_path = out_dir / f'{prefix}.mp4'

    if video_path.exists():
        return

    target_dataset = get_seq_dataset(moped_seq_path, pointcloud, object_scale=object_scale)
    target_obs = Observation.from_dataset(target_dataset)

    moped_cameras = []
    refined_camera = None
    for i, obs in enumerate(tqdm(target_obs)):
        print(i)
        if i == 0:
            coarse_camera = coarse_estimator.estimate(z_obj, obs, cameras=refined_camera)
            refine_init = coarse_camera.clone()
        else:
            refine_init = refined_camera.clone()[:2]
        refined_camera = refine_estimator.estimate(z_obj, obs, camera=refine_init)
        moped_cameras.append(refined_camera.cpu())

    torch.save(moped_cameras, out_dir / f"{prefix}.pth")

    mesh = Object3D(
        local_dir / f'kpar/latentfusion/moped/{object_id}/reference/integrated_raw.obj')
    mesh = mesh.recenter()

    poserbpf_cameras = parse_poserbpf_cameras(poserbpf_seq_path)
    context = SceneContext(mesh, intrinsic=poserbpf_cameras.intrinsic[0])
    context.randomize_lights(5, 5, random_color=False)

    with PyAVWriter(video_path, 10) as writer:
        for i, camera in enumerate(
                tqdm(moped_cameras[:min(len(poserbpf_cameras), len(moped_cameras))])):
            mask_gt = target_obs[i].mask.squeeze()
            bg = target_obs[i].color.squeeze().permute(1, 2, 0)

            context.set_pose_from_extrinsic(poserbpf_cameras.extrinsic[i], frame='realsense')
            rbpf_color, rbpf_depth, rbpf_mask = renderer.render(context)
            rbpf_color = composite_bg(rbpf_color, rbpf_mask.unsqueeze(-1), mask_gt, bg,
                                      title='PoseRBPF')

            lf_extrinsic = camera.extrinsic[0]
            lf_extrinsic[:3, 3] /= object_scale
            context.set_pose_from_extrinsic(lf_extrinsic, frame='realsense')
            lf_color, lf_depth, lf_mask = renderer.render(context)
            lf_color = composite_bg(lf_color, lf_mask.unsqueeze(-1), mask_gt, bg,
                                    title='LatentFusion')

            frame = np.hstack([rbpf_color, lf_color])
            writer.put_frame(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument('--num-views', type=int, default=8)
    parser.add_argument('--coarse-config', type=Path, required=True)
    parser.add_argument('--refine-config', type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint_path

    model = LatentFusionModel.from_checkpoint(checkpoint_path, device)
    renderer = Renderer(640, 480)

    poserbpf_dir = Path('/projects/grail/kpar/latentfusion/poserbpf_results/')
    moped_dir = Path(local_dir, 'kpar/latentfusion/moped')
    out_dir = args.out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    num_input_views = args.num_views
    object_ids = [
        'black_drill',
        'cheezit',
        'duplo_dude',
        'duster',
        'graphics_card',
        'orange_drill',
        'pouch',
        'remote',
        'rinse_aid',
        'toy_plane',
        'vim_mug',
    ]

    for object_id in object_ids:
        print(f"processing {object_id}")
        poserbpf_seq_dir = poserbpf_dir / object_id / 'evaluation'
        moped_seq_dir = moped_dir / object_id / 'evaluation'
        seq_paths = sorted(moped_seq_dir.iterdir())

        input_scene_dir = moped_dir / object_id / 'reference'
        pointcloud = load_ply(input_scene_dir / 'pointcloud_eval.ply')
        diameter = three.points_diameter(pointcloud)
        object_scale = 1.0 / diameter

        input_paths = [x for x in input_scene_dir.iterdir() if x.is_dir()]
        input_dataset = RealsenseDataset(input_paths,
                                         image_scale=1.0,
                                         object_scale=object_scale,
                                         center_object=True,
                                         odometry_type='open3d',
                                         ref_points=pointcloud)
        input_obs = Observation.from_dataset(input_dataset,
                                             inds=input_dataset.sample_evenly(num_input_views))
        input_obs_pp = model.preprocess_observation(input_obs)

        coarse_estimator = pe.load_from_config(args.coarse_config, model, return_camera_history=False)
        refine_estimator = pe.load_from_config(args.refine_config, model)

        with torch.no_grad():
            z_obj = model.build_latent_object(input_obs_pp)

        for seq_path in seq_paths:
            seq_name = seq_path.name
            process_seq(renderer, model, coarse_estimator, refine_estimator,
                        z_obj, out_dir,
                        poserbpf_seq_dir / seq_name,
                        moped_seq_dir / seq_name,
                        pointcloud,
                        object_scale)


if __name__ == '__main__':
    main()
