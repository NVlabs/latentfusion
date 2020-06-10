"""
Uses farthest point sampling to create a evenly distributed pointcloud.
To be used to compute evaluation metrics like ADD or ADD-S.
"""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from latentfusion.datasets.realsense import RealsenseDataset
from latentfusion.meshutils import Object3D
from latentfusion.three.utils import farthest_points
from latentfusion.pointcloud import save_ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, required=True)
    # parser.add_argument('--object-id', type=str, required=True)
    args = parser.parse_args()

    for object_dir in args.dataset_dir.iterdir():
        print(object_dir)
        ref_dir = object_dir / 'reference'
        obj_path = ref_dir / 'integrated_raw.obj'
        if not obj_path.exists():
            continue
        points = torch.tensor(Object3D(obj_path).vertices, dtype=torch.float32)
        print(points.shape)
        # paths = sorted(x for x in ref_dir.iterdir() if x.is_dir())
        # dataset = RealsenseDataset(paths,
        #                            image_scale=1.0,
        #                            object_scale=1.0,
        #                            center_object=False,
        #                            odometry_type='open3d')
        # points = dataset.points
        _, inds = farthest_points(points, n_clusters=4096, dist_func=F.pairwise_distance,
                                  return_center_indexes=True)
        points = points[inds]

        save_ply(object_dir / 'reference/pointcloud_eval.ply', points)


if __name__ == '__main__':
    main()