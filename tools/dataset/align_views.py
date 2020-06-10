from pathlib import Path

import cv2
import numpy as np
import argparse
import json

import mayavi.mlab as mlab
import torch
from scipy import linalg

import latentfusion.observation
from latentfusion.datasets.realsense import RealsenseDataset
from latentfusion.modules.geometry import Camera


def backproject(depth_cv, intrinsic_matrix):
    depth = depth_cv.astype(np.float32, copy=True)
    depth[depth == 0] = np.nan

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = Kinv @ x2d.transpose()

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()

    output_dict = {}
    print(x2d.shape, X.shape)
    for x2, x3 in zip(x2d, X):
        output_dict[(x2[0], x2[1])] = x3
    return output_dict


def make_pc_dict(obs):
    points = obs.pointcloud(frame='camera', segment=False, return_colors=False)
    points = points.view(obs.camera.height, obs.camera.width, 3)
    # point_mask = inference.compute_point_mask(obs.camera, obs.mask.bool(), points)
    pc_dict = {}
    for y, x in torch.nonzero(obs.mask.squeeze().bool()).tolist():
        pc_dict[(x, y)] = points[y, x].numpy()

    return pc_dict


def event_handler(event, x, y, flags, params):
    # x = x // 2
    # y = y // 2
    image, kps, width, colors, pc_dicts, kps_3d = params
    if event == cv2.EVENT_LBUTTONDOWN:
        xoffset = 0
        if x <= width:
            index = 0
        else:
            xoffset = width
            index = 1

        print(type(x), type(y))
        coord_3d = pc_dicts[index][(x - xoffset, y)]
        print('3d coord: ', coord_3d)
        if not np.all(np.isfinite(coord_3d)):
            print('invalid point')
            return

        kps[index].append([x - xoffset, y])
        kps_3d[index].append(coord_3d.copy())
        if len(kps[index]) not in colors:
            colors[len(kps[index])] = tuple([int(np.random.randint(0, 255)) for _ in range(3)])

        image = cv2.circle(image, (x, y), 3, colors[len(kps[index])], 1)
        image = cv2.circle(image, (x, y), 0, colors[len(kps[index])], 1)

        cv2.putText(
            image,
            str(len(kps[index])),
            (x - 20, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors[len(kps[index])],
            2,
            cv2.LINE_AA
        )


def inverse_transform(trans):
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


def tensor_to_cv2(tensor):
    return (tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, [2, 1, 0]].copy()


class ImageAligner:
    def __init__(self, ref_obs, tar_obs, K):
        self._image = None
        self._images = [tensor_to_cv2(ref_obs.color), tensor_to_cv2(tar_obs.color)]
        self._depths = [ref_obs.depth.squeeze().numpy().copy(), tar_obs.depth.squeeze().numpy().copy()]

        self._K = K.copy()
        self._unified_image = np.concatenate((self._images[0], self._images[1]), axis=1)
        # self._unified_image = np.vstack([
        #     np.hstack((self._images[0], self._images[1])),
        #     np.hstack((visualization.colorize_numpy(self._depths[0]),
        #                visualization.colorize_numpy(self._depths[1])))
        # ])
        self._width = self._images[0].shape[1]
        self._kps = [[] for _ in range(2)]
        self._kps_3d = [[] for _ in range(2)]
        self._colors = {}
        # self._pc_dicts = [backproject(img, self._K) for img in self._depths]
        self._pc_dicts = [make_pc_dict(ref_obs), make_pc_dict(tar_obs)]

    def plot_valid_points(self, pc_dict, color=(1, 1, 1), prob=1., rt=None):
        pc = np.asarray(list(pc_dict.values()))
        selection = np.logical_and(np.isfinite(pc[:, 2]), np.random.rand(pc.shape[0]) <= prob)
        pc = pc[selection, :]
        if rt is not None:
            pc = pc.dot(rt[:3, :3].T)
            pc += np.expand_dims(rt[:3, 3], 0)

        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], scale_factor=0.004, color=color)

    def plot_points_with_colors(self, points, colors):
        for p, c in zip(points, colors):
            print(c)
            mlab.points3d([p[0]], [p[1]], [p[2]], color=(float(c[2]) / 255, float(c[1]) / 255, float(c[0]) / 255),
                          scale_factor=0.02)

    def label_images(self):
        params = (self._unified_image, self._kps, self._width, self._colors, self._pc_dicts, self._kps_3d)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', event_handler, params)
        while True:
            # cv2.imshow('image', cv2.resize(self._unified_image,
            #                                (self._unified_image.shape[1] * 2,
            #                                 self._unified_image.shape[0] * 2)))
            cv2.imshow('image', self._unified_image)
            key = cv2.waitKey(1)

            if key == 27:
                print('clicking phase is done')
                break

        print(
            'collected {} kps from image 1, and {} kps from image 2'.format(len(self._kps[0]), len(self._kps[1]))
        )
        num_used = min(len(self._kps[0]), len(self._kps[1]))
        for i in range(2):
            self._kps[i] = self._kps[i][:num_used]
            self._kps_3d[i] = self._kps_3d[i][:num_used]

        print('using the first {} points'.format(num_used))

        # self.plot_valid_points(self._pc_dicts[0], prob=0.1)
        # self.plot_points_with_colors(self._kps_3d[0], [self._colors[x+1] for x in range(num_used)])
        # mlab.show()

    def rigid_transform(self):
        pcs = [np.asarray(self._kps_3d[i]).copy() for i in range(2)]
        means = [np.mean(pc, 0, keepdims=True) for pc in pcs]
        pcs_normalized = [pc - m for pc, m in zip(pcs, means)]

        H = pcs_normalized[0].T.dot(pcs_normalized[1])
        u, _, vt = np.linalg.svd(H)
        # R = vt.T.dot(u.T)
        R = u.dot(vt)

        if np.linalg.det(R) < 0:
            print('********* not so sure about this part ************')
            vt[2, :] *= -1
            R = u.dot(vt)

        T = np.squeeze(means[0]) - R.dot(np.squeeze(means[1]))
        print(T.shape)

        output = np.eye(4, dtype=np.float32)
        output[:3, :3] = R
        output[:3, 3] = T

        transformed = pcs[1].dot(output[:3, :3].T)
        transformed += np.expand_dims(output[:3, 3], 0)

        print('error', linalg.norm(pcs[0] - transformed, axis=-1))

        print(output)
        self.plot_valid_points(self._pc_dicts[0], prob=0.1)
        self.plot_valid_points(self._pc_dicts[1], prob=0.1, color=(1, 0, 0), rt=output)
        mlab.show()
        return output


def item_to_obs(item):
    height, width = item['color'].shape[-2:]
    return latentfusion.observation.Observation(item['color'].unsqueeze(0),
                                                item['depth'].unsqueeze(0).unsqueeze(0),
                                                item['mask'].unsqueeze(0).unsqueeze(0).float(),
                                                Camera(intrinsic=item['intrinsic'],
                                        extrinsic=item['extrinsic'],
                                        width=width,
                                        height=height))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='view registerer')
    parser.add_argument(dest='ref_path', type=Path)
    parser.add_argument(dest='tar_path', type=Path)
    parser.add_argument(dest='ref_id', type=int)
    parser.add_argument(dest='tar_id', type=int)
    args = parser.parse_args()

    ref_base = args.ref_path
    ref_id = args.ref_id
    tar_base = args.tar_path
    tar_id = args.tar_id

    assert ref_base.parent.parent == tar_base.parent.parent

    use_registration = int(ref_base.name) > 0

    ref_dataset = RealsenseDataset(ref_base, image_scale=1.0, object_scale=1.0, center_object=False,
                                   odometry_type='open3d', use_registration=use_registration)
    ref_obs = item_to_obs(ref_dataset[int(ref_id)])
    tar_dataset = RealsenseDataset(tar_base, image_scale=1.0, object_scale=1.0, center_object=False,
                                   odometry_type='open3d', use_registration=False)
    tar_obs = item_to_obs(tar_dataset[int(tar_id)])

    with open(ref_base / 'intrinsics.json') as f:
        K = json.load(f)['intrinsic_matrix']
    K = np.asarray(K)
    K = np.reshape(K, [3, 3]).T
    print('camera intrinsics: ', K)

    aligner = ImageAligner(ref_obs, tar_obs, K)
    aligner.label_images()
    transform = aligner.rigid_transform()
    ref_E = ref_obs.camera.extrinsic.squeeze().numpy()
    tar_E = tar_obs.camera.extrinsic.squeeze().numpy()
    transform = inverse_transform(ref_E) @ transform @ tar_E
    out = {
        'reference_id': int(ref_id),
        'reference_frame': str(args.ref_path),
        'target_frame': str(args.tar_path),
        'transform': transform.tolist(),
    }
    print(out)

    out_path = tar_base / 'registration' / 'manual.json'
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w') as f:
        print(f"Writing to {out_path}")
        json.dump(out, f, indent=2)
