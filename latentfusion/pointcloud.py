import structlog
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from latentfusion import three

logger = structlog.get_logger(__name__)

try:
    import pcl
except ImportError:
    logger.warning("could not import PCL")


def load_ply(path):
    data = PlyData.read(str(path))['vertex']
    return torch.tensor(np.vstack((data['x'], data['y'], data['z'])).T, dtype=torch.float32)


def save_ply(path, points, colors=None):
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if colors is not None:
        if torch.is_tensor(colors):
            colors = colors.cpu().numpy()
        colors = (colors * 255.0).astype(np.uint8)

    points_descr = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if colors is None:
        dtype_descr = points_descr
    else:
        colors_descr = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        dtype_descr = points_descr + colors_descr

    vertex = np.empty(len(points), dtype=dtype_descr)
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    if colors is not None:
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]

    data = PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
            ),
        ],
    )
    data.write(str(path))


def project_pointcloud(camera, points):
    image_points = (camera.obj_to_image @ three.homogenize(points).transpose(1, 2)).transpose(1, 2)
    image_points = three.dehomogenize(image_points)
    return image_points.long()


def compute_point_mask(camera, mask, points):
    mask = mask.squeeze(1).bool()
    height, width = camera.height, camera.width

    image_points = project_pointcloud(camera, points)

    point_mask = ((image_points[..., 0] >= 0)
                  & (image_points[..., 0] < width)
                  & (image_points[..., 1] >= 0)
                  & (image_points[..., 1] < height))

    for i in range(len(camera)):
        valid_points = image_points[i, point_mask[i]]
        in_foreground = mask[i, valid_points[:, 1], valid_points[:, 0]]
        point_mask[i, point_mask[i]] &= in_foreground

    return point_mask


def filter_outliers(points, n_estimators=100, contamination=0.05, type='isolation_forest'):
    if type == 'elliptic':
        clf = EllipticEnvelope(contamination=contamination)
    elif type == 'isolation_forest':
        clf = IsolationForest(n_estimators=n_estimators,
                              contamination=contamination)
    else:
        raise ValueError("Unknown outlier filter type")
    y = clf.fit_predict(points.numpy())
    y = torch.tensor(y)
    num_valid = (y > 0).sum()
    num_filtered = (y <= 0).sum()
    logger.info('filtered points',
                num_filtered=num_filtered.item(), num_valid=num_valid.item())
    return y > 0


def segment_plane(points):
    if torch.is_tensor(points):
        points = points.numpy().astype(np.float32)
    points = np.hstack((points, np.zeros((points.shape[0], 1)))).astype(np.float32)

    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_array(points)
    seg = cloud.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)

    inliers, plane = seg.segment()

    a, b, c, d = plane
    normal = np.array((a, b, c))
    normal /= np.linalg.norm(normal)
    point_on_plane = np.array((0, 0, -d / c))
    ray = point_on_plane
    sign = np.sign(ray.dot(normal))
    below_plane_mask = (sign * (points[:, :3] @ normal) <= 0)

    inlier_mask = np.zeros(points.shape[0], dtype=bool)
    inlier_mask[inliers] = True
    inlier_mask[below_plane_mask] = True

    return inlier_mask, plane
