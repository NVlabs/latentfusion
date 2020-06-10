import math

import torch

from latentfusion import three
from latentfusion.three import quaternion as q


def spiral_orbit(n, c=16):
    phi = torch.linspace(0, math.pi, n)
    theta = c * phi
    rot_quat = q.from_spherical(phi, theta)
    return rot_quat


def _check_up(up, n):
    if not torch.is_tensor(up):
        up = torch.tensor(up, dtype=torch.float32)
    if len(up.shape) == 1:
        up = up.expand(n, -1)
    return three.normalize(up)


def _is_ray_in_segment(ray, up, min_angle, max_angle):
    angle = torch.acos(three.inner_product(up, ray))
    return (min_angle <= angle) & (angle <= max_angle)


def sample_segment_rays(n, up, min_angle, max_angle):
    up = _check_up(up, n)

    # Sample random new 'up' orientation.
    rays = three.normalize(torch.randn(n, 3))
    num_invalid = n
    while num_invalid > 0:
        valid = _is_ray_in_segment(rays, up, min_angle, max_angle)
        num_invalid = (~valid).sum().item()
        rays[~valid] = three.normalize(torch.randn(num_invalid, 3))

    return three.normalize(rays)


def sample_hemisphere_rays(n, up):
    """
    Samples a ray in the upper hemisphere (defined by `up`).

    Implemented by sampling a uniform random ray on the unit sphere and reflecting
    the vector to be on the same side as the up vector.

    Args:
        n (int): number of rays to sample
        up (tensor, tuple): the up direction defining the hemisphere

    Returns:
        (tensor): `n` rays uniformly sampled on the hemisphere

    """
    up = _check_up(up, n)

    # Sample random new 'up' orientation.
    rays = three.normalize(torch.randn(n, 3))

    # Reflect to upper hemisphere.
    dot = (up * rays).sum(dim=-1)
    rays[dot < 0] = rays[dot < 0] - 2 * dot[dot < 0, None] * up[dot < 0]

    return rays


def random_quat_from_ray(forward, up=None):
    """
    Sample uniformly random quaternions that orients the camera forward direction.

    Args:
        forward: a vector representing the forward direction.

    Returns:

    """
    n = forward.shape[0]
    if up is None:
        down = three.uniform_unit_vector(n)
    else:
        up = torch.tensor(up).unsqueeze(0).expand(n, 3)
        up = up + forward
        down = -up
    right = three.normalize(torch.cross(down, forward))
    down = three.normalize(torch.cross(forward, right))

    mat = torch.stack([right, down, forward], dim=1)

    return three.quaternion.mat_to_quat(mat)


def sample_segment_quats(n, up, min_angle, max_angle):
    """
    Sample a quaternion where the resulting `up` direction is constrained to a segment of the sphere.

    This is performed by first sampling a 'yaw' angle and then sampling a random 'up' direction in the segment.

    The sphere segment is defined as being [min_angle,max_angle] radians away from the 'up' direction.

    Args:
        n (int): number of rays to sample
        up (tensor, tuple): the up direction defining sphere segment
        min_angle (float): the min angle from the up direction defining the sphere segment
        max_angle (float): the max angle from the up direction defining the sphere segment

    Returns:
        (tensor): a batch of sampled quaternions
    """
    up = _check_up(up, n)

    yaw_angle = torch.rand(n) * math.pi * 2.0
    yaw_quat = q.from_axis_angle(up, yaw_angle)

    rays = sample_segment_rays(n, up, min_angle, max_angle)

    pivot = torch.cross(up, rays)
    angles = torch.acos(three.inner_product(up, rays))
    quat = q.from_axis_angle(pivot, angles)

    return q.qmul(quat, yaw_quat)


def evenly_distributed_points(n: int, hemisphere=False, pole=(0.0, 0.0, 1.0)):
    """
    Uses the sunflower method to sample points on a sphere that are
    roughly evenly distributed.

    Reference:
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
    """
    indices = torch.arange(0, n, dtype=torch.float32) + 0.5

    if hemisphere:
        phi = torch.acos(1 - 2 * indices / n / 2)
    else:
        phi = torch.acos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5 ** 0.5) * indices

    points = torch.stack([
        torch.cos(theta) * torch.sin(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(phi),
    ], dim=1)

    if hemisphere:
        default_pole = torch.tensor([(0.0, 0.0, 1.0)]).expand(n, 3)
        pole = torch.tensor([pole]).expand(n, 3)
        if (default_pole[0] + pole[0]).abs().sum() < 1e-5:
            # If the pole is the opposite side just flip.
            points = -points
        elif (default_pole[0] - pole[0]).abs().sum() < 1e-5:
            points = points
        else:
            # Otherwise take the cross product as the rotation axis.
            rot_axis = torch.cross(pole, default_pole)
            rot_angle = torch.acos(three.inner_product(pole, default_pole))
            rot_quat = three.quaternion.from_axis_angle(rot_axis, rot_angle)
            points = three.quaternion.rotate_vector(rot_quat, points)

    return points


def evenly_distributed_quats(n: int, hemisphere=False, hemisphere_pole=(0.0, 0.0, 1.0),
                             upright=False, upright_up=(0.0, 0.0, 1.0)):
    rays = evenly_distributed_points(n, hemisphere, hemisphere_pole)
    return random_quat_from_ray(-rays, upright_up if upright else None)


@torch.jit.script
def disk_sample_quats(n: int, min_angle: float, max_tries: int = 64):

    quats = q.random(1)

    num_tries = 0
    while quats.shape[0] < n:
        new_quat = q.random(1)
        angles = q.angular_distance(quats, new_quat)
        if torch.all(angles >= min_angle) or num_tries > max_tries:
            quats = torch.cat((quats, new_quat), dim=0)
            num_tries = 0
        else:
            num_tries += 1

    return quats
