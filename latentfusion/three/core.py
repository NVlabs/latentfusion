import torch


@torch.jit.script
def acos_safe(t, eps: float = 1e-7):
    return torch.acos(torch.clamp(t, min=-1.0 + eps, max=1.0 - eps))


@torch.jit.script
def ensure_batch_dim(tensor, num_dims: int):
    unsqueezed = False
    if len(tensor.shape) == num_dims:
        tensor = tensor.unsqueeze(0)
        unsqueezed = True

    return tensor, unsqueezed


@torch.jit.script
def normalize(vector, dim: int = -1):
    """
    Normalizes the vector to a unit vector using the p-norm.
    Args:
        vector (tensor): the vector to normalize of size (*, 3)
        p (int): the norm order to use

    Returns:
        (tensor): A unit normalized vector of size (*, 3)
    """
    return vector / torch.norm(vector, p=2.0, dim=dim, keepdim=True)


@torch.jit.script
def uniform(n: int, min_val: float, max_val: float):
    return (max_val - min_val) * torch.rand(n) + min_val


def uniform_unit_vector(n):
    return normalize(torch.randn(n, 3), dim=1)


def inner_product(a, b):
    return (a * b).sum(dim=-1)


@torch.jit.script
def homogenize(coords):
    ones = torch.ones_like(coords[..., 0, None])
    return torch.cat((coords, ones), dim=-1)


@torch.jit.script
def dehomogenize(coords):
    return coords[..., :coords.size(-1) - 1] / coords[..., -1, None]


def transform_coord_grid(grid, transform):
    if transform.size(0) != grid.size(0):
        raise ValueError('Batch dimensions must match.')

    out_shape = (*grid.shape[:-1], transform.size(1))

    grid = homogenize(grid)
    coords = grid.view(grid.size(0), -1, grid.size(-1))
    coords = transform @ coords.transpose(1, 2)
    coords = coords.transpose(1, 2)
    return dehomogenize(coords.view(*out_shape))


@torch.jit.script
def transform_coords(coords, transform):
    coords, unsqueezed = ensure_batch_dim(coords, 2)

    coords = homogenize(coords)
    coords = transform @ coords.transpose(1, 2)
    coords = coords.transpose(1, 2)
    coords = dehomogenize(coords)
    if unsqueezed:
        coords = coords.squeeze(0)

    return coords


@torch.jit.script
def grid_to_coords(grid):
    return grid.view(grid.size(0), -1, grid.size(-1))


def spherical_to_cartesian(theta, phi, r=1.0):
    x = r * torch.cos(theta) * torch.sin(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack((x, y, z), dim=-1)


def points_bound(points):
    min_dim = torch.min(points, dim=0)[0]
    max_dim = torch.max(points, dim=0)[0]
    return torch.stack((min_dim, max_dim), dim=1)


def points_radius(points):
    bounds = points_bound(points)
    centroid = bounds.mean(dim=1).unsqueeze(0)
    max_radius = torch.norm(points - centroid, dim=1).max()
    return max_radius


def points_diameter(points):
    return 2* points_radius(points)


def points_centroid(points):
    return points_bound(points).mean(dim=1)


def points_bounding_size(points):
    bounds = points_bound(points)
    return torch.norm(bounds[:, 1] - bounds[:, 0])
