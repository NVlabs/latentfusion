import torch
from torch.nn import functional as F


def cosine_distance(x1, x2, dim=1, eps=1e-8):
    if len(x1.shape) == 1:
        dim = 0

    return 1.0 - torch.cosine_similarity(x1, x2, dim, eps)


def pairwise_distance(x1, x2, metric='cosine', p=2, eps=1e-8):
    if metric == 'cosine':
        return 1.0 - F.cosine_similarity(x1, x2, eps=eps)
    elif metric == 'euclidean':
        return F.pairwise_distance(x1, x2, eps=eps, p=p)
    else:
        raise ValueError(f'Unknown type {metric!r}')


def distance(x1, x2, metric='cosine', p=2, eps=1e-8, dim=0):
    if metric == 'cosine':
        return 1.0 - F.cosine_similarity(x1, x2, eps=eps, dim=dim)
    return torch.norm(x1 - x2, p=p, dim=dim)


def outer_distance(x1, x2, metric='cosine', p=2, eps=1e-8):
    if metric == 'cosine':
        x12 = x1 @ x2.t()
        w1 = torch.norm(x1, dim=1, keepdim=True)
        w2 = torch.norm(x2, dim=1, keepdim=True)
        return 1.0 - x12 / (w1 @ w2.t()).clamp(min=eps)
    elif metric == 'euclidean':
        return torch.cdist(x1, x2)
    elif metric == 'inner':
        return -(x1 @ x2.t())
    elif metric == 'ols_coef':
        x12 = x1 @ x2.t()
        w1 = torch.norm(x1, dim=1, keepdim=True)
        return -(x12 / w1.pow(2).clamp(min=eps))

    raise ValueError(f'Unknown type {metric!r}')
