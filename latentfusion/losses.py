import torch
from torch import nn

from latentfusion.modules import LayerExtractor


class PerceptualLoss(nn.Module):

    def __init__(self, base, layers, layer_weights, w_act=0.1, reduction='mean'):
        super().__init__()
        self.layers = layers
        self.layer_weights = layer_weights
        self.reduction = reduction
        self.w_act = w_act

        self.features = LayerExtractor(base, self.layers)

    def forward(self, x1, x2):
        feats1 = self.features(x1)
        feats2 = self.features(x2)
        loss = 0
        for w, f1, f2 in zip(self.layer_weights, feats1, feats2):
            f1 = f1.view(f1.size(0), -1)
            f2 = f2.view(f2.size(0), -1)
            loss += w * torch.mean((self.w_act * (f1 - f2)) ** 2, dim=1)

        if self.reduction is not None:
            return reduce_loss(loss, self.reduction)

        return loss


class HardPixelLoss(nn.Module):
    __constants__ = ['k']

    def __init__(self, base_loss, k, reduction='mean', **kwargs):
        super().__init__()
        self.base_loss = base_loss(reduction='none', **kwargs)
        self.k = k
        self.reduction = reduction

    def forward(self, x, y):
        if len(x.shape) > 4:
            x = x.view(-1, *x.shape[-3:])
        if len(y.shape) > 4:
            y = y.view(-1, *y.shape[-3:])

        if len(x.shape) != 4:
            raise ValueError('x must be in BCHW format')
        if len(y.shape) != 4:
            raise ValueError('y must be in BCHW format')

        loss = self.base_loss(x, y)
        loss = reduce_loss(loss, dim=1, reduction=self.reduction).view(x.size(0), -1)
        loss, _ = torch.topk(loss, k=self.k, dim=1, largest=True)

        return reduce_loss(loss, self.reduction)


def reduce_loss(loss, reduction='mean', dim=None):
    if reduction is None:
        return loss
    elif reduction == 'mean':
        if dim is None:
            return loss.mean()
        return loss.mean(dim=dim)
    elif reduction == 'sum':
        if dim is None:
            return loss.sum()
        return loss.sum(dim=dim)

    raise ValueError(f"Unknown reduction {reduction!r}")


def lsgan_loss(input, target, reduction='mean'):
    loss = (input.squeeze() - target) ** 2
    return reduce_loss(loss, reduction=reduction)


def multiscale_lsgan_loss(inputs, target, reduction='mean'):
    loss = 0
    for input in inputs:
        loss += lsgan_loss(input, target, reduction)

    return loss


def _log_beta(alpha, beta):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)


def beta_prior_loss(tensor, alpha, beta, reduction='mean', eps=1e-4):
    loss = ((alpha - 1.0) * torch.log(tensor.clamp(min=eps))
            + (beta - 1.0) * torch.log((1.0 - tensor).clamp(min=eps))
            - _log_beta(alpha, beta))
    loss = (-loss).clamp(min=0)
    return reduce_loss(loss, reduction=reduction)
