import torch


def mad(tensor, dim=0):
    median, _ = tensor.median(dim=dim)
    return torch.median(torch.abs(tensor - median), dim=dim)[0]


def mask_outliers_mad(data, m=2.0):
    median = data.median()
    mad = torch.median(torch.abs(data - median))
    mask = torch.abs(data - median) / mad < m
    return mask


def reject_outliers_mad(data, m=2.0):
    return data[mask_outliers_mad(data, m)]


def mask_outliers(data, m=2.0):
    mean = data.mean()
    std = torch.std(data)
    mask = torch.abs(data - mean) / std < m
    return mask


def reject_outliers(data, m=2.0):
    return data[mask_outliers(data, m)]


def robust_mean(data, m=2.0):
    return torch.mean(reject_outliers(data, m))


def robust_mean_mad(data, m=2.0):
    return torch.mean(reject_outliers_mad(data, m))
