import torch


def extract_features(x, submodule, layers):
    outputs = []
    for name, module in submodule.named_children():
        x = module(x)
        if name in layers:
            outputs.append(x)
    return outputs


def normalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    if len(tensor.shape) == 4:
        std = std[None, :, None, None]
        mean = mean[None, :, None, None]
    elif len(tensor.shape) == 3:
        std = std[:, None, None]
        mean = mean[:, None, None]
    else:
        raise ValueError(f"Unsupported number of dimensions ({len(tensor.shape)}.")

    return (tensor - mean) / std


def denormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    if len(tensor.shape) == 4:
        std = std[None, :, None, None]
        mean = mean[None, :, None, None]
    elif len(tensor.shape) == 3:
        std = std[:, None, None]
        mean = mean[:, None, None]
    else:
        raise ValueError(f"Unsupported number of dimensions ({len(tensor.shape)}.")

    return (tensor * std) + mean


def unit_normalize(tensor, dim, eps=1e-3):
    return tensor / (eps + torch.norm(tensor, dim=dim, keepdim=True))


def absolute_max_pool(tensor, dim):
    _, index = tensor.abs().max(dim=dim, keepdim=True)
    return torch.gather(tensor, dim, index)
