import torch


@torch.jit.script
def bvmm(a, b):
    if a.shape[0] != b.shape[0]:
        raise ValueError("batch dimension must match")
    if a.shape[1] != b.shape[1]:
        raise ValueError("view dimension must match")

    nbatch, nview, nrow, ncol = a.shape
    a = a.view(-1, nrow, ncol)
    b = b.view(-1, nrow, ncol)
    out = torch.bmm(a, b)
    out = out.view(nbatch, nview, out.shape[1], out.shape[2])
    return out


def bv2b(x):
    if not x.is_contiguous():
        return x.reshape(-1, *x.shape[2:])
    return x.view(-1, *x.shape[2:])


def b2bv(x, num_view=-1, batch_size=-1):
    if num_view == -1 and batch_size == -1:
        raise ValueError('One of num_view or batch_size must be non-negative.')
    return x.view(batch_size, num_view, *x.shape[1:])


def vcat(tensors, batch_size):
    tensors = [b2bv(t, batch_size=batch_size) for t in tensors]
    return bv2b(torch.cat(tensors, dim=1))


def vsplit(tensor, sections):
    num_view = sum(sections)
    tensor = b2bv(tensor, num_view=num_view)
    return tuple(bv2b(t) for t in torch.split(tensor, sections, dim=1))
