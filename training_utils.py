import math, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def augment(x, strong=False):
    """CSI replacement for image transforms, fixed before observing target.

    x is [N, amp_phase, subcarrier, time]. Time shifts use edge padding (no
    time reversal); gain acts only on amplitude. Strong adds small noise,
    a time mask <=10%, and a subcarrier mask <=10%. No RGB/color transforms.
    """
    n, _, c, t = x.shape
    limit = 32 if strong else 16
    shifts = torch.randint(-limit, limit + 1, (n, 1, 1, 1), device=x.device)
    indices = (torch.arange(t, device=x.device).view(1, 1, 1, t) + shifts).clamp(0, t - 1)
    out = x.gather(-1, indices.expand(n, 2, c, t)).clone()
    gain = (0.8 + 0.4 * torch.rand(n, 1, 1, device=x.device)) if strong else (0.9 + 0.2 * torch.rand(n, 1, 1, device=x.device))
    out[:, 0] *= gain
    if strong:
        std = out.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        out += 0.02 * std * torch.randn_like(out)
        tlen = torch.randint(0, 81, (n, 1, 1, 1), device=x.device)
        tstart = torch.randint(0, t, (n, 1, 1, 1), device=x.device)
        tidx = torch.arange(t, device=x.device).view(1, 1, 1, t)
        clen = torch.randint(0, 4, (n, 1, 1, 1), device=x.device)
        cstart = torch.randint(0, c, (n, 1, 1, 1), device=x.device)
        cidx = torch.arange(c, device=x.device).view(1, 1, c, 1)
        out.masked_fill_(((tidx >= tstart) & (tidx < tstart + tlen)) |
                         ((cidx >= cstart) & (cidx < cstart + clen)), 0)
    return out

class CycleIndices:
    def __init__(self, length, device):
        self.length, self.device = length, device
        self.order = torch.randperm(length, device=device)
        self.pos = 0

    def take(self, count):
        pieces = []
        while count:
            remaining = min(count, self.length - self.pos)
            pieces.append(self.order[self.pos:self.pos + remaining])
            self.pos += remaining
            count -= remaining
            if self.pos == self.length:
                self.order = torch.randperm(self.length, device=self.device)
                self.pos = 0
        return torch.cat(pieces)

def kernel_groups(model, wd):
    return [dict(params=[p for p in model.parameters() if p.ndim > 1], weight_decay=wd),
            dict(params=[p for p in model.parameters() if p.ndim <= 1], weight_decay=0)]
def bn_snapshot(model):
    return [(b, b.detach().clone()) for b in model.buffers()]

@torch.no_grad()
def restore_buffers(snapshot):
    # .data avoids changing autograd's version for first-pass BatchNorm buffers.
    # Restore the running statistics before the mixed weak/strong pass.
    for target, value in snapshot:
        target.data.copy_(value)
