import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def __init__(self):
        super(SiluAndMul, self).__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
