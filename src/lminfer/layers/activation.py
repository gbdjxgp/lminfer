import torch.nn as nn
import torch.nn.functional as F
import torch


class SiluAndMul(nn.Module):
    def __init__(self):
        super(SiluAndMul, self).__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
