import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
