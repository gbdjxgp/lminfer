import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, gamma, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(gamma.detach().clone())
        self.eps = eps

    @property
    def gamma(self):
        return self.weight

    @torch.compile
    def rms_forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = x * torch.rsqrt(variance) * self.weight
        return x_norm

    def residual_rms_forward(self, x, residual):
        x = x + residual
        return self.rms_forward(x)

    def forward(self, x, residual=None):
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        return self.rms_forward(x)
