import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps=1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps)).mul_(self.weight)
        return x

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x.add_(residual)
        return self.rms_forward(x), x

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
