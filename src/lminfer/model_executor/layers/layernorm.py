import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 转换精度，相当于使用torch.float精度
        x = x.float()
        if residual is not None:
            # float精度相加
            x = x.add_(residual.float())
            residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt为倒数平方根
        x = x.mul_(torch.rsqrt(var + self.variance_epsilon))
        x = x.to(orig_dtype).mul_(self.weight)
        return x if residual is None else (x, residual)
