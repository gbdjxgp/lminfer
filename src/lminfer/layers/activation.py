import torch
import torch.nn as nn
import torch.nn.functional as F
from lminfer.utils import device as device_module


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

        if device_module.deviceinfo.is_npu_available():
            import torch_npu

            def npu_forward(x: torch.Tensor) -> torch.Tensor:
                return torch_npu.npu_swiglu(x, dim=-1)

            self.impl = npu_forward
        else:
            self.impl = self.native_forward

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)

    def native_forward(x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
