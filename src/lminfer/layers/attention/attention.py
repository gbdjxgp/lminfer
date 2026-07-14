import torch
import torch.nn as nn

from lminfer.utils import device as device_module


class AttentionBackend(nn.Module):
    """Base class for platform-specific attention implementations."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class Attention(nn.Module):
    """Facade that exposes a stable module API and delegates to a backend."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        if device_module.deviceinfo.is_npu_available():
            from .npu_attention import NPUAttentionBackend

            backend_cls = NPUAttentionBackend
        else:
            from .triton_attention import TritonAttentionBackend

            backend_cls = TritonAttentionBackend

        self.backend = backend_cls(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return self.backend(q, k, v)
