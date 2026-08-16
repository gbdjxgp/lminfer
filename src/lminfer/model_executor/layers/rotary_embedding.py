from functools import lru_cache

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        assert is_neox_style
        # https://huggingface.co/blog/zh/designing-positional-encoding#%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81-rope
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        # PyTorch 中，nn.Parameter会参与dict更新被保存到模型的state_dict中。
        # torch.tensor对象不会参与梯度更新，不会保存到state_dict中。
        # register_buffer:不会被优化器导出
        # 行为                |nn.Parameter|register_buffer|torch.Tensor
        # 是否参与梯度计算       是            否                     否
        # model.parameters()   是            否                     否
        # model.state_dict()   是            是(persistent=True)    否
        # .cuda()/.to(device)  是            是                     否
        self.register_buffer(
            "cos_sin_cache",
            cache.to(dtype or torch.get_default_dtype()),
            persistent=False,
        )

    @staticmethod
    def _apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        # 注意，这里的实现用的矩阵是[[cos,-sin],[sin,cos]],这个与前面huggingface的链接是不同的哈
        # 实际工程中实现还是以下面这种为准，都是以逆时针旋转矩阵来推导应用的。
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query/key: (num_tokens, num_heads, head_size)
        # cos/sin: (num_tokens, 1, head_size/2)
        cos, sin = self.cos_sin_cache[positions].unsqueeze(1).chunk(2, dim=-1)
        return self._apply(query, cos, sin), self._apply(key, cos, sin)


@lru_cache
# lru装饰器缓存get_rope结果，只要入参不变，返回的类就是同一个，不需要反复初始化类
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
) -> RotaryEmbedding:
    return RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
    )
