from functools import lru_cache
import torch.nn as nn
import torch


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x: (total_tokens, num_heads, head_dim)->2*(total_tokens ,num_heads ,head_dim//2)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # cos/sin: (total_tokens, 1, head_dim//2)
    # out: (total_tokens, num_heads, head_dim//2)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float = 1.0,
    ) -> None:
        super().__init__()
        # inv_freq: (rotary_dim //2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        # positions: (max_position_embeddings)
        positions = torch.arange(max_position_embeddings, dtype=torch.float)
        # freqs: (max_position_embeddings, rotary_dim //2)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        # cos/sin: (max_position_embeddings, rotary_dim //2)
        cos = freqs.cos()
        sin = freqs.sin()

        # cos_sin_cache: (max_position_embeddings, rotary_dim)->(max_position_embeddings, 1, rotary_dim)
        cos_sin_cache = torch.cat([cos, sin], dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query/key: (total_tokens, num_heads ,head_dim)
        cos_sin_cache = self.cos_sin_cache[positions]
        cos, sin = cos_sin_cache.chunk(2, dim=-1)
        return (
            apply_rotary_emb(query, cos, sin),
            apply_rotary_emb(key, cos, sin),
        )


@lru_cache(1)
def get_rope(
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(rotary_dim, max_position, base)
    return rotary_emb
