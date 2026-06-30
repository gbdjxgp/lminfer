from dataclasses import dataclass

# from transformers.models.qwen3 import Qwen3Config
from lminfer.layers.activation import SiluAndMul
from lminfer.layers.attention import Attention
from lminfer.layers.layernorm import RMSNorm
from lminfer.layers.embedding_head import PrarllelLMHead, VocabParallelEmbedding
from lminfer.layers.rotary_embedding import get_rope
from lminfer.layers.linear import (
    QKVColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
import torch
import torch.nn as nn
from lminfer.utils import deviceinfo, set_context


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 16384
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True  # qkv_bias
    head_dim: int | None = None
    rope_theta: int = 1000000  # base
    rope_scaling: float | None = None
    intermediate_size: int = 4 * 1024
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias=False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = deviceinfo.tp_size
        # total_num_heads
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        # num_kv_heads
        self.total_num_kv_heads = num_kv_heads or num_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        # head_dim
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        self.qkv_proj = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)
        self.qkv_bias = qkv_bias
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

        self.attention = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
    ):
        q, k, v = self.qkv_proj(x).split(
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attention(q, k, v)
        o = self.o_proj(o.flatten(1, -1))
        return o


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        assert hidden_act == "silu"
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
        )

    def forward(self, x):
        x = self.down_proj(self.act_fn(self.gate_up_proj(x)))
        return x


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            x, residual = self.input_layernorm(x), x
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(positions, x)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(positions, x, residual)
        x, _ = self.norm(x, residual)
        return x


class Qwen3ForCasuallLM(nn.Module):
    packed_module_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = PrarllelLMHead(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ):
        logits = self.lm_head(hidden_states)
        return logits


if __name__ == "__main__":
    device = deviceinfo.device()
    print(f"Detected platform={deviceinfo.platform()}, " f"device={device}")
    config = Qwen3Config(
        vocab_size=50257,
        hidden_size=768,
        num_attention_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_hidden_layers=2,
    )
    model = Qwen3ForCasuallLM(config).to(device)
    model.eval()
    with torch.inference_mode():
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=8,
            max_seqlen_k=8,
        )
        input_ids = torch.randint(0, 50257, (16,), device=device)
        positions = torch.cat(
            [
                torch.arange(8, dtype=torch.long, device=device),
                torch.arange(8, dtype=torch.long, device=device),
            ]
        )
        hidden_states = model(input_ids, positions)
        logits = model.compute_logits(hidden_states)
    print(hidden_states.shape)
    print(logits.shape)
