from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from lminfer.config import LMInferConfig
from lminfer.model_executor.layers.activation import SiluAndMul
from lminfer.model_executor.layers.layernorm import RMSNorm
from lminfer.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from lminfer.model_executor.layers.logits_processor import LogitsProcessor
from lminfer.model_executor.layers.rotary_embedding import get_rope
from lminfer.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from lminfer.model_executor.model_loader.weight_utils import default_weight_loader


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        assert hidden_act == "silu", f"当前只支持silu,不支持{hidden_act}"
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3Attention(nn.Module):
    def __init(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        rope_theta: float,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim, num_heads, num_kv_heads, bias=qkv_bias
        )
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position,
            base=rope_theta,
        )

    def _split_qkv(
        self, positions: torch.Tensor, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # RoPE需要加在每层attention的q/k上。
        # 相比之下，绝对PE只需要加入在输入的embedding上一次就够了。
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def _naive_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = q.size(0)
        # (num_tokens, num_heads, head_dim) -> (num_heads, num_tokens, head_dim)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        group = q.size(0) // k.size(0)
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
        attn = q @ k.transpose(-1, -2) * self.scaling
        # causal为下三角(包括主对角线)为1的矩阵。
        causal = torch.tril(
            torch.ones(num_tokens, num_tokens, device=q.device, dtype=torch.bool)
        )
        # 上三角变为-inf
        attn = attn.masked_fill(~causal, float("-inf"))
        # 输出的attn为行轴=query,列轴=key
        # dim=-1表示在key维度做softmax，也就是在每一行的方向上做归一化！
        # torch.float等同于torch.float32,torch.double等同于torch.float64
        # 整个模型需要使用float的场景：RoPE,RMSNorm,Attention的softmax
        probs = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        return (probs @ v).transpose(0, 1)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = self._split_qkv(positions, self.qkv_proj(hidden_states))
        attn_output = self._naive_attention(q, k, v)
        return self.o_proj(attn_output.flatten(1, -1))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            rope_theta=_get_rope_theta(config),
            prefix=f"{prefix}.self_attn",
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
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping: ClassVar = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def __init__(self, lminfer_config: LMInferConfig) -> None:
        super().__init__()
        config = lminfer_config.model_config.hf_config
        self.config = config
        self.model = Qwen3Model(config, prefix="model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 共享参数，lm_head与embedding层共享权重
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in self.packed_modules_mapping:
                if weight_name not in name:
                    continue
                # 命中前提：根据规则替换映射权重的名字，调用读取权重方法。
                param = params_dict[name.replace(weight_name, param_name)]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 没命中的情况
                if name not in params_dict:
                    # 权重名字不在models里面
                    continue
                # 没命中的情况，尝试weight_loader方法
                param = params_dict[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)


def _get_rope_theta(config: PretrainedConfig) -> float:
    theta = getattr(config, "rope_theta", None)
    if theta is None and getattr(config, "rope_scaling", None):
        theta = config.rope_scaling.get("rope_theta")
    assert theta is not None, "config里找不到rope_theta"
    return theta
