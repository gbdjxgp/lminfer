from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, PretrainedConfig

if TYPE_CHECKING:
    from lminfer.config.parallel import ParallelConfig


@dataclass
class ModelConfig:
    """
    存放模型本身的属性
    """

    # 模型路径
    model: str
    dtype: torch.dtype = field(default=None)
    max_model_len: int = 4096
    seed: int | None = None
    hf_config: PretrainedConfig = field(default=None)

    def __post_init__(self):
        assert os.path.isdir(self.model), f"模型路径不存在：{self.model}"

        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.dtype is None:
            self.dtype = self.hf_config.dtype
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )

    @property
    def architecture(self) -> str:
        return self.hf_config.architectures[0]

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    def get_head_size(self) -> int:
        head_dim = getattr(self.hf_config, "head_dim", None)
        if head_dim is not None:
            return head_dim
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_total_num_kv_heads(self) -> int:
        return getattr(
            self.hf_config,
            "num_key_value_heads",
            self.hf_config.num_attention_heads,
        )

    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        return (
            self.hf_config.num_attention_heads // parallel_config.tensor_parallel_size
        )

    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        return max(
            1, self.get_total_num_kv_heads() // parallel_config.tensor_parallel_size
        )

    def get_num_layers(self, parallel_config: ParallelConfig) -> int:
        # 暂未实现PP,因此默认每个rank持有全部层
        return self.hf_config.num_hidden_layers
