from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeVar

from lminfer.config.cache import CacheConfig
from lminfer.config.compilation import CompilationConfig, CUDAGraphMode
from lminfer.config.model import ModelConfig
from lminfer.config.parallel import ParallelConfig
from lminfer.config.scheduler import SchedulerConfig

__all__ = [
    "CUDAGraphMode",
    "CacheConfig",
    "CompilationConfig",
    "LMInferConfig",
    "ModelConfig",
    "ParallelConfig",
    "SchedulerConfig",
    "get_current_config",
    "get_layers_from_config",
    "set_current_config",
]

_T = TypeVar("_T")


@dataclass
class LMInferConfig:
    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)

    def __post_init__(self):
        self.scheduler_config.max_model_len = self.model_config.max_model_len

        self.scheduler_config.max_num_batched_tokens = max(
            self.scheduler_config.max_num_batched_tokens,
            self.scheduler_config.max_num_seqs,
        )

        self.compilation_config.init_with_sizes(self.scheduler_config.max_num_seqs)

    @property
    def max_num_tokens(self) -> int:
        """单步前向的最大token数"""
        return max(
            self.scheduler_config.max_num_batched_tokens,
            self.compilation_config.max_capture_size,
        )


_current_config: LMInferConfig | None = None


def get_current_config() -> LMInferConfig:
    assert _current_config is not None
    return _current_config


@contextmanager
def set_current_config(config: LMInferConfig):
    global _current_config
    prev, _current_config = _current_config, config
    try:
        yield
    finally:
        _current_config = prev


def get_layers_from_config(
    config: LMInferConfig, layer_type: type[_T]
) -> dict[str, _T]:
    return {
        name: layer
        for name, layer in config.compilation_config.static_forward_context.items()
        if isinstance(layer, layer_type)
    }
