from dataclasses import dataclass


@dataclass
class CacheConfig:
    block_size: int = 128
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    num_gpu_blocks: int | None = None

    def __post_init__(self):
        assert 0.0 < self.gpu_memory_utilization <= 1.0
        assert self.block_size > 0
