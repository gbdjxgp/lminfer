from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCacheSpec:
    """单层kvcache规格"""

    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """一个block占用字节数"""
        raise NotImplementedError

    @property
    def type_id(self) -> str:
        """规格相同的层会被分到同一个kv cache group"""
        raise NotImplementedError


@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    """一层的数据组织方式"""

    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    @property
    def page_size_bytes(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * self.dtype.itemsize
        )


@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    """标准因果注意力，每个token需要看到前面的全部kv，cache不会被截断"""

    @property
    def type_id(self) -> str:
        return f"full_attention_{self.block_size}_{self.page_size_bytes}"


@dataclass
class KVCacheTensor:
    """一块要分配的显存，以及哪些层公用它"""

    size: int
    shared_by: list[str]


@dataclass
class KVCacheGroupSpec:
    """规格相同的一组层，共用一套block_table,一般一种注意力对应一个kv cache group"""

    layer_names: list[str]
    kv_cache_spec: KVCacheSpec


@dataclass
class KVCacheConfig:
    """整机的block抽象"""

    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]
