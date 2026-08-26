from __future__ import annotations

from abc import ABC, abstractmethod

from lminfer.v1.core.block_pool import BlockPool
from lminfer.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from lminfer.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
)
from lminfer.v1.kv_cache_interface import KVCacheConfig
from lminfer.v1.request import Request


class KVCacheCoordinator(ABC):
    def __init__(self, kv_cache_config: KVCacheConfig, enable_caching: bool) -> None:
        # TODO: kv_cache_config的kv_cache_tensors好像创建了之后没被用到。
        self.kv_cache_config = kv_cache_config
        self.enable_caching = enable_caching
        self.block_pool = BlockPool(kv_cache_config.num_blocks, enable_caching)
        self.single_type_managers: tuple[SingleTypeKVCacheManager, ...] = tuple(
            FullAttentionManager(group.kv_cache_spec, self.block_pool, i)
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[list[KVCacheBlock], ...],
    ) -> int:
        return sum(
            manager.get_num_blocks_to_allocate(
                request_id, num_tokens, new_computed_blocks[i]
            )
            for i, manager in enumerate(self.single_type_managers)
        )

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: tuple[list[KVCacheBlock], ...]
    ) -> None:
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_blocks(request_id, new_computed_blocks[i])

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> tuple[list[KVCacheBlock], ...]:
        return tuple(
            manager.allocate_new_blocks(request_id, num_tokens)
            for manager in self.single_type_managers
        )

    def cache_blocks(
        self, request: Request, block_hashes: list[BlockHash], num_tokens: int
    ) -> None:
        for manager in self.single_type_managers:
            manager.cache_blocks(request, block_hashes, num_tokens)

    def free(self, request_id: str) -> None:
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        return tuple(
            manager.get_blocks(request_id) for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """返回各group命中的块，命中的token数。"""


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(kv_cache_config, enable_caching=False)
        self.num_groups = len(self.single_type_managers)

    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in range(self.num_groups)), 0

    def cache_blocks(
        self, request: Request, block_hashes: list[BlockHash], num_tokens: int
    ) -> None:
        pass


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """只有一个kvcache group"""

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(kv_cache_config, enable_caching=True)
        assert len(self.single_type_managers) == 1
        self.manager = self.single_type_managers[0]
        self.block_size = self.manager.block_size

    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit = type(self.manager).find_longest_cache_hit(
            block_hashes=block_hashes,
            max_num_blocks=max_cache_hit_length // self.block_size,
            kv_cache_group_id=0,
            block_pool=self.block_pool,
        )
        return (hit,), len(hit) * self.block_size


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig, enable_caching: bool
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(kv_cache_config)
    assert len(kv_cache_config.kv_cache_groups) == 1, "多 kv cache group暂未实现"
    return UnitaryKVCacheCoordinator(kv_cache_config)
