from __future__ import annotations

from dataclasses import dataclass

from lminfer.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from lminfer.v1.core.kv_cache_utils import KVCacheBlock, hash_request_tokens
from lminfer.v1.kv_cache_interface import KVCacheConfig
from lminfer.v1.request import Request


@dataclass
class KVCacheBlocks:
    """外层的tuple表示不同的kv cache group"""

    blocks: tuple[list[KVCacheBlock], ...]

    def __add__(self, other: KVCacheBlocks) -> KVCacheBlocks:
        return KVCacheBlocks(
            tuple(a + b for a, b in zip(self.blocks, other.blocks, strict=True))
        )

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    @classmethod
    def create_empty(cls, num_groups: int) -> KVCacheBlocks:
        return cls(tuple([] for _ in range(num_groups)))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
    ) -> None:
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        self.coordinator = get_kv_cache_coordinator(kv_cache_config, enable_caching)
        self.block_pool = self.coordinator.block_pool

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """只在请求从waiting到running的时候调用一次，计算prompt前多少token可以复用已有的kv"""
        empty = KVCacheBlocks.create_empty(self.num_kv_cache_groups)
        if not self.enable_caching or request.num_computed_tokens > 0:
            # 这里的num_computed_tokens实际上是已经算完的token数量，只有第一次进入的时候才可能为0
            return empty, 0
        self._extend_block_hashes(request, request.num_prompt_tokens)
        # 这里-1是防止prompt正好全部命中，导致没有hidden_state采样的问题
        # TODO:这里是使用num_prompt_tokens完成的，导致发生抢占之后，只能缓存到num_prompt_tokens,缓存不到先前计算出来的tokens
        max_hit_length = (
            (request.num_prompt_tokens - 1) // self.block_size * self.block_size
        )
        blocks, num_hit_tokens = self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_hit_length
        )
        return KVCacheBlocks(blocks), num_hit_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        """准备KV cache槽位"""
        """num_new_tokens：本次要算几个token
        num_new_computed_tokens：本次命中的token数量，为block_size的整数倍
        num_new_computed_tokens为get_computed_blocks的返回值，只会在请求准入的时候有值。
        TODO: 这个函数入参、计算过程、返回值的入参都是乱七八糟的，需要整理。
        """
        assert num_new_tokens > 0
        # 第一次请求准入的时候了，分配new_blocks_list
        new_blocks_list = (
            new_computed_blocks.blocks
            if new_computed_blocks is not None
            else tuple([] for _ in range(self.num_kv_cache_groups))
        )
        # 需要槽位=已有的+本次命中的+本次要算的
        num_tokens_need_slot = min(
            request.num_computed_tokens + num_new_computed_tokens + num_new_tokens,
            self.max_model_len,
        )
        num_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request.request_id, num_tokens_need_slot, new_blocks_list
        )
        # num_to_allocate:排除缓存之后，实际需要分配多少块。
        if num_to_allocate > self.block_pool.get_num_free_blocks():
            # 没空闲的块了。
            return None
        if self.enable_caching:
            # 开启前缀缓存的时候，调用touch函数新建块。
            # 二维遍历之后，实际上是touch(list[KVCacheBlock])
            # TODO: 搞明白前缀缓存是否开启在逻辑上的边界。
            self.block_pool.touch([blk for group in new_blocks_list for blk in group])
        self.coordinator.save_new_computed_blocks(request.request_id, new_blocks_list)
        return KVCacheBlocks(
            self.coordinator.allocate_new_blocks(
                request.request_id, num_tokens_need_slot
            )
        )

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        if not self.enable_caching:
            return
        self._extend_block_hashes(request, num_tokens)
        self.coordinator.cache_blocks(request, request.block_hashes, num_tokens)

    def free(self, request: Request) -> None:
        self.coordinator.free(request.request_id)

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return KVCacheBlocks(self.coordinator.get_blocks(request_id)).get_block_ids()

    def _extend_block_hashes(self, request: Request, num_tokens: int) -> None:
        """增量计算block hash"""
        num_full_blocks = num_tokens // self.block_size
        start = len(request.block_hashes)
        if num_full_blocks > start:
            request.block_hashes.extend(
                hash_request_tokens(self.block_size, request, start, num_full_blocks)
            )
