from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from lminfer.utils import cdiv
from lminfer.v1.core.block_pool import BlockPool
from lminfer.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from lminfer.v1.kv_cache_interface import KVCacheSpec
from lminfer.v1.request import Request


class SingleTypeKVCacheManager(ABC):
    """承接具体的requestid到blockpool之间的数据对接."""

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
    ) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        self.block_pool = block_pool
        self.kv_cache_group_id = kv_cache_group_id
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        # defaultdict表示当字典的key不存在的时候，不会报keyerror，反而会调用list方法返回空的列表
        self.num_cached_block: dict[str, int] = {}

    def get_num_blocks_to_allocate(
        self, request_id: str, num_tokens: int, new_computed_blocks: list[KVCacheBlock]
    ) -> int:
        num_required = cdiv(num_tokens, self.block_size)
        # 需要多少block存下
        num_have = len(self.req_to_blocks[request_id]) + len(new_computed_blocks)
        # 已经分类了多少块+已经缓存了多少块
        return max(0, num_required - num_have)

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: list[KVCacheBlock]
    ) -> None:
        """这个函数在每个requestid里面只会调用一次,且发生前缀缓存的时候才会调用"""
        if request_id in self.num_cached_block:
            assert not new_computed_blocks, "前缀缓存只能发生在请求命中时候"
            return
        blocks = self.req_to_blocks[request_id]
        blocks.extend(new_computed_blocks)
        self.num_cached_block[request_id] = len(new_computed_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        """需要新增新的block的时候调用"""
        blocks = self.req_to_blocks[request_id]
        num_new = max(0, cdiv(num_tokens, self.block_size) - len(blocks))
        # 这里是计算需要额外分配的blocks数量
        if num_new == 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new)
        blocks.extend(new_blocks)
        return new_blocks

    def cache_blocks(
        self, request: Request, block_hashes: list[BlockHash], num_tokens: int
    ) -> None:
        """num_tokens是值已经确定的token数。把num_tokens个token里面新写满的块更新到前缀缓存。"""
        num_cached = self.num_cached_block[request.request_id]
        # num_cached: 块表里面已经登记到前缀字典的有几块?
        num_full_blocks = num_tokens // self.block_size
        if num_full_blocks <= num_cached:
            # TODO: 全命中的情况下这里的小于号可能会成立！
            # 这里有一个问题：上游vllm也是这么做的，既然全部命中需要删除最后一个token，那是不是应该把最后一个block删掉：
            # 首先，约定就是num_tokens单调不减，缓存增量登记,此外，仅仅是多调度一次token，输入都是不变的，因此算出来的值大概率不变。
            # 因此这里就没必要更新kvcache了，而且，kvcache不能被二次更新
            return
        self.block_pool.cache_full_blocks(
            blocks=self.req_to_blocks[request.request_id],
            block_hashes=block_hashes,
            num_cached_blocks=num_cached,
            num_full_blocks=num_full_blocks,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        # 这里的reversed，假设某个序列kvcache:abc->变为cba,然后放到block_pool链表里面顺序就算cba
        # 由于block_pool分配新block用的是popleft，因此c最先被回收重新利用。
        self.block_pool.free_blocks(list(reversed(blocks)))

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.req_to_blocks[request_id]

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_num_blocks: int,
        kv_cache_group_id: int,
        block_pool: BlockPool,
    ) -> list[KVCacheBlock]: ...

    # 这里的...表示方法体没实现，千万不能加入pass/raise Error，否则abstractmethod没用了。


class FullAttentionManager(SingleTypeKVCacheManager):
    """标准因果注意力"""

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_num_blocks: int,
        kv_cache_group_id: int,
        block_pool: BlockPool,
    ) -> list[KVCacheBlock]:
        hit: list[KVCacheBlock] = []
        for block_hash in block_hashes[:max_num_blocks]:
            block = block_pool.get_cached_block(block_hash, kv_cache_group_id)
            if block is None:
                break
            hit.append(block)
        return hit
