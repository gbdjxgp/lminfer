from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xxhash

from lminfer.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)

if TYPE_CHECKING:
    from lminfer.v1.request import Request

BlockHash = int


@dataclass
class BlockHashWithGroupId:
    # 这个类通过dataclass实现了eq方法,同时实现了hash方法.因此上层(BlockPool)可以直接用这个实例化的类作为dict的key
    block_hash: BlockHash
    group_id: int

    def __hash__(self) -> int:
        return hash((self.block_hash, self.group_id))


class KVCacheBlock:
    """链表结构,使用__slots__可以节省内存，缺点是不能动态加入属性"""

    __slots__ = (
        "block_hash",
        "block_id",
        "next_free_block",
        "prev_free_block",
        "ref_cnt",
    )

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash: BlockHashWithGroupId | None = None
        self.prev_free_block: KVCacheBlock | None = None
        self.next_free_block: KVCacheBlock | None = None

    def incr_ref(self) -> None:
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        self.ref_cnt -= 1

    def reset_hash(self) -> None:
        self.block_hash = None

    def __repr__(self) -> str:
        return f"KVCacheBlock(id={self.block_id},ref_cnt={self.ref_cnt})"


class FreeKVCacheBlockQueue:
    """链表结构，表示空闲的块"""

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        self.fake_free_list_head.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
        for block in blocks:
            self._insert_before_tail(block)

    def _insert_before_tail(self, block: KVCacheBlock) -> None:
        last = self.fake_free_list_tail.prev_free_block
        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

    def popleft(self) -> KVCacheBlock:
        block = self.fake_free_list_head.next_free_block
        assert block is not self.fake_free_list_tail, "没有空闲块"
        self.remove(block)
        return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        return [self.popleft() for _ in range(n)]

    def remove(self, block: KVCacheBlock) -> None:
        prev, nxt = block.prev_free_block, block.next_free_block
        prev.next_free_block = nxt
        nxt.prev_free_block = prev
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        self._insert_before_tail(block)
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.append(block)


def hash_block_tokens(
    parent_block_hash: BlockHash | None, curr_block_token_ids: list[int]
) -> BlockHash:
    # xxh64返回的数据为64位的哈希值,哈希上一个块的结果加上本节块的token
    h = xxhash.xxh64()
    if parent_block_hash is not None:
        h.update(parent_block_hash.to_bytes(8, "little"))
    h.update(np.asarray(curr_block_token_ids, dtype=np.int64).tobytes())
    return h.intdigest()


def hash_request_tokens(
    block_size: int, request: Request, start_block_idx: int, end_block_idx: int
) -> list[BlockHash]:
    token_ids = request.all_token_ids
    parent = request.block_hashes[start_block_idx - 1] if start_block_idx else None
    hashes = []
    for i in range(start_block_idx, end_block_idx):
        parent = hash_block_tokens(
            parent, token_ids[i * block_size : (i + 1) * block_size]
        )
        hashes.append(parent)
    return hashes


def get_kv_cache_config(
    kv_cache_specs: list[dict[str, KVCacheSpec]], available_memory: list[int]
) -> KVCacheConfig:
    """根据每个rank上的显存，输出统一的kvcache config"""

    specs = kv_cache_specs[0]
    assert all(s == specs for s in kv_cache_specs), "各rank上面的kv cache规格必须一致。"
    type_ids = {spec.type_id for spec in specs.values()}
    assert len(type_ids) == 1, f"暂时不支持混合层的kv cache:{type_ids}"

    layer_names = list(specs)
    spec = specs[layer_names[0]]

    page_size = spec.page_size_bytes * len(layer_names)
    # 单个block在所有层需要的大小
    num_blocks = min(available_memory) // page_size
    # 根据显存计算可用block数
    assert num_blocks > 0, "显存不足以分配kv cache，请调低gpu_memory_utilization"

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[name])
            for name in layer_names
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
    )
