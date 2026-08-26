from __future__ import annotations

from collections import defaultdict

from lminfer.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


class BlockPool:
    # 一个大前提(使用原则)请牢记：KVCacheBlock只在借用/归还时变，cached_block_hash_to_block只在内容变化之后才变。
    # TODO: 搞清楚enable_caching的作用
    def __init__(self, num_gpu_blocks: int, enable_caching: bool) -> None:
        assert num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        # 显存块
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # 空闲的块，用Queue(底层链表)管理
        self.cached_block_hash_to_block: dict[
            BlockHashWithGroupId, dict[int, KVCacheBlock]
        ] = defaultdict(dict)
        # 这里注意！cached_block_hash_to_block是个dict，key为实例化后的类，这个类是实现了hash与eq方法的
        # 因此只要block_hash与group_id一样，用这两个值实例化出来的BlockHashWithGroupId就是一个key
        # 此外，cached_block_hash_to_block的值也很有意思，它的value为dict[int, KVCacheBlock]
        # 有同样内容的block可能会有重复，也就是相同的内容但是不同的KVCacheBlock，这里用字典对应
        # 当命中前缀缓存的时候，这些有相同内容的block任意一个KVCacheBlock都可以用
        # 细看这里的KVCacheBlock.block_hash，存储的同样是BlockHashWithGroupId
        # 因此,"value里面有数据结构索引到了key上面"
        # 已经缓存的块，使用dict管理，key为实例化后的类.

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_id: int
    ) -> KVCacheBlock | None:
        cached = self.cached_block_hash_to_block.get(
            BlockHashWithGroupId(block_hash, kv_cache_group_id)
        )
        # 查找缓存的函数
        # cached:dict[int, KVCacheBlock]|None
        if not cached:
            # 没有命中缓存
            return None
        # cached.values()拿到list[KVCacheBlock],然后转为迭代器.
        # 这里先iter转为迭代器，然后next估计就是为了拿到第一个值
        # 这里注意一下,因为dictvalues没法直接索引[0]
        # 直接强制转换list(cached.values())也是可行的,但是估计内存问题?
        return next(iter(cached.values()))

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        assert num_blocks <= self.get_num_free_blocks(), "空闲块不够"
        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            self._maybe_evict_cached_block(block)
            block.incr_ref()
        return blocks

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """入参为新命中的块"""
        for block in blocks:
            if block.ref_cnt == 0:
                # 如果新命中的块ref_cnt为0,证明其在空闲队列里面
                self.free_block_queue.remove(block)
            # 不管是否在空闲队列里面，该块目前都被用了，因此增加引用计数
            block.incr_ref()

    def free_blocks(self, ordered_blocks: list[KVCacheBlock]) -> None:
        for block in ordered_blocks:
            # 释放块，顺序：先释放的先append,先放到空闲块链表的的尾部,先被回收
            block.decr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> None:
        # 该函数目前只有get_new_blocks的时候被调用
        block_hash = block.block_hash
        if block_hash is None:
            # 大多数情况块没有被cached_block_hash_to_block登记过
            # 因此可以什么都不做,直接返回
            return
        cached = self.cached_block_hash_to_block.get(block_hash)
        # 被登记过的情况,用block里面的哈希查外层的dict查找是否在缓存当中
        if cached is not None:
            # 如果在缓存的dict当中，直接pop,这里加入默认值None是防御性编程
            cached.pop(block.block_id, None)
            if not cached:
                # 前面的cached.pop是inplace操作,这里是检查这个pop后的dict[int, KVCacheBlock]是否为空
                # 为空的情况,需要删除整个的block_hash,非空的情况,对应的block_id的KVCacheBlock也已经被我pop走了.啥也不需要做
                del self.cached_block_hash_to_block[block_hash]
        # 最终的效果都是：cached_block_hash_to_block里面不包含这个block
        # 并且block里面没对应的hash
        block.reset_hash()

    def cache_full_blocks(
        self,
        blocks: list[KVCacheBlock],
        block_hashes: list[BlockHash],
        num_cached_blocks: int,
        num_full_blocks: int,
        kv_cache_group_id: int,
    ) -> None:
        # 这里的入参应该提前准备好对应的需要存入的blocks。
        for i in range(num_cached_blocks, num_full_blocks):
            block = blocks[i]
            assert block.block_hash is None, "同一个块不应该被登记两次"
            key = BlockHashWithGroupId(block_hashes[i], kv_cache_group_id)
            block.block_hash = key
            self.cached_block_hash_to_block[key][block.block_id] = block

    def reset_prefix_cache(self) -> None:
        # 删除所有的块缓存
        for block in self.blocks:
            block.reset_hash()
        self.cached_block_hash_to_block.clear()
