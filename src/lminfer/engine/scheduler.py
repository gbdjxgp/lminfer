from collections import deque

from lminfer.config import Config
from lminfer.engine.sequence import Sequence, SequenceStatus
from lminfer.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # 最大一次前向的序列数(decode)
        self.max_num_seqs = config.max_num_seqs
        # 一次前向最大的token数
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 这个是个int
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # 创建block_manager
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # seq->add_to_waiting->schedule->prefill->...->move_to_running->decode->finish.
        scheduled_seqs = []
        num_batched_tokens = 0

        # 当前有waiting(prefill)队列,并且当前队列数小于最大队列数
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 按照waiting队列依次取出seq
            seq = self.waiting[0]
            # 剩余的可以调度的token数量
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 还没有block_table,优先考虑分配block_table
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # kvcache不够，直接返回。
                    break
                # num_tokens: 还需要调度的token数量.
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已经有block_table,说明是chunked-prefill阶段,只需要调度剩余的token数量
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                remaining < num_tokens and scheduled_seqs
            ):  # only allow chunked prefill for the first seq

                # 剩余的token数量小于规划的token数量，并且scheduler里面有已经调度的seq，证明没必要调度后续的seqs了
                break
            if not seq.block_table:
                # 尝试分配对应seq的block_table，前面已经通过can_allocate检查了，因此这个必定成功
                self.block_manager.allocate(seq, num_cached_blocks)
            # 这里remaining有可能>=num_tokens,也有可能<num_tokens,因此取min
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            # 更新当前batch的token总数
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # seq.num_tokens：当前这个序列总共有多少 token
                # seq.num_cached_tokens：其中已经做完并进入 KV cache 的 token 数
                # seq.num_scheduled_tokens：这一轮准备再计算多少 token
                # 说明此轮之后，seq将变为decode序列，因此需要更新序列状态
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 更新调度
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # 优先prefill
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 这里与prefill不同，直接popleft取seq
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 这里表示发生抢占,空闲block不够这个seq进行decode,因此释放空间
                if self.running:
                    # running非空,抢占其他的seq,这里按照先到先得的顺序,先进入的请求优先被处理
                    self.preempt(self.running.pop())
                else:
                    # running为空,当前只有这个seq,只能抢占自己.
                    self.preempt(seq)
                    break
            else:
                # kvcache够用的情况
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 看情况分配kvcache
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        # 不管是prefill还是decode，走到这里肯定是非空的
        assert scheduled_seqs
        # 使用reversed反复更新running队列
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 被抢占，更新seq状态、回到prefill、释放显存、更新调度队列
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # seqs: 需要处理的seqs，token_ids:当前seqs的结果，长度为len(seqs)
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            # 更新num_cached_tokens/num_scheduled_tokens
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # 增量prefill的情况，token_id被丢弃
                continue
            # 其他情况(prefill完成/decode的情况)
            # 更新token_id
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                # 当前序列完成的情况
                seq.status = SequenceStatus.FINISHED
                # 这里的deallocate并不会清空hash映射，但是会释放块以便给其他请求用
                self.block_manager.deallocate(seq)
                # 删除队列
                self.running.remove(seq)
