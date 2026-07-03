import torch
from lminfer.utils import get_context
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from lminfer.utils import deviceinfo


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.tp_size = deviceinfo.tp_size
        self.tp_rank = deviceinfo.tp_rank
        # 总共的词表大小
        self.num_embeddings = num_embeddings
        # pad之后的词表大小，可以被tp_size整除
        self.padded_num_embeddings = (
            (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        )
        # 每个GPU维护的词表大小
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.vocab_start_idx = min(
            self.num_embeddings_per_partition * self.tp_rank, self.num_embeddings
        )
        self.vocab_end_idx = min(
            self.vocab_start_idx + self.num_embeddings_per_partition,
            self.num_embeddings,
        )
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        shard_size = self.vocab_end_idx - self.vocab_start_idx
        start_idx = self.vocab_start_idx
        # actual_start = min(offset, self.num_embeddings)
        # actual_end = min(offset + shard_size, self.num_embeddings)
        # actual_size = actual_end - actual_start
        if shard_size > 0:
            # 有权重需要读取
            sharded_weights = loaded_weights.narrow(0, start_idx, shard_size)
            param_data.narrow(dim=0, start=0, length=shard_size).copy_(sharded_weights)
        if shard_size < self.num_embeddings_per_partition:
            param_data.narrow(
                0, shard_size, self.num_embeddings_per_partition - shard_size
            ).zero_()

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y *= mask.unsqueeze(-1)
            dist.all_reduce(y, op=dist.reduceOp.SUM)
        return y


class PrarllelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        # 这里正好是借用了VocabParallelEmbedding
        # num_embeddings=vocab_size
        # embedding_dim=hidden_size
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # x: (total_tokens,hidden_size)
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            # x: (total_tokens,hidden_size) -> (num_seqs,hidden_size)
            x = x[last_indices].contiguous()
        # self.weight: (vocab_size, hidden_size),F.linear:x @ weight.T
        # logits: (num_seqs, vocab_size)
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 创建gather_list,非0的卡只发送数据,不接收数据.
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, gather_list=all_logits, dst=0)
            logits = torch.cat(all_logits, dim=-1) if self.tp_rank == 0 else None
            # logits: (num_seqs, num_embeddings_per_partition)
            logits = logits[..., : self.num_embeddings]
            # logits: (num_seqs, num_embeddings)
        return logits
