import torch
from torch.multiprocessing import get_context
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        # 总共的此表大小
        self.num_embeddings = num_embeddings
        # pad之后的此表大小，可以被tp_size整除
        self.padded_num_embeddings = (
            (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        )
        # 每个GPU维护的词表大小
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weights):
        param_data = param.data
        shard_size = self.num_embeddings_per_partition
        offset = self.tp_rank * shard_size
        actual_start = min(offset, self.num_embeddings)
        actual_end = min(offset + shard_size, self.num_embeddings)
        actual_size = actual_end - actual_start
        if actual_size > 0:
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            param_data.narrow(0, 0, actual_size).copy_(sharded_weights)
        if actual_size < shard_size:
            param_data.narrow(0, actual_size, shard_size - actual_size).zero_()

    def forward(self, x):
        mask = (
            (x >= (self.tp_rank) * self.num_embeddings_per_partition)
            & (x < (self.tp_rank + 1) * self.num_embeddings_per_partition)
            & (x < self.num_embeddings)
        )
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        output = F.embedding(x, self.weight)
        if self.tp_size > 1:
            output = output * mask.unsqueeze(-1)
            dist.all_reduce(output, op=dist.reduceOp.SUM)
        return output


class PrarllelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x):
        # x:[seq_len,embedding_dim]
        context = get_context()
        if context.is_prefill:
            last_token = context.cu_seqlens_q[1:] - 1
            x = x[last_token].contiguous()
            # x:[num_seqs,embedding_dim]
        logits = F.linear(x, self.weight)
        # logits:[num_seqs,num_embeddings_per_partition]
        if self.tp_size > 1:
            all_logits = (
                [(torch.empty_like(logits) for _ in range(self.tp_size))]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, gather_list=all_logits, dst=0)
            if self.tp_rank == 0:
                logits = torch.cat(all_logits, dim=-1)
                # logits:[num_seqs,num_embeddings_padded]
                logits = logits[..., : self.num_embeddings]
                # logits:[num_seqs,num_embeddings]
        return logits
