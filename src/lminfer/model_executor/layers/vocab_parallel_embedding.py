import torch
import torch.nn.functional as F
from torch import nn

from lminfer.model_executor.layers.linear import _TP_RANK, _TP_SIZE


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.tp_size = _TP_SIZE
        self.tp_rank = _TP_RANK
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # -(-a//b)在python中是向上取整的方法！因为python中//表示向下取整
        # 但是这个方法对于其他语言（例如C）不奏效，因为他们可能是向零取整。
        # C语言中推荐使用(a + b - 1) / b;
        self.num_embeddings_per_partition = -(-num_embeddings // self.tp_size)
        self.vocab_start_index = min(
            self.num_embeddings_per_partition * self.tp_rank, num_embeddings
        )
        self.vocab_end_index = min(
            self.vocab_start_index + self.num_embeddings_per_partition, num_embeddings
        )
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        shard_size = self.vocab_end_index - self.vocab_start_index
        if shard_size > 0:
            param.data[:shard_size].copy_(
                loaded_weight.narrow(
                    0,
                    self.vocab_start_index,
                    shard_size,
                )
            )
        # 部分TP超出词表的部分清0，方便all_reduce。
        if shard_size < self.num_embeddings_per_partition:
            param.data[shard_size:].zero_()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, self.weight)


class ParallelLMHead(VocabParallelEmbedding):
    # 输出侧的词表投影,这里只代为保管权重，并不参与前向
    # 这么设计的原因是我们并不需要所有的token去做输出
    # 采样只需要每个序列的最后一个token的logits
    # 这么设计的原因是减小计算量!
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("输出侧词表投影请使用LogitsProcessor")
