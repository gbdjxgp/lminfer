import torch
import torch.nn.functional as F
from torch import nn

from lminfer.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


class LogitsProcessor(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        lm_head: ParallelLMHead,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return F.linear(hidden_states, lm_head.weight)
