import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        # logits: (num_seqs, vocab_size)
        # temperature: (num_seqs)->(num_seqs, 1)
        logits = logits / temperature.unsqueeze(-1)
        # probs: batch_size, vocab_size
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
