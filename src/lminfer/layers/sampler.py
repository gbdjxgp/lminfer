import torch
import torch.nn as nn

class SamplerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,logits,temperature):
        # logits: batch_size, num_words
        # temperature: ???
        logits /= temperature.unsqueeze(-1)
        # probs: batch_size, num_words
        probs = torch.softmax(logits,dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
