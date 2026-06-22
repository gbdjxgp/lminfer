import torch.nn as nn
import torch


def apply_rotart_pos_emb(x:torch.Tensor,cos:torch.Tensor ,sin:torch.Tensor):
    if x.dim() ==3:
        # x: (total_tokens, num_heads,head_dim)
        # cos,sin: (total_tokens, head_dim//2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        # x: (Batch_size, seq_len, num_heads,head_dim)
        # cos,sin: (total_tokens,head_dim//2)->(1,seq_len,1,head)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
    x1,x2 = x.chunk(2,dim=-1)
    
    out1 = x1*cos -x2*sin
    out2 = x1*sin +x2*cos
    return torch.cat([out1,out2],dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self,
                 base:int,
                 rotary_embedding:int,
                 max_position:int=2048
                 ):
        super().__init__()
        self.base = base
        self.max_position = max_position
        self.inv_freq = 1/(base**(torch.arange(0,rotary_embedding,2)/rotary_embedding))
        positions = torch.arange(self.max_position).float()
        # positions: (max_position)
        # inv_freq: (rotary_embedding)
        # freqs: (max_position,rotary_embedding)
        freqs = torch.einsum("i,j->ij",positions,self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # cos_sin_cache: (max_position,2*rotary_embedding)
        cos_sin_cache = torch.cat([cos,sin],dim=-1)
        self.register_buffer("cos_sin_cache",cos_sin_cache)
        
    def forward(self,positions,query,key):
        # query/key: (total_tokens, num_heads,head_dim)
        cos_sin_cache = self.cos_sin_cache[positions]
        cos,sin = cos_sin_cache.chunk(2,dim=-1)
        
        return (
            apply_rotart_pos_emb(query,cos,sin),
            apply_rotart_pos_emb(key,cos,sin)
        )
        