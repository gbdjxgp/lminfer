from lminfer.layers import *
import torch
import torch.nn as nn
import torch.distributed as dist


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        scale=1.0,
        num_kv_heads=None,
        qkv_bias=False,
        base=10000,
        max_position=16384,
        block_size=256,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()
        
        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size
        
        self.total_num_kv_heads = num_kv_heads or num_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = scale
        
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=self.head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias = qkv_bias
        )
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias
        
        self.q_norm = RMSNorm(torch.ones(self.head_dim))
        self.k_norm = RMSNorm(torch.ones(self.head_dim))
        
        self.rotary_emb = RotaryEmbedding(
            base = base,
            rotary_embedding=head_dim,
            max_position=max_position
        )
        
        self.attention = Attention(
            self.num_heads,
            self.head_dim,
            scale,
            self.num_kv_heads,
            block_size
            )
        self.o_proj = RowParallelLinear(
            input_size=head_dim*self.total_num_heads,
            output_size=hidden_size,
            bias=False
            )
        
        
    def forward(self,x,positions):
        qkv = self.qkv_projection(x)
        
        q,k,v = qkv.split([self.q_size,self.kv_size,self.kv_size],dim=-1)
        
        if q.dim()==2:
            q = q.view(-1,self.num_heads,self.head_dim)
            k = k.view(-1,self.num_kv_heads,self.head_dim)
            v = v.view(-1,self.num_kv_heads,self.head_dim)
            
        else:
            B,N,_ = q.shape
            q = q.view(B,N,self.num_heads,self.head_dim)
            k = k.view(B,N,self.num_kv_heads,self.head_dim)
            v = v.view(B,N,self.num_kv_heads,self.head_dim)
            
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)
            
        q,k = self.rotary_emb(positions,q,k)
        o = self.attention(q,k,v)
        
        o = self.o_proj(o)
        
        return o
    
    
class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        bias=True
        ):
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size]*2,
            bias = bias
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias
        )
    def forward(self,x):
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x
    
class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        scale=1.0,
        num_kv_heads=None,
        qkv_bias = False,
        base=10000,
        max_position=16384,
        intermediate_size=4*1024,
        ffn_bias=True,
        block_size=256
    ):
        super().__init__()
        gamma = torch.ones(hidden_size)
        self.input_layernorm = RMSNorm(gamma)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size
        )
        self.post_attention_layernorm = RMSNorm(gamma)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias
        )
    def forward(self,x,residual):
        if residual is not None:
            x,residual = self.input_layernorm(x,residual)
        else:
            residual=x
            x = self.input_layernorm(x)
        from lminfer.utils import get_context
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            # prefill, batched
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens)-1):
                seq_len = cu_seqlens[i+1]-cu_seqlens[i]
                positions.extend(range(seq_len))
                positions = torch.tensor(positions,dtype=torch.long,device=x.device)
        elif context.is_prefill:
            # single sequence prefill
            positions = torch.arange(x.size(0),device=x.device)
        else:
            positions = context.context_lens-1
            
        x = self.self_attn(x,positions=positions)
        x,residual = self.post_attention_layernorm(x,residual)
        x=self.mlp(x)
        return x,residual
    
    
class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_heads,
        head_dim,
        scale,
        num_kv_heads=None,
        qkv_bias=False,
        base=10000,
        max_position=16384,
        intermediate_size=4*1024,
        ffn_bias=True,
        num_layers=12,
        block_size=256
    ):
        super().__init__()
        self.embed_tokens=VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.layers=nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size
            ) for _ in range(num_layers)
        ])
        gamma=torch.ones(hidden_size)
        self.norm=RMSNorm(gamma)
    def forward(self,input_ids):
        x = self.embed_tokens(input_ids)
        residual=None
        for layer in self.layers:
            x,residual = layer(x,residual)
            x,_=self.norm(x,residual)
            return x
        
        
class Qwen3ForCasuallLM(nn.Module):
    packed_module_mapping = {
        "q_proj":('q_proj','q'),
        "k_proj":('k_proj','k'),
        "v_proj":('v_proj','v'),
        "gate_up":('gate_up_proj','0'),
        "gate_down":('gate_down_proj','1')
    }
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_heads,
        head_dim,
        scale=1.0,
        num_kv_heads=None,
        qkv_bias=False,
        base=10000,
        max_position=16384,
        intermediate_size=4*1024,
        ffn_bias=True,
        num_layers=12,
        tie_word_embeddings=False,
        block_size=256
    ):
        super().__init__()
        head_dim = head_dim or hidden_size //num_heads
        self.model=Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size
        )
        self.lm_head=PrarllelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        if tie_word_embeddings:
            self.lm_head.weight=self.model.embed_tokens.weight
            
    def forward(self,input_ids):
        x = self.model(input_ids)
        return x
    def compute_logits(self,hidden_states):
        logits = self.lm_head(hidden_states)
        return logits
