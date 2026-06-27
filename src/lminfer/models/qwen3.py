from lminfer.layers import *
import torch
import torch.nn as nn
import torch.distributed as dist
from lminfer.utils import deviceinfo, get_context, set_context


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias=False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = deviceinfo.tp_size
        # total_num_heads
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        # num_kv_heads
        self.total_num_kv_heads = num_kv_heads or num_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        # head_dim
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        self.qkv_proj = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=self.head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)
        self.qkv_bias = qkv_bias
        if not self.qkv_bias:
            self.q_norm = RMSNorm(torch.ones(self.head_dim))
            self.k_norm = RMSNorm(torch.ones(self.head_dim))

        self.rotary_emb = get_rope(
            self.num_heads,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

        self.attention = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * head_dim,
            output_size=hidden_size,
            bias=False,
        )

    def forward(self, x, positions):
        q, k, v = self.qkv_proj(x).split(
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attention(q, k, v)
        o = self.o_proj(o)
        return o


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
        )

    def forward(self, x):
        x = self.down_proj(self.act_fn(self.gate_up_proj(x)))
        return x


class Qwen3DecoderLayer(nn.Module):
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
        intermediate_size=4 * 1024,
        ffn_bias=True,
        block_size=256,
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
            block_size=block_size,
        )
        self.post_attention_layernorm = RMSNorm(gamma)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size, bias=ffn_bias
        )

    def forward(self, x, residual):
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            residual = x
            x = self.input_layernorm(x)
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            # prefill, batched
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
                positions.extend(range(seq_len))
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            # B,S,D/S,D
            positions = torch.arange(x.size(-2), device=x.device)
        else:
            # decode,直接传每一个seq的最后一个位置即可
            positions = context.context_lens - 1

        x = self.self_attn(x, positions=positions)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


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
        intermediate_size=4 * 1024,
        ffn_bias=True,
        num_layers=12,
        block_size=256,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        self.layers = nn.ModuleList(
            [
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
                    block_size=block_size,
                )
                for _ in range(num_layers)
            ]
        )
        gamma = torch.ones(hidden_size)
        self.norm = RMSNorm(gamma)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
            x, _ = self.norm(x, residual)
        return x


class Qwen3ForCasuallLM(nn.Module):
    packed_module_mapping = {
        "q_proj": ("q_proj", "q"),
        "k_proj": ("k_proj", "k"),
        "v_proj": ("v_proj", "v"),
        "gate_up": ("gate_up_proj", "0"),
        "gate_down": ("gate_down_proj", "1"),
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
        intermediate_size=4 * 1024,
        ffn_bias=True,
        num_layers=12,
        tie_word_embeddings=False,
        block_size=256,
    ):
        super().__init__()
        head_dim = head_dim or hidden_size // num_heads
        self.model = Qwen3Model(
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
            block_size=block_size,
        )
        self.lm_head = PrarllelLMHead(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        x = self.model(input_ids)
        return x

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        return logits


if __name__ == "__main__":
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    device = deviceinfo.device()
    print(f"Detected platform={deviceinfo.platform()}, " f"device={device}")
    model = Qwen3ForCasuallLM(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_layers=2,
    ).to(device)
    model.eval()
    with torch.inference_mode():
        # TODO:support cu_seqlen_k,max_seqlen_q
        set_context(True, cu_seqlens_q=torch.tensor([0, 16], device=device))
        # B,S格式输入
        input_ids = torch.randint(0, 50257, (16,), device=device)
        print(input_ids.shape)
        output = model(input_ids)
    print(output.shape)
