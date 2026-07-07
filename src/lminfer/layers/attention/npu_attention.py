"""
NPU attention backend — pure torch_npu implementation, no Triton dependency.

All attention compute goes through torch_npu native operators:
- KV cache store:   torch_npu.npu_scatter_pa_kv_cache
- Prefill (no paging): torch_npu.npu_fused_infer_attention_score  (TND, block_table=None)
- Prefill (paged):     torch_npu.npu_fused_infer_attention_score  (TND, block_table!=None)
- Decode:              torch_npu.npu_fused_infer_attention_score  (TND, block_table!=None)
"""

import torch
import torch.nn as nn
import torch_npu
from lminfer.utils import get_context
from .attention import AttentionBackend

# ---------------------------------------------------------------------------
# Cached tensors — reused across calls to avoid repeated allocation
# ---------------------------------------------------------------------------

_ATTN_MASK_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_ACTUAL_Q_LENS_CPU_CACHE: dict[int, torch.Tensor] = {}


def _get_splitfuse_attn_mask(device: torch.device, max_seq_len: int) -> torch.Tensor:
    """Causal attention mask (upper-triangular) used by fused_infer_attention_score."""
    key = (str(device), max_seq_len)
    mask = _ATTN_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.int8, device=device),
            diagonal=1,
        ).contiguous()
        _ATTN_MASK_CACHE[key] = mask
    return mask


def _get_actual_seq_lengths_q_cpu(batch_size: int) -> torch.Tensor:
    """Cumulative Q token counts [1, 2, ..., batch_size] for decode (seq_len=1 per batch)."""
    tensor = _ACTUAL_Q_LENS_CPU_CACHE.get(batch_size)
    if tensor is None:
        tensor = torch.arange(
            1, batch_size + 1, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        _ACTUAL_Q_LENS_CPU_CACHE[batch_size] = tensor
    return tensor


# ---------------------------------------------------------------------------
# KV cache store
# ---------------------------------------------------------------------------

def store_kvcache_kernel(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter projected K/V into the paged KV cache (NPU-native).

    Matches torch_npu.npu_scatter_pa_kv_cache scene 2:
        key/value:   (total_tokens,  num_kv_heads, head_dim)
        key/value_cache: (num_blocks, block_size,   num_kv_heads, head_dim)
        slot_mapping: (total_tokens,)  —  flat indices in [0, num_blocks * block_size)
    """
    # npu_scatter_pa_kv_cache requires contiguous inputs.
    # key/value from QKV projection are already contiguous; call .contiguous()
    # just to be safe (it's a no-op when already contiguous).
    torch_npu.npu_scatter_pa_kv_cache(
        key.contiguous(),
        value.contiguous(),
        k_cache,          # contiguous slice from the global kv_cache pool
        v_cache,
        slot_mapping,
        cache_mode="Norm",  # ND layout — no NZ conversion needed
    )


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    store_kvcache_kernel(key, value, k_cache, v_cache, slot_mapping)


# ---------------------------------------------------------------------------
# Prefill attention (no prefix cache — contiguous K/V)
# ---------------------------------------------------------------------------

def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int | None,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """NPU prefill attention with a triton-compatible public interface.

    The NPU backend uses torch_npu fused attention under the hood and accepts
    the same argument list as the Triton backend for call-site compatibility.
    """
    del max_seqlen_q, max_seqlen_k
    if not causal:
        raise NotImplementedError("NPU attention backend only supports causal attention")

    attn_mask = _get_splitfuse_attn_mask(q.device, 2048)
    num_key_value_heads = k.shape[1]
    if block_table is not None:
        # torch_npu.npu_fused_infer_attention_score with block_table expects
        # K/V reshaped as (num_blocks, block_size, num_kv_heads * head_dim).
        num_key_value_heads = k.shape[2]
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)

    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q,
        key=k,
        value=v,
        atten_mask=attn_mask,
        block_table=block_table,
        input_layout="TND",
        block_size=(k.shape[1] if block_table is not None else 128),
        actual_seq_lengths=cu_seqlens_q[1:].tolist(),
        actual_seq_lengths_kv=cu_seqlens_k[1:].tolist(),
        num_key_value_heads=num_key_value_heads,
        num_heads=q.shape[1],
        scale=softmax_scale,
        sparse_mode=3,
    )
    return output.view(q.shape[0], q.shape[1], q.shape[2])


# ---------------------------------------------------------------------------
# Decode attention (paged KV cache, one token per sequence)
# ---------------------------------------------------------------------------

def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor | list[int],
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """NPU decode attention with a triton-compatible public interface.

    Accepts the Triton decode signature, including `q` in
    `(num_seqs, seq_len, num_heads, head_dim)` layout.
    """
    if not causal:
        raise NotImplementedError("NPU attention backend only supports causal attention")

    if q.dim() == 4:
        if q.shape[1] != 1:
            raise NotImplementedError("NPU decode path only supports seq_len == 1")
        q = q.squeeze(1)

    batch_size = q.shape[0]
    actual_seq_lengths_q = _get_actual_seq_lengths_q_cpu(batch_size)
    actual_seq_lengths_kv = (
        cache_seqlens
        if isinstance(cache_seqlens, torch.Tensor)
        else torch.tensor(
            cache_seqlens, dtype=torch.int32, pin_memory=True, device="cpu"
        )
    )
    attn_mask = _get_splitfuse_attn_mask(q.device, 2048)

    k_flat = k_cache.view(k_cache.shape[0], k_cache.shape[1], -1)
    v_flat = v_cache.view(v_cache.shape[0], v_cache.shape[1], -1)

    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q,
        key=k_flat,
        value=v_flat,
        atten_mask=attn_mask,
        block_table=block_table,
        input_layout="TND",
        block_size=k_cache.shape[1],
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=k_cache.shape[2],
        num_heads=q.shape[1],
        scale=softmax_scale,
        sparse_mode=3,
    )
    return output.view(batch_size, q.shape[1], q.shape[2])


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class NPUAttentionBackend(AttentionBackend):
    """NPU attention backend implementation."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q/k/v: (total_tokens, num_heads|num_kv_heads, head_dim)"""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 1. Store projected K/V into the paged cache (when cache is allocated).
        if k_cache.numel() and v_cache.numel():
            store_kvcache_kernel(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. Attention compute — dispatch on prefill vs decode.
        if context.is_prefill:
            if context.block_tables is None:
                # No prefix cache: K/V are contiguous tensors from QKV projection.
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                )
            else:
                # Prefix cache: K/V were scattered into the paged cache above;
                # use the cache directly with block_table for attention.
                o = flash_attn_varlen_func(
                    q,
                    k_cache,
                    v_cache,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
        else:
            # Decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o
