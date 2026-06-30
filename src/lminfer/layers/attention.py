import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
import triton
from lminfer.utils import get_context, deviceinfo


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # key/value: (total_tokens, num_kv_heads, head_dim)
    # key_stride/value_stride: key/value.stride(0)
    # D: (num_kv_heads * head_dim)
    # slot_mapping: (num_tokens)
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if slot_idx == -1:
        return

    key_offsets = token_idx * key_stride + tl.arange(0, D)
    value_offset = token_idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offset)

    # cache_offsets: kvcache池子的offset, (*, D)=(*, num_kv_heads * head_dim)
    cache_offsets = slot_idx * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    TOTAL_TOKENS, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    # 要求key-value在-1维度连续
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == TOTAL_TOKENS
    grid = (TOTAL_TOKENS,)
    store_kvcache_kernel[grid](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


@triton.jit
def flash_attn_varlen_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    softmax_scale,
    causal,
    block_table,  # (num_seqs, max_pages_per_seq), None if not paged
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,  # page size when block_table is used
    use_block_table: tl.constexpr,  # True if paged KV-cache
    max_pages_per_seq: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_offset_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)

    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # Q sequence range
    q_seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    q_seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    q_seq_len = q_seq_end - q_seq_start

    if q_offset_m * BLOCK_M >= q_seq_len:
        return

    # K sequence range — may be longer than Q (includes cached prefix)
    k_seq_start = tl.load(cu_seqlens_k_ptr + seq_idx)
    k_seq_end = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    k_seq_len = k_seq_end - k_seq_start

    offs_m = q_offset_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < q_seq_len
    offs_d = tl.arange(0, head_dim)

    # Load Q block — layout: (total_q_tokens, num_heads, head_dim)
    q_ptrs = (
        Q
        + (q_seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # For causal masking:
    # - offs_m are *positions within Q chunk* (0-based inside this seq's Q)
    # - but in the full KV sequence, these Q tokens sit at the END
    # - so the absolute KV position of offs_m[i] = k_seq_len - q_seq_len + offs_m[i]
    # This offset shifts Q indices into the KV coordinate space.
    q_offset_in_k = k_seq_len - q_seq_len  # >=0: how many cached prefix tokens

    num_blocks = tl.cdiv(k_seq_len, BLOCK_N)

    for block_n in range(num_blocks):
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)  # KV positions (absolute)
        mask_n = offs_n < k_seq_len

        if use_block_table:
            # -------------------------------------------------------
            # Paged KV-cache path
            # block_table row for this sequence: (max_pages_per_seq,)
            # Physical page index for position offs_n[j]:
            #   page_idx  = offs_n[j] // page_size
            #   page_off  = offs_n[j] %  page_size
            # Physical token index = page_ids[page_idx] * page_size + page_off
            # K/V layout: (total_pages, page_size, num_kv_heads, head_dim)
            # -------------------------------------------------------
            page_ids = offs_n // page_size  # (BLOCK_N,)
            page_offs = offs_n % page_size  # (BLOCK_N,)

            # Gather physical page numbers from block_table
            # block_table ptr for this seq: block_table + seq_idx * max_pages_per_seq
            bt_ptrs = block_table + seq_idx * max_pages_per_seq + page_ids
            # mask invalid pages (offs_n >= k_seq_len already handled by mask_n,
            # but page_ids could be out of range, so guard with mask_n)
            physical_pages = tl.load(bt_ptrs, mask=mask_n, other=0)

            # Physical flat token index in the paged tensor
            phys_token_idx = physical_pages * page_size + page_offs  # (BLOCK_N,)

            # K: load (head_dim, BLOCK_N) — transpose for dot product
            k_ptrs = (
                K
                + phys_token_idx[None, :] * num_kv_heads * head_dim
                + kv_head_idx * head_dim
                + offs_d[:, None]
            )
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

            # V: load (BLOCK_N, head_dim)
            v_ptrs = (
                V
                + phys_token_idx[:, None] * num_kv_heads * head_dim
                + kv_head_idx * head_dim
                + offs_d[None, :]
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        else:
            # -------------------------------------------------------
            # Contiguous KV path (no paging)
            # K/V layout: (total_kv_tokens, num_kv_heads, head_dim)
            # -------------------------------------------------------
            k_ptrs = (
                K
                + (k_seq_start + offs_n[None, :]) * num_kv_heads * head_dim
                + kv_head_idx * head_dim
                + offs_d[:, None]
            )
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

            v_ptrs = (
                V
                + (k_seq_start + offs_n[:, None]) * num_kv_heads * head_dim
                + kv_head_idx * head_dim
                + offs_d[None, :]
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # QK^T
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N)
        qk = qk * softmax_scale

        # Causal mask:
        # Q token i attends to KV token j iff (q_offset_in_k + offs_m[i]) >= offs_n[j]
        if causal:
            causal_mask = (q_offset_in_k + offs_m[:, None]) >= offs_n[None, :]
            qk = tl.where(causal_mask & mask_n[None, :], qk, -1e10)
        else:
            qk = tl.where(mask_n[None, :], qk, -1e10)

        # Online softmax update
        m_temp = tl.max(qk, axis=1)  # (BLOCK_M,)
        m_i_new = tl.maximum(m_i, m_temp)
        alpha = tl.exp(m_i - m_i_new)  # rescale factor for previous acc
        p = tl.exp(qk - m_i_new[:, None])  # (BLOCK_M, BLOCK_N)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output — layout mirrors Q: (total_q_tokens, num_heads, head_dim)
    o_ptrs = (
        O
        + (q_seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


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
    # q: (total_tokens, num_heads, head_dim)
    # k/v: (total_tokens, num_kv_heads, head_dim) if block_table not exist
    # k/v: (total_pages, page_size, num_heads, head_dim) if block_table exist
    # max_seqlen_q
    # cu_seqlens_q: 每个seq的q的其实与结束index
    # max_seqlen_k: 具体与算子下放有关系,可以不管
    # cu_seqlens_k: chunked prefill需要用到，且传入block_table的情况下需要用到
    # softmax_scale: 默认应该为1/sert(head_dim)
    # causal: 是否为因果mask，否的话为双向注意力
    # block_table: (num_seqs, max_pages_per_seq)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    # assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous()

    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[-2]
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    if head_dim <= 64:
        BLOCK_M, BLOCK_N = 64, 64
    elif head_dim <= 128:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 16

    use_block_table = block_table is not None
    num_seqs = cu_seqlens_q.shape[0] - 1

    if use_block_table:
        # k/v: (total_pages, page_size, num_kv_heads, head_dim)
        page_size = k.shape[1]
        max_pages_per_seq = block_table.shape[1]
    else:
        page_size = 1  # unused, but must be constexpr > 0
        max_pages_per_seq = 1  # unused

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), num_heads, num_seqs)

    flash_attn_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        causal,
        (
            block_table
            if use_block_table
            else torch.empty(0, dtype=torch.int32, device=q.device)
        ),
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        use_block_table,
        max_pages_per_seq,
        BLOCK_M,
        BLOCK_N,
    )
    return output


@triton.jit
def flash_attn_with_kvcache_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # q:        (num_seqs, seq_len, num_heads, head_dim)
    # k_cache:  (num_blocks, block_size, num_kv_heads, head_dim)
    # v_cache:  (num_blocks, block_size, num_kv_heads, head_dim)
    # block_tables: (num_seqs, max_num_blocks)
    # context_lens: (num_seqs,)
    # grid: (num_seqs, seq_len, num_heads)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)
    offs_d = tl.arange(0, head_dim)

    if context_len == 0:
        # (num_seqs, seq_len, num_heads, head_dim)
        output_offset = (
            batch_idx * seq_len * num_heads * head_dim
            + seq_idx * num_heads * head_dim
            + head_idx * head_dim
            + offs_d
        )
        tl.store(output_ptr + output_offset, tl.zeros([head_dim], dtype=tl.float32))
        return

    # q: (num_seqs, seq_len, num_heads, head_dim)
    q_offset = (
        batch_idx * seq_len * num_heads * head_dim
        + seq_idx * num_heads * head_dim
        + head_idx * head_dim
        + offs_d
    )
    q = tl.load(query_ptr + q_offset)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    # 某个序列的所有tokens，按照BLOCK_N切分
    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            # 有效的kvcache
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            # 内层循环：对qk进行切块计算
            # TODO: 这里的循环是否能优化掉?
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    # TODO: 这个判断是否必要?
                    if block_num < max_num_blocks:
                        block_tables_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(
                            block_tables_ptr + block_tables_offset
                        )
                        # TODO: 这个判断是否必要?
                        if physical_block_idx != -1:
                            # k_cache_ptr: (num_blocks, block_size, num_kv_heads, head_dim)
                            # k_offset shape: (head_dim)
                            k_offset = (
                                physical_block_idx
                                * block_size
                                * num_kv_heads
                                * head_dim
                                + block_offset * num_kv_heads * head_dim
                                + kv_head_idx * head_dim
                                + offs_d
                            )
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            # q shape: (head_dim)
                            # 计算qk^T
                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)
            qk = tl.where(mask_n, qk, -1e10)
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            acc = acc * alpha
            l_i = l_i * alpha

            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    if block_num < max_num_blocks:
                        block_tables_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(
                            block_tables_ptr + block_tables_offset
                        )
                        if physical_block_idx != -1:
                            v_offset = (
                                physical_block_idx
                                * block_size
                                * num_kv_heads
                                * head_dim
                                + block_offset * num_kv_heads * head_dim
                                + kv_head_idx * head_dim
                                + offs_d
                            )
                            v_vec = tl.load(v_cache_ptr + v_offset)
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            m_i = m_i_new
    # 最终结果
    output = acc / l_i
    # 输出
    # (batch_size,num_heads,head_dim)
    output_offset = (
        batch_idx * seq_len * num_heads * head_dim
        + seq_idx * num_heads * head_dim
        + head_idx * head_dim
        + offs_d
    )
    tl.store(output_ptr + output_offset, output)


def flash_attn_with_kvcache(
    q: torch.Tensor,  # (num_seqs, seq_len, num_heads, head_dim)
    k_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    cache_seqlens: torch.Tensor,  # (num_seqs,)
    block_table: torch.Tensor,  # (num_seqs, max_num_blocks)
    softmax_scale: float,
    causal: bool,
):
    assert causal

    num_seqs, seq_len, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    max_num_blocks = block_table.shape[1]
    assert q.is_contiguous
    # output mirrors q layout
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    BLOCK_N = 64 if head_dim <= 128 else 32

    # grid: one program per (sequence, query-token, head)
    grid = (num_seqs, seq_len, num_heads)

    flash_attn_with_kvcache_kernel[grid](
        output,
        q,
        k_cache,
        v_cache,
        block_table,
        cache_seqlens,
        scale=softmax_scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        seq_len=seq_len,
        BLOCK_N=BLOCK_N,
    )

    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q:(total_tokens, num_heads, head_dim)
        # kv: (total_tokens, num_kv_heads, head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # TODO: 实现prefix cache,这里的注意力计算也应该是有问题的
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
                block_table=context.block_tables,
            )
        else:
            # decode
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


if __name__ == "__main__":
    pass
