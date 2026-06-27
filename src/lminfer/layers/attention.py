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
def flash_attention_varlen_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # start_m: 当前block在序列维度上的起始位置,实际处理的序列为起始位置=start_m*BLOCK_M,长度为BLOCK_M
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # GQA中每个Q头对应的KV头的索引，off_h是Q头的索引
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # 读取每个索引的序列起始与结束位置
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    # 起始位置超出序列长度则直接返回(这里可以取得等号)
    if start_m * BLOCK_M >= seq_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    offs_d = tl.arange(0, head_dim)
    # Q格式：S,N,D(total_tokens,num_heads,head_dim)
    # 注意这里地址表示可能会超出范围，因此后面在load之前要mask
    # offs_m:(BLOCK_M,1),offs_d:(1,head_dim)，因此读取出来的q_ptrs就是二维的(BLOCK_M,head_dim)
    q_ptrs = (
        Q
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    # 使用(BLOCK_M,1)的mask进行读取操作，读取出来(BLOCK_M,head_dim)的一块Q数据
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    ####################################

    # 初始化累加器
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    for block_n in range(num_blocks):
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        # 读取k,k_ptrs:head_dim,BLOCK_N
        k_ptrs = (
            K
            + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + offs_d[:, None]
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, k)
        qk = qk * scale

        # mask形状应该为(BLOCK_M,BLOCK_N)
        mask_causal = (offs_m[:, None]) >= (offs_n[None, :])
        # 语法：tl.where(valid_mask,a,b)，valid_mask为True的部分变为a否则变为b
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        ####################################
        # online softmax,行维度，因此是m行每一行的n个数字一起做softmax，当然求softmax要遍历两次，但是我们只要PV的结果，因此只需要遍历一次
        ####################################

        # 找到n个数字中的最大值m_temp:(m)
        m_temp = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_temp)
        alpha = tl.exp(m_i - m_i_new)
        # qk:(BLOCK_M,BLOCK_N),m_i_new:(BLOCK_M,1)->p:(BLOCK_M,BLOCK_N)
        p = tl.exp(qk - m_i_new[:, None])
        acc = acc * alpha[:, None]
        # 当前块的p可以直接与V相乘，结果直接存放到acc中
        # v_ptrs:(BLOCK_N,head_dim)
        v_ptrs = (
            V
            + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        # 这里p与v的精度应该是相同的，不知道为什么需要to一下？
        acc += tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    o_ptrs = (
        O
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    # 由于O是指针，因此需要O.dtype.element_ty获取元素类型,element_ty = element type
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim
):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # 序列数量
    num_seqs = cu_seqlens.shape[0] - 1
    # 求所有batch中最长的序列长度
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    grid = (triton.cdiv(max_seqlen, BLOCK_M), num_heads, num_seqs)
    flash_attention_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        BLOCK_M,
        BLOCK_N,
    )
    return output


@triton.jit
def paged_attention_decode_kernel(
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
    BLOCK_N: tl.constexpr,
):
    # k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    # v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    # block_tables: (batch_size, max_num_blocks)
    # context_lens: (batch_size,)
    # grid: (batch_size, num_heads)
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)
    offs_d = tl.arange(0, head_dim)

    if context_len == 0:
        output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
        tl.store(output_ptr + output_offset, tl.zeros([head_dim], dtype=tl.float32))
        return

    # query: (batch_size,num_heads,head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    # 外层循环: max_chunks,某个序列的所有token,按照BLOCK_N进行切分
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
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
):
    # query: (batch_size,num_heads,head_dim)
    # k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    # v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    # block_tables: (batch_size, max_num_blocks)
    # context_lens: (batch_size,)
    # batch_size: 正在同时进行decode的请求数，num_blocks: 全局物理kvcache池子
    # max_num_tokens: 最长的序列使用的block长度
    # context_lens: 记录每个序列的实际长度
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    query = query.contiguous()
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    BLOCK_N = 64 if head_dim <= 128 else 32
    # grid dim0: 同时进行decode的序列的数量，dim1: 头数量
    grid = (batch_size, num_heads)

    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
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
            o = flash_attention_prefill(
                q,
                k,
                v,
                context.cu_seqlens_q,
                self.scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            # o: (num_tokens,num_heads,head_dim)->(num_tokens,num_heads*head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # decode
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                self.scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )
            return o


if __name__ == "__main__":
    NUM_HEADS = 4
    NUM_KV_HEADS = 2  # GQA: each KV head is shared by (4/2)=2 Q heads
    HEAD_DIM = 16
    DTYPE = torch.float16
    DEVICE = deviceinfo.device()
    # Two sequences: seq1 has tokens [0..4] (len=5), seq2 has tokens [5..9] (len=5)
    SEQ_LENS = [5, 5]  # per-sequence lengths
    TOTAL_TOKENS = sum(SEQ_LENS)  # 10  (packed / varlen layout)
    NUM_SEQS = len(SEQ_LENS)

    # Packed tensors: (total_tokens, num_heads, head_dim)
    q_prefill = torch.randn(
        TOTAL_TOKENS, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    k_prefill = torch.randn(
        TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    v_prefill = torch.randn(
        TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )

    def make_cu_seqlens(seq_lens):
        """Cumulative sum, starting from 0.  Shape: (batch+1,)"""
        cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEVICE)
        for i, l in enumerate(seq_lens):
            cu[i + 1] = cu[i] + l
        return cu

    cu_seqlens = make_cu_seqlens(SEQ_LENS)  # [0, 5, 10]
    # TODO: TO BE SUPPORTED
    max_seqlen = max(SEQ_LENS)  # 5

    out_prefill = flash_attention_prefill(
        q_prefill,
        k_prefill,
        v_prefill,
        cu_seqlens,
        scale=1.0,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )

    # out_prefill = flash_attn_varlen_func(
    #     q_prefill,
    #     k_prefill,
    #     v_prefill,
    #     cu_seqlens_q  = cu_seqlens,
    #     cu_seqlens_k  = cu_seqlens,
    #     max_seqlen_q  = max_seqlen,
    #     max_seqlen_k  = max_seqlen,
    #     dropout_p     = 0.0,
    #     causal        = True,   # each token attends only to itself and earlier tokens
    # )

    print(f"q shape  : {q_prefill.shape}")  # (10, 4, 16)
    print(f"k shape  : {k_prefill.shape}")  # (10, 2, 16)
    print(f"cu_seqlens: {cu_seqlens.tolist()}")
    print(f"out shape: {out_prefill.shape}")  # (10, 4, 16)
    print()

    # ══════════════════════════════════════════════════════════════════════════════
    # PAGED DECODE CONFIG
    # ══════════════════════════════════════════════════════════════════════════════
    PAGE_SIZE = 2  # tokens per page (block)
    MAX_PAGES = 16  # total pages in the pool
    # Each sequence needs ceil(seq_len / page_size) pages
    # seq1: ceil(5/2)=3 pages, seq2: ceil(5/2)=3 pages

    # ─── Paged KV Cache Pool ──────────────────────────────────────────────────────
    # Shape: (max_pages, page_size, num_kv_heads, head_dim)
    k_cache = torch.zeros(
        MAX_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.zeros(
        MAX_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )

    # ─── Fill cache with "prefill" KV data ────────────────────────────────────────
    # seq1 tokens [0..4] → pages 0,1,2   (page2 has 1 valid token, 1 padding)
    # seq2 tokens [5..9] → pages 3,4,5   (page5 has 1 valid token, 1 padding)
    prefill_k = torch.randn(10, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    prefill_v = torch.randn(10, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    def fill_paged_cache(
        k_src, v_src, seq_lens, page_size, k_cache, v_cache, page_table
    ):
        """Write packed KV tensors into paged cache according to page_table."""
        token_offset = 0
        for seq_id, seq_len in enumerate(seq_lens):
            for tok_idx in range(seq_len):
                page_idx = tok_idx // page_size  # which page within sequence
                slot_idx = tok_idx % page_size  # slot within page
                phys_page = page_table[seq_id][page_idx]  # physical page in pool
                k_cache[phys_page, slot_idx] = k_src[token_offset + tok_idx]
                v_cache[phys_page, slot_idx] = v_src[token_offset + tok_idx]
            token_offset += seq_len

    # Page table: (num_seqs, max_pages_per_seq)
    # Maps (sequence, logical_page) → physical page index in the pool
    max_pages_per_seq = max((l + PAGE_SIZE - 1) // PAGE_SIZE for l in SEQ_LENS)
    # page table, -1==padding
    page_table = torch.tensor(
        [
            [0, 1, 2, -1],  # seq1 uses physical pages 0,1,2
            [3, 4, 5, -1],  # seq2 uses physical pages 3,4,5
        ],
        dtype=torch.int32,
        device=DEVICE,
    )  # shape: (num_seqs, max_pages_per_seq)

    fill_paged_cache(
        prefill_k, prefill_v, SEQ_LENS, PAGE_SIZE, k_cache, v_cache, page_table
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # PAGED DECODE
    # ══════════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PAGED DECODE")
    print("=" * 60)

    # Q: one new token per sequence — shape (num_seqs, num_heads, head_dim)
    q_decode = torch.randn(NUM_SEQS, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # How many KV tokens are valid per sequence (prefill length before this decode step)
    cache_seqlens = torch.tensor(SEQ_LENS, dtype=torch.int32, device=DEVICE)  # [5, 5]

    # out_decode = flash_attn_with_kvcache(
    #     q              = q_decode,        # (num_seqs, num_heads, head_dim)
    #     k_cache        = k_cache,         # (max_pages, page_size, num_kv_heads, head_dim)
    #     v_cache        = v_cache,         # (max_pages, page_size, num_kv_heads, head_dim)
    #     cache_seqlens  = cache_seqlens,   # valid KV length per sequence
    #     block_table    = page_table,      # (num_seqs, max_pages_per_seq)
    #     causal         = False,
    #     softmax_scale  = HEAD_DIM ** -0.5,
    # )

    out_decode = paged_attention_decode(
        query=q_decode,  # (num_seqs, num_heads, head_dim)
        k_cache=k_cache,  # (max_pages, page_size, num_kv_heads, head_dim)
        v_cache=v_cache,  # (max_pages, page_size, num_kv_heads, head_dim)
        block_tables=page_table,  # (num_seqs, max_pages_per_seq)
        context_lens=cache_seqlens,  # valid KV length per sequence
        scale=HEAD_DIM**-0.5,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=PAGE_SIZE,
    )
    print(f"q_decode shape   : {q_decode.shape}")  # (2, 4, 16)
    print(f"k_cache shape    : {k_cache.shape}")  # (16, 2, 2, 16)
    print(f"page_table       :\n{page_table}")
    print(f"cache_seqlens    : {cache_seqlens.tolist()}")
    print(f"out_decode shape : {out_decode.shape}")  # (2, 4, 16)

    assert out_decode.shape == (NUM_SEQS, NUM_HEADS, HEAD_DIM)
    print("\nAll assertions passed ✓")
