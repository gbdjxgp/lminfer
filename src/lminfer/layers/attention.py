import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
import triton


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if slot_idx == -1:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    head_idx = tl.program_id(1)

    head_offsets = tl.arange(0, head_dim)

    # input: (num_tokens,num_kv_heads,head_dim)
    input_offset = (
        token_idx * num_kv_heads * head_dim + head_idx * head_dim + head_offsets
    )
    # cache: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (
        block_idx * block_size * num_kv_heads * head_dim
        + block_offset * num_kv_heads * head_dim
        + head_idx * head_dim
        + head_offsets
    )

    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)

    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    num_tokens, num_kv_heads, head_dim = key.shape
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    assert k_cache.shape == v_cache.shape
    assert slot_mapping.numel() == num_tokens
    grid = (num_tokens, num_kv_heads)
    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
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
        l_i += l_i * alpha + tl.sum(p, axis=1)
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
    output = torch.empty_like(q)

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
    max_seqlen = cu_seqlens.max().item()

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
    
    offs_d = tl.arange(0,head_dim)
    
    # query: (batch_size,num_heads,head_dim)
    q_offset = batch_idx*num_heads*head_dim+head_idx*head_dim+offs_d
    q = tl.load(query_ptr+q_offset)
    
    acc = tl.zeros([head_dim],dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    max_chunks = tl.cdiv(max_num_blocks*block_size,BLOCK_N)
    pass


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
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    query = query.contiguous()
    output = torch.empty_like(query)
    BLOCK_N = 64 if head_dim <= 128 else 32
    
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
        BLOCK_N=BLOCK_N
    )
    
    return output
    


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self, head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        pass


if __name__ == "__main__":
    batch_size = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    seq_len = 10

    q = torch.randn(batch_size, seq_len, num_heads * head_dim).npu()
    k = torch.randn(batch_size, seq_len, num_kv_heads * head_dim).npu()
    v = torch.randn(batch_size, seq_len, num_kv_heads * head_dim).npu()

    cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.int32).npu()

    output = flash_attention_prefill(
        q,
        k,
        v,
        cu_seqlens,
        scale=1.0,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    print(output.shape)
