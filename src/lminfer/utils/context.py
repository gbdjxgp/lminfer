from dataclasses import dataclass
import torch

@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: int | None = None
    block_tables: torch.Tensor | None = None
    
_context=Context()

def get_context():
    return _context

def reset_context():
    global _context
    _context=Context()
    
def set_context(
    is_prefill,
    # prefill用
    cu_seqlens_q=None,
    # TODO: To be supported
    cu_seqlens_k=None,
    # TODO: to be supported
    max_seqlen_q=0,
    # TODO: to be supported
    max_seqlen_k=0,
    # kvcache用
    slot_mapping=None,
    # context_lens,decode时候的增量位置，由于decode的时候长度为num_seqs,并非seq_len,因此需要标记每个seq的长度
    context_lens=None,
    # decode用
    block_tables=None
):
    global _context
    _context=Context(is_prefill,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,slot_mapping,context_lens,block_tables)