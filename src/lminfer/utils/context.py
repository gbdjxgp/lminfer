from dataclasses import dataclass
import torch

@dataclass
class Context:
    is_prefill=False
    cu_seqlens_q=None
    cu_seqlens_k=None
    max_seqlen_q=0
    max_seqlen_k=0
    slot_mapping=None
    context_lens=None
    block_tables=None
    
_context=Context()

def get_context():
    return _context

def reset_context():
    global _context
    _context=Context()
    
def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None
):
    global _context
    _context=Context(is_prefill,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,slot_mapping,context_lens,block_tables)