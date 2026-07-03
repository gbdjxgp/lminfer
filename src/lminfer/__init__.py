import torch

# from torch_npu.contrib import transfer_to_npu

if hasattr(torch, "npu") and torch.npu.is_available():
    torch.compile = lambda fn=None, *args, **kwargs: (
        fn if fn is not None else (lambda f: f)
    )

from lminfer.llm import LLM
from lminfer.sampling_params import SamplingParams
