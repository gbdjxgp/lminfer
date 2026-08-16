from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class CUDAGraphMode(enum.Enum):
    """
    非元组的时候表示
    None: 不启用图模式
    PIECEWISE: 混批模式
    FULL: CUDA Graph 模式

    元组的时候表示
    FULL_DECODE_ONLY: TODO: Decode用FULL，Prefill用None?
    FULL_AND_PIECEWISE: TODO: Decode用FULL，Prefill用PIECEWISE?
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def separate_routine(self) -> bool:
        """纯 decode 和混批是否走不同的图模式。"""
        return isinstance(self.value, tuple)

    def decode_mode(self) -> CUDAGraphMode:
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> CUDAGraphMode:
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    def max_cudagraph_mode(self) -> CUDAGraphMode:
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL


@dataclass
class CompilationConfig:
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_DECODE_ONLY
    enforce_eager: bool = False
    cudagraph_capture_sizes: list[int] = field(default_factory=list)
    # TODO: 搞清楚这个是干什么
    static_forward_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.enforce_eager:
            self.cudagraph_mode = CUDAGraphMode.NONE

    @property
    def max_capture_size(self) -> int:
        return self.cudagraph_capture_sizes[-1] if self.cudagraph_capture_sizes else 0

    def init_with_sizes(self, max_num_seqs: int) -> None:
        if self.cudagraph_mode == CUDAGraphMode.NONE:
            self.cudagraph_capture_sizes = []
            return
        if self.cudagraph_capture_sizes:
            return
        max_bs = min(max_num_seqs, 512)
        sizes = [i for i in (1, 2, 4) if i <= max_bs]
        sizes += list(range(8, min(max_bs + 1, 256), 8))
        if max_bs >= 256:
            sizes += list(range(256, max_bs + 1, 16))
        self.cudagraph_capture_sizes = sorted(set(sizes))
