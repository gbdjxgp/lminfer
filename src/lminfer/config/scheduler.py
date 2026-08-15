from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    # 一次前向最多token数（主要约束prefill）
    max_num_batched_tokens: int = 8192
    # 一次前向最多的序列数
    max_num_seqs: int = 256

    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    async_scheduling: bool = True
