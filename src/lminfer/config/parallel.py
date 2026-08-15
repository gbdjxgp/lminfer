from dataclasses import dataclass


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    # 单卡走UniProcExecutor，多卡走MultiProcExecutor
    distributed_executor_backend: str | None = None
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    def __post_init__(self):
        assert 1 <= self.tensor_parallel_size <= 8
        if self.distributed_executor_backend is None:
            self.distributed_executor_backend = (
                "uni" if self.tensor_parallel_size == 1 else "mp"
            )

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size
