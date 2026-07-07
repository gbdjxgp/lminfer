import torch


class CpuGpuBuffer:
    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        self.cpu = torch.empty(
            *size,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.gpu = torch.empty(*size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> None:
        if n is None:
            self.gpu.copy_(self.cpu, non_blocking=True)
        else:
            self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
