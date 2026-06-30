import torch.nn as nn
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lminfer.utils import deviceinfo


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_size = deviceinfo.tp_size
        self.tp_rank = deviceinfo.tp_rank
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self):
        raise NotImplementedError


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = deviceinfo.tp_size
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param: nn.parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        shard_size = param.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weights.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_shard_id: int
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(0, shard_offset, shard_size)
        loaded_weight = loaded_weights.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        tp_size = deviceinfo.tp_size
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size

        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias=bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weights: torch.Tensor,
        load_weight_id: int,
    ):
        param_data = param.data
        assert load_weight_id in ["q", "k", "v"]
        if load_weight_id == "q":
            shard_offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == "k":
            # 权重顺序应该是Q-K-V,因此这里的偏移是Q的
            shard_offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            # Q+K的偏移
            shard_offset = (
                self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            )
            shard_size = self.head_size * self.num_kv_heads
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weights.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weights)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = deviceinfo.tp_size
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param.data.copy_(loaded_weights)
            return
        shard_size = param_data.size(self.tp_dim)
        start_index = self.tp_rank * shard_size
        loaded_weight = loaded_weights.narrow(self.tp_dim, start_index, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        if self.tp_size > 1:
            y = F.linear(x, self.weight, None)
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            if self.bias:
                y += self.bias
        else:
            y = F.linear(x, self.weight, self.bias)
        return y
