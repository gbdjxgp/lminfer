import torch.nn as nn
import torch
import torch.distributed as dist


class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, bias=True, tp_dim=0):
        super(LinearBase, self).__init__()
        # self.tp_dim = tp_dim
        # FIXME: This assumes torch.distributed has already been initialized.
        # If not, get_world_size/get_rank will raise instead of falling back to single-process execution.
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self):
        raise NotImplementedError("Forward method should be implemented in subclass")


class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=True):
        tp_size = dist.get_world_size()
        assert (
            output_size % tp_size == 0
        ), "output_size must be divisible by the number of processes"
        super(ColumnParallelLinear, self).__init__(
            input_size, output_size // tp_size, bias, tp_dim=0
        )

    def weight_loader(self, param, loaded_weights):
        param_data = param.data
        full_data_out_features = loaded_weights.size(0)
        shard_size = full_data_out_features // self.tp_size
        assert shard_size == param_data.size(
            0
        ), "Shard size does not match parameter size"
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_sizes, bias=True):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param, loaded_weights, loaded_weight_id):
        param_data = param.data
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        param_data = param_data.narrow(0, offset, shard_size)
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, head_size, num_heads, num_kv_heads=None, bias=False):
        tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        super().__init__(input_size, self.output_size * tp_size, bias=bias)

    def weight_loader(self, param, loaded_weights, load_weight_id):
        param_data = param.data
        assert load_weight_id in ["q", "k", "v"]
        if load_weight_id == "q":
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == "k":
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            offset = (
                self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            )
            shard_size = self.head_size * self.num_kv_heads
        param_data = param.data.narrow(0, offset, shard_size)
        load_weight_start_index = self.tp_rank * shard_size
        param_data.copy_(loaded_weights.narrow(0, load_weight_start_index, shard_size))


class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=True):
        tp_size = dist.get_world_size()
        assert (
            input_size % tp_size == 0
        ), "input_size must be divisible by the number of processes"
        super(RowParallelLinear, self).__init__(
            input_size // tp_size, output_size, bias, tp_dim=1
        )

    def weight_loader(self, param, loaded_weights):
        param_data = param.data
        if param_data.ndim == 1:
            param.data.copy_(loaded_weights)
            return
        full_data_out_features = loaded_weights.size(1)
        shard_size = full_data_out_features // self.tp_size
        assert shard_size == param_data.size(
            1
        ), "Shard size does not match parameter size"
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x):
        # FIXME: This adds the full bias on every rank before all_reduce.
        # With tp_size > 1, bias will be summed tp_size times in the final result.
        # Correct order is local matmul without bias -> all_reduce -> add bias once.
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result
