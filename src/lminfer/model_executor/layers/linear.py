import torch
import torch.nn.functional as F
from torch import nn

_TP_SIZE = 1
_TP_RANK = 0


def _divide(a: int, b: int) -> int:
    assert a % b == 0, f"{a}不能整除{b}"
    return a // b


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_size = _TP_SIZE
        self.tp_rank = _TP_RANK
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.parameter, loaded_weight: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里要牢记：深度学习中的linear与线性代数中的定义不太一样
        # 线性代数中的是y=Ax+b,其中b是列向量，而深度学习中的类似这里进行转置了
        # 实际上是out=xW^T+bias，因此这里的bias实际上是行向量
        raise NotImplementedError


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        # 列并行,默认tp切分维度为0,也就是输出维度
        super().__init__(input_size, _divide(output_size, _TP_SIZE), bias, tp_dim=0)

    def weight_loader(
        self,
        param: nn.parameter,
        loaded_weight: torch.Tensor,
    ):
        shard_size = param.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: int,
        bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        # 这个类比较特殊，在output_size维度，又要进行TP切分，又要进行output_size合并。
        # 这里要求每个分片（shard_id）都要在tp卡上均分！这很重要，
        # 因为这个层的下游时gate/up，各自时RowParallelLinear
        # 下游行并行要求每个输入（gate/up）在每个卡上均分
        # 否则的话在列并行后行并行前要多余做一次卡间通信。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param.data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight.chunk(self.tp_size, 0)[self.tp_rank])


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ) -> None:
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # self.num_heads与self.num_kv_heads已经是按照TP_SIZE缩放过了！
        self.num_heads = _divide(total_num_heads, _TP_SIZE)
        self.num_kv_heads = _divide(total_num_kv_heads, _TP_SIZE)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        # 注意：这里计算的所有偏移（q_size/kv_size/shard_offset等都是已经被TP_SIZE缩放过了的）
        q_size = self.head_size * self.num_heads
        kv_size = self.head_size * self.num_kv_heads
        shard_offset, shard_size = {
            "q": (0, q_size),
            "k": (q_size, kv_size),
            "v": (q_size + kv_size, kv_size),
        }[loaded_shard_id]
        param_data = param.data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight.chunk(self.tp_size, 0)[self.tp_rank])


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__(
            _divide(input_size, _TP_SIZE),
            output_size,
            bias,
            tp_dim=1,
        )

    def weight_loader(
        self,
        param: nn.parameter,
        loaded_weight: torch.Tensor,
    ):
        if param.data.ndim == 1:
            # 带有bias的情况，bias应该存在所有的权重上面
            # 简化表示：xW+b->cat(x1*W1+b,x2*W2+b,x3*W3+b)。
            # 其中x123为列切分，W123为行切分。
            param.data.copy_(loaded_weight)
            return
        shard_size = param.data.size(1)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(1, start, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 这里的实现是错误的，TP情况下bias会被重复相加，导致后续all_reduce之后的结果是错误的。
        # bias必须在all_reduce之后加入。
        return F.linear(x, self.weight, self.bias)
