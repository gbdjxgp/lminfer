高风险: RowParallelLinear.forward 里重复加了 bias，多卡结果会错。src/lminfer/layers/linear.py (line 58)
当前每个 rank 都执行 linear(x, self.weight, self.bias)，然后再 all_reduce(sum)。这样 bias 也会被求和 tp_size 次。行并行里应该先算无 bias 的局部结果、all_reduce，最后只加一次完整 bias。

高风险: LinearBase 在未初始化分布式进程组时会直接报错。src/lminfer/layers/linear.py (line 9)
dist.get_world_size() 和 dist.get_rank() 需要先 init_process_group()。如果你希望这些层也能单卡/未初始化场景可用，这里要兜底成 tp_size=1, tp_rank=0。

中风险: ColumnParallelLinear.forward 的行为和常见列并行语义不完整。src/lminfer/layers/linear.py (line 35)
它现在只返回本地 shard 输出，没有 all_gather。这不一定错，但要求后续调用方明确知道拿到的是分片结果；如果你原本想要“行为像普通 Linear”，那现在是不对的。

中风险: LayerNorm 实现的其实是 RMSNorm，不是 LayerNorm。src/lminfer/layers/layernorm.py (line 6)
你只做了平方均值归一化，没有减均值，也没有 bias。如果调用方按标准 LayerNorm 理解，会产生行为偏差。名字和语义最好统一。

中风险: LayerNorm 的构造签名不太稳，直接接收 gamma 张量而不是 hidden_size。src/lminfer/layers/layernorm.py (line 7)
这会把“创建模块”和“外部提供已有权重”耦合在一起，不利于和常见 state_dict/初始化方式对齐。除非你就是专门做权重映射层，否则更推荐用 shape 参数创建参数，再靠 loader/load_state_dict 灌权重。

低风险: @torch.compiler 这行看起来不对。src/lminfer/layers/layernorm.py (line 16)
PyTorch 常见的是 torch.compile(...) 包装模块/函数，不是直接 @torch.compiler。如果这是想做编译优化，这里大概率会报错或无效，取决于你本地版本。