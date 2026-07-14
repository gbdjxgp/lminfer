import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from lminfer.utils import device as device_module

from lminfer.config import Config
from lminfer.engine.sequence import Sequence
from lminfer.models.qwen3 import Qwen3ForCausalLM
from lminfer.layers.attention import Attention
from lminfer.layers.sampler import Sampler
from lminfer.utils.context import set_context, get_context, reset_context
from lminfer.utils.buffer import CpuGpuBuffer
from lminfer.utils.loader import load_model
from lminfer.engine.model_runner import ModelRunner


class NPUModelRunner(ModelRunner):
    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
    ) -> None:
        super().__init__(config, rank, event)
        if self.rank == 0:
            self.allocate_runtime_buffers()

    def allocate_runtime_buffers(self):
        max_bs = self.config.max_num_seqs
        max_num_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size
        self.decode_input_ids = CpuGpuBuffer(
            max_bs, dtype=torch.int64, device=self.device
        )
        self.decode_positions = CpuGpuBuffer(
            max_bs, dtype=torch.int64, device=self.device
        )
        self.decode_slot_mapping = CpuGpuBuffer(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.decode_context_lens = CpuGpuBuffer(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.decode_block_tables = CpuGpuBuffer(
            max_bs, max_num_blocks, dtype=torch.int32, device=self.device
        )
        self.temperatures = CpuGpuBuffer(
            max_bs, dtype=torch.float32, device=self.device
        )
        self.temperature_scalar = torch.empty(
            (), dtype=torch.float32, device=self.device
        )
        # blocktable的长度(fastpath下面配合seq_ids检测是否可重复)
        self.decode_block_table_seq_ids = [-1] * max_bs
        # blocktable的长度(fastpath下面配合seq_ids检测是否可重复)
        self.decode_block_table_num_blocks = [0] * max_bs
        # 配合更新block_tables(那些行需要重复值)
        self.decode_changed_rows: list[int] = []
        # 配合block_tables(索引用)
        self.decode_max_num_blocks = 0
        # 缓存
        self.prev_decode_seq_ids: list[int] = []
        # 缓存
        self.prev_decode_token_ids: torch.Tensor | None = None

    def prepare_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        # 拿到每个sequence的id用于比较
        seq_ids = [seq.seq_id for seq in seqs]
        # 下面判断是否可以走快速路径，判断标准：之前是decode，且seqid不变
        if (
            self.prev_decode_token_ids is not None
            and len(self.prev_decode_seq_ids) >= bs
            and self.prev_decode_seq_ids[:bs] == seq_ids
        ):
            # fastpath:直接复用前面的输出token_ids
            self.decode_input_ids.gpu[:bs].copy_(
                self.prev_decode_token_ids[:bs], non_blocking=True
            )
            # TODO:下面这些变量，CPU tensor是不是都没用到呢？
            # 位置+1，position（位置编码用）/context_lens(attention用)/slotmapping（存储kv用）
            self.decode_positions.gpu[:bs].add_(1)
            # 这里先对所有请求的全局slot+1，但是这可能是错的，因为某些seq会跨block
            self.decode_slot_mapping.gpu[:bs].add_(1)
            # 这个在NPU下要指定为是CPU的设备
            self.decode_context_lens.cpu[:bs].add_(1)

            # 下面是在更新block_tables，因为可能涉及到新的block被分配
            changed_rows: list[int] = []
            # fastpath下max_num_blocks只增不减
            max_num_blocks = self.decode_max_num_blocks
            for i, seq in enumerate(seqs):
                num_blocks = len(seq.block_table)
                max_num_blocks = max(max_num_blocks, num_blocks)
                if seq.last_block_num_tokens != 1:
                    # 最后一个block不是1->不需要修正block_tables和slot_mapping
                    continue
                # TODO:理论上没必要全部初始化再赋值，直接将新的一个block更新一下就行，flash-attn输入并非一定要-1，当然这个需要验证
                # 某个seq有新block加入更新decode_block_tables/slot_mapping
                self.decode_slot_mapping.gpu[i] = seq.block_table[-1] * self.block_size
                # 现在CPU下面准备变量，赋值快,先初始化block_table
                self.decode_block_tables.cpu[i].fill_(-1)
                # 赋值新的block_table
                for j, block_id in enumerate(seq.block_table):
                    self.decode_block_tables.cpu[i, j] = block_id
                # # cpu到gpu数据同步
                # self.decode_block_tables.gpu[i, :num_blocks].copy_(
                #     self.decode_block_tables.cpu[i, :num_blocks], non_blocking=True
                # )
                # 更新seqid与num_blocks
                self.decode_block_table_seq_ids[i] = seq.seq_id
                self.decode_block_table_num_blocks[i] = num_blocks
                changed_rows.append(i)
            for row in changed_rows:
                # cpu到gpu数据同步
                self.decode_block_tables.gpu[row, :max_num_blocks].copy_(
                    self.decode_block_tables.cpu[row, :max_num_blocks],
                    non_blocking=True,
                )
            if self.device.type != "npu":
                self.decode_context_lens.copy_to_gpu(bs)
            # 更新改变的行的列表
            self.decode_changed_rows = changed_rows
            # 更新本轮最大block
            self.decode_max_num_blocks = max_num_blocks
            # 更新本轮的seqid
            self.prev_decode_seq_ids = seq_ids
            set_context(
                False,
                slot_mapping=self.decode_slot_mapping.gpu[:bs],
                context_lens=(
                    self.decode_context_lens.cpu[:bs]
                    if self.device.type == "npu"
                    else self.decode_context_lens.gpu[:bs]
                ),
                block_tables=self.decode_block_tables.gpu[:bs, :max_num_blocks],
            )
            return self.decode_input_ids.gpu[:bs], self.decode_positions.gpu[:bs]
        # 非fastpath
        max_num_blocks = 0
        changed_rows: list[int] = []
        for i, seq in enumerate(seqs):
            # decode阶段，input_ids只包含最后一个token
            self.decode_input_ids.cpu[i] = seq.last_token
            # 最后一个token的位置
            self.decode_positions.cpu[i] = len(seq) - 1
            # decode阶段bs=1,这里就是最后一个token对应的slot_mapping
            self.decode_slot_mapping.cpu[i] = (
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            # Attention的每个序列的长度
            self.decode_context_lens.cpu[i] = len(seq)
            num_blocks = len(seq.block_table)
            max_num_blocks = max(max_num_blocks, num_blocks)
            if (
                self.decode_block_table_seq_ids[i] != seq.seq_id
                or self.decode_block_table_num_blocks[i] != num_blocks
            ):
                # seq_id或者num_blocks不等，这两个条件都是必须的，seqid表示请求是否还是原来的请求，num_blocks表示是否有新的block_table
                # 还是重建block_tables
                self.decode_block_tables.cpu[i].fill_(-1)
                for j, block_id in enumerate(seq.block_table):
                    self.decode_block_tables.cpu[i, j] = block_id
                # # cpu到gpu数据同步
                # self.decode_block_tables.gpu[i, :num_blocks].copy_(
                #     self.decode_block_tables.cpu[i, :num_blocks],
                #     non_blocking=True,
                # )
                self.decode_block_table_seq_ids[i] = seq.seq_id
                self.decode_block_table_num_blocks[i] = num_blocks
                changed_rows.append(i)
        for row in changed_rows:
            # cpu到gpu数据同步
            self.decode_block_tables.gpu[row, :max_num_blocks].copy_(
                self.decode_block_tables.cpu[row, :max_num_blocks],
                non_blocking=True,
            )
        self.decode_input_ids.copy_to_gpu(bs)
        self.decode_positions.copy_to_gpu(bs)
        self.decode_slot_mapping.copy_to_gpu(bs)
        if self.device.type != "npu":
            self.decode_context_lens.copy_to_gpu(bs)
        self.decode_changed_rows = changed_rows
        self.decode_max_num_blocks = max_num_blocks
        self.prev_decode_seq_ids = seq_ids
        set_context(
            False,
            slot_mapping=self.decode_slot_mapping.gpu[:bs],
            context_lens=(
                self.decode_context_lens.cpu[:bs]
                if self.device.type == "npu"
                else self.decode_context_lens.gpu[:bs]
            ),
            block_tables=self.decode_block_tables.gpu[:bs, :max_num_blocks],
        )
        return self.decode_input_ids.gpu[:bs], self.decode_positions.gpu[:bs]

    def prepare_sample(self, seqs: list[Sequence]):
        if not hasattr(self, "temperature_scalar"):
            # 预热阶段会执行到
            temperatures = [seq.temperature for seq in seqs]
            first_temperature = temperatures[0]
            if all(t == first_temperature for t in temperatures[1:]):
                return torch.tensor(
                    first_temperature, dtype=torch.float32, device=self.device
                )
            return torch.tensor(
                temperatures, dtype=torch.float32, pin_memory=True, device="cpu"
            ).to(self.device, non_blocking=True)
        # 有缓存的路径,优先用缓存
        first_temperature = seqs[0].temperature
        if all(seq.temperature == first_temperature for seq in seqs[1:]):
            self.temperature_scalar.fill_(first_temperature)
            return self.temperature_scalar
        bs = len(seqs)

        for i, seq in enumerate(seqs):
            self.temperatures.cpu[i] = seq.temperature
        self.temperatures.copy_to_gpu(bs)
        return self.temperatures.gpu[:bs]

    @torch.inference_mode()
    def capture_backendgraph(self):
        config = self.config
        hf_config = config.hf_config
        # 最大512个seqs
        max_bs = min(self.config.max_num_seqs, 512)
        # 单条请求的最大blocks数量，向上取整
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        if self.device.type == "npu":
            # TODO: 这个context_lens是否能收编到NPU上？
            with torch.inference_mode(False):
                context_lens = torch.zeros(
                    max_bs, dtype=torch.int32, pin_memory=True, device="cpu"
                )
        else:
            context_lens = torch.zeros(max_bs, dtype=torch.int32)
        # block_tables全0
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = device_module.deviceinfo.graph_cls()
            # 只会捕获decode阶段
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            # 预热，前向之后返回的数据应该是(total_tokens, hidden_size)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with device_module.deviceinfo.backend.graph(graph, self.graph_pool):
                # 捕获图，把前bs行填入buffer中
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                # 只要不是并发replay/输出之间相互依赖, 这种操作可以减少显存占用
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            device_module.deviceinfo.backend.synchronize()
            reset_context()
        # 保存对应图的输入输出地址
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        # 该函数返回的值是调用compute_logits后面的结果。返回的logits: (num_seqs, vocab_size)
        # TODO: 这里的512应该与capture_backendgraph里面捕获的图大小联动，目前是写死的
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # prefill/单算子模式/长序列，单算子模式
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 图模式图模式,设置输入,graph_vars中只需要前bs的有效数据
            bs = input_ids.size(0)
            context = get_context()
            # 对应的计算图
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            if self.decode_changed_rows:
                for row in self.decode_changed_rows:
                    # 更新block_tables!
                    graph_vars["block_tables"][row, : self.decode_max_num_blocks] = (
                        context.block_tables[row, : self.decode_max_num_blocks]
                    )
            # 重放图
            graph.replay()
            # 返回值只取前bs个
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 准备输入
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        # 温度
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # logits: (num_seqs, vocab_size)
        logits = self.run_model(input_ids, positions, is_prefill)
        # token_ids: (num_seqs)
        token_ids = self.sampler(logits, temperatures) if self.rank == 0 else None
        if self.rank == 0:
            # 缓存decode的token，为后续fastpath做准备
            if not is_prefill:
                # prefill可能是在初始化，干脆跳过这个
                self.prev_decode_token_ids = token_ids
            token_ids = token_ids.tolist()
        reset_context()
        return token_ids
