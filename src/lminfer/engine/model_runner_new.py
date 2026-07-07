import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from lminfer.config import Config
from lminfer.engine.sequence import Sequence
from lminfer.models.qwen3 import Qwen3ForCausalLM
from lminfer.layers.attention import Attention
from lminfer.layers.sampler import Sampler
from lminfer.utils.context import set_context, get_context, reset_context
from lminfer.utils.buffer import CpuGpuBuffer
from lminfer.utils.loader import load_model
from lminfer.utils.device import deviceinfo


class ModelRunner:
    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
    ) -> None:
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.device = deviceinfo.device(rank)

        deviceinfo.backend.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        self.allocate_runtime_buffers()
        if not self.enforce_eager:
            # 图模式
            self.capture_backendgraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="lminfer", create=True, size=2 * 20)
                dist.barrier()

            else:
                dist.barrier()
                self.shm = SharedMemory(name="lminfer")
                self.loop()

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4, n + 4])
        self.event.clear()
        return method_name, args

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name)
        return method(*args)

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def warmup_model(self):
        deviceinfo.backend.empty_cache()
        deviceinfo.backend.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        assert max_model_len < max_num_batched_tokens
        # 固定2个seqs前向预热，每个seq长度为max_num_batched_tokens//2
        seqs = [Sequence([0] * (max_num_batched_tokens // 2)) for _ in range(2)]
        for seq in seqs:
            seq.num_scheduled_tokens = max_num_batched_tokens // 2
        self.run(seqs, True)
        deviceinfo.backend.empty_cache()

    def allocate_kv_cache(self):
        config, hf_config = self.config, self.config.hf_config
        free, total = deviceinfo.backend.mem_get_info()
        used = total - free
        peak = deviceinfo.backend.memory_stats()["allocated_bytes.all.peak"]
        current = deviceinfo.backend.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        # 显存总量*利用率-当前已用-(峰值-当前)，这个公式保证程序运行期间显存始终不超过利用率
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        # kvcache,(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.backend.k_cache = self.kv_cache[0, layer_id]
                module.backend.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

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

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).to(self.device, non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        # 仅当某些序列有缓存前缀token时才填充；否则在prefill中不使用
        block_tables = None
        for seq in seqs:
            # start = 第一个未缓存token的索引 — 即该序列已有多少个token的KV被缓存（来自前缀缓存）。
            start = seq.num_cached_tokens
            # 此序列在此prefill步骤中需要运行的新token数量。
            seqlen_q = seq.num_scheduled_tokens
            # end = start + seqlen_q -> 最后一个新token之后的位置索引。
            end = start + seqlen_q
            # seqlen_k = 此序列中查询将要关注的总token数（缓存的+新的）,[0:start] 的token已缓存，[start:end] 是新的。
            seqlen_k = end
            # 仅追加新token的ID（我们在此步骤中实际计算的）。
            input_ids.extend(seq[start:end])
            # 新token的位置ID：start, start+1, ..., end-1
            positions.extend(range(start, end))
            # 按此序列的长度扩展运行中的累积长度数组。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # 更新max_seqlen
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                # 如果block_table为空（预热请求,seq对应的block_table为空）
                continue

            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # 当前prefill需要计算的的block
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 考虑prefix cache情况，只取相对block_size的偏移量，例如某个请求的block_size=8,已经缓存了13个token，那么start=13,start%block_size=5,slot_start=block_id*8+5
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 考虑最后一个token，可能不满一个block_size，因此需要更新slot_end
                    slot_end = (
                        seq.block_table[i] * self.block_size + end - i * self.block_size
                    )
                # 理论上slot_end-slot_start<=block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            # prefix cache存在的情况，这两个序列不等，因此需要准备block_tables
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(
            self.device, non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(
            self.device, non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).to(self.device, non_blocking=True)
        # 这个slot_mapping用于存储kvcache用，并且这个是token粒度的，没有block_size的概念
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).to(self.device, non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

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

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        deviceinfo.backend.synchronize()
        dist.destroy_process_group()

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
            graph = deviceinfo.graph_cls()
            # 只会捕获decode阶段
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            # 预热，前向之后返回的数据应该是(total_tokens, hidden_size)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with deviceinfo.backend.graph(graph, self.graph_pool):
                # 捕获图，把前bs行填入buffer中
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                # 只要不是并发replay/输出之间相互依赖, 这种操作可以减少显存占用
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            deviceinfo.backend.synchronize()
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
            assert context.context_lens is not None
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
