import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from lminfer.config import Config
from lminfer.engine.sequence import Sequence
from lminfer.models.qwen3 import Qwen3ForCausalLM
from lminfer.layers.sampler import Sampler
from lminfer.utils.context import set_context, get_context, reset_context
from lminfer.utils.loader import load_model
from lminfer.utils.device import deviceinfo

# from torch_npu.contrib import transfer_to_npu


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
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

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
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # decode阶段，input_ids只包含最后一个token
            input_ids.append(seq.last_token)
            # 最后一个token的位置
            positions.append(len(seq) - 1)
            # Attention的每个序列的长度
            context_lens.append(len(seq))
            # 最后一个token对应的slot_mapping
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(
            self.device, non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(
            self.device, non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).to(self.device, non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).to(self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).to(self.device, non_blocking=True)
        return temperatures

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

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).to(self.device, non_blocking=True)
        return temperatures

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
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
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
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids
