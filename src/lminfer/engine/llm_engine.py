import atexit
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from lminfer.config import Config
from lminfer.sampling_params import SamplingParams
from lminfer.engine.sequence import Sequence
from lminfer.engine.scheduler import Scheduler
from lminfer.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)
        # Sequence只需要Rank0持有
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            # 启动其他的rank，创建event用于进程同步。
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            # self.ps用于停止进程用
            self.ps.append(process)
            # self.event用于同步
            self.events.append(event)
        # 创建自己的modelrunner
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # scheduler
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # num_tokens:正数是prefill,否则是decode
        # decode记录num_scheduled_tokens,否则记录seqs的长度(也就是decode阶段的吞吐)
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        )
        # 前向,拿到tokenid
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理,更新seq/kvcache/队列
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        # outputs:当前step完成的请求,num_tokens:吞吐
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 加入到scheduler
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            # 前向一次
            output, num_tokens = self.step()
            # 更新吞吐量
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            # 实时显示吞吐
            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )
            # 这里的output是已经完成的请求
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 处理结束,返回推理结果
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        return outputs
