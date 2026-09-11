from __future__ import annotations

import itertools

from lminfer.config import LMInferConfig
from lminfer.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from lminfer.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from lminfer.v1.core.sched.request_queue import FCFSRequestQueue
from lminfer.v1.kv_cache_interface import KVCacheConfig
from lminfer.v1.outputs import ModelRunnerOutput
from lminfer.v1.request import Request, RequestStatus


class Scheduler:
    def __init__(self, config: LMInferConfig, kv_cache_config: KVCacheConfig) -> None:
        scheduler_config = config.scheduler_config
        self.max_num_running_reqs = scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config,
            max_model_len=scheduler_config.max_model_len,
            enable_caching=config.cache_config.enable_prefix_caching,
        )
        self.waiting = FCFSRequestQueue()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        # 上一步结束的请求，随下一次 schedule 一起告诉 worker
        self.finished_req_ids: set[str] = set()
        # 上一步调度过哪些请求。中断后重新进入 batch 的请求要补发一次全量 token
        self.prev_step_scheduled_req_ids: set[str] = set()

    def add_request(self, request: Request) -> None:
        self.requests[request.request_id] = request
        self.waiting.add_request(request)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        token_budget = self.max_num_scheduled_tokens
        preempted = False

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # 本轮调度的新的token：占位的token减去已经计算的token，是本轮的token预算。
            num_new_tokens = min(
                request.num_tokens_with_placeholders - request.num_computed_tokens,
                token_budget,
            )
            # TODO: 调度后这个变量应该会更新，因此running应该始终是有值的。
            assert num_new_tokens > 0
            while True:
                # 抢占逻辑
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is not None:
                    # 分配成功，直接退出while
                    break
                # 否则被抢占。
                preempted = True
                self._preempt(self.running.pop())
                # 抢占逻辑：从running的0号队尾开始尝试抢占，顺序：free(-1)->尝试分配当前请求->失败的话free(-2)->尝试分配当前请求...
                if request.status is RequestStatus.PREEMPTED:
                    # 如果当前请求也设置为抢占了，则停止prefill，改为decode。
                    break
            if request.status is RequestStatus.PREEMPTED:
                # 调度失败，改为decode
                break
            # 调度成功，更新prefill的值。
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            req_index += 1

        while (
            not preempted
            and self.waiting
            and token_budget > 0
            and len(self.running) < self.max_num_running_reqs
        ):
            # 有kvcache且有等待队列且有调度预算且在限制内

            request = self.waiting.peek_request()
            # TODO: 分析下kvcache manager，这里的逻辑
            computed_blocks, num_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            # num_tokens:已经确定的token长度，num_computed_tokens是kvcache命中的长度。
            num_new_tokens = min(request.num_tokens - num_computed_tokens, token_budget)
            if num_new_tokens <= 0:
                # 如果<=0，表示不需要decode调度，这是防御性编程。实际上这种状态是错误的。正常文本路径会补BOS,实际上这里不可达。
                break
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, num_computed_tokens, computed_blocks
            )
            if new_blocks is None:
                # TODO: 这里有bug么？有没有可能prefill与decode都不够？
                break

            self.waiting.pop_request()
            resumed = request.status is RequestStatus.PREEMPTED
            request.status = RequestStatus.RUNNING
            # 复用缓存
            request.num_computed_tokens = num_computed_tokens
            self.running.append(request)
            (scheduled_resumed_reqs if resumed else scheduled_new_reqs).append(request)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            # 新请求 / 恢复的请求都要拿到全量 block（命中的前缀块 + 新分的）
            req_to_new_blocks[request.request_id] = computed_blocks + new_blocks

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id=r.request_id,
                    prompt_token_ids=r.prompt_token_ids,
                    sampling_params=r.sampling_params,
                    block_ids=req_to_new_blocks[r.request_id].get_block_ids(),
                    num_computed_tokens=r.num_computed_tokens,
                )
                for r in scheduled_new_reqs
            ],
            scheduled_cached_reqs=self._make_cached_request_data(
                scheduled_running_reqs, scheduled_resumed_reqs, req_to_new_blocks
            ),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=self.max_num_scheduled_tokens - token_budget,
            finished_req_ids=self.finished_req_ids,
        )
        self.finished_req_ids = set()
        self.prev_step_scheduled_req_ids = set(num_scheduled_tokens)
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        resumed_req_ids: set[str] = set()
        all_token_ids: dict[str, list[int]] = {}
        new_block_ids: list[tuple[list[int], ...] | None] = []
        num_computed_tokens: list[int] = []

        num_running = len(running_reqs)
        for idx, request in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = request.request_id
            req_ids.append(req_id)
            if idx >= num_running:
                resumed_req_ids.add(req_id)
            if req_id not in self.prev_step_scheduled_req_ids:
                # 上一步没被调度：worker 那边缓存的 token 可能已经跟不上了，补一份全量
                all_token_ids[req_id] = request.all_token_ids.copy()
            new_block_ids.append(req_to_new_blocks[req_id].get_block_ids())
            num_computed_tokens.append(request.num_computed_tokens)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # TODO: 搞清楚这里异步调度的逻辑。
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_scheduled

    def _preempt(self, request: Request) -> None:
        """把请求踢回等待队首，已算的 KV 全部释放，下次被调度时重算 prompt。"""
        request.status = RequestStatus.PREEMPTED
        # TODO: 这里抢占之后，已经计算的token都为0了，因此需要重新计算？？？不知道重新prefill还是需要已经计算的token全部从头来
        request.num_computed_tokens = 0
        request.num_output_placeholders = 0
        request.block_hashes.clear()
        self.kv_cache_manager.free(request)
        # prepend为放到等待队列的队首，优先调度。
        self.waiting.prepend_request(request)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> list[Request]:
        finished: list[Request] = []
        for req_id, new_token_ids in zip(
            model_runner_output.req_ids,
            model_runner_output.sampled_token_ids,
            strict=True,
        ):
            request = self.requests.get(req_id)
            if request is None or request.status is not RequestStatus.RUNNING:
                continue
            stopped = False
            for token_id in new_token_ids:
                request.append_output_token_ids(token_id)
                if self._check_stop(request, token_id):
                    stopped = True
                    break
            self.kv_cache_manager.cache_blocks(
                request, min(request.num_computed_tokens, request.num_tokens)
            )
            if stopped:
                self._free_request(request)
                finished.append(request)
        return finished

    def _check_stop(self, request: Request, token_id: int) -> bool:
        if request.num_output_tokens >= request.max_tokens:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        if not request.sampling_params.ignore_eos and token_id == request.eos_token_id:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    def _free_request(self, request: Request) -> None:
        request.num_output_placeholders = 0
        self.kv_cache_manager.free(request)
        self.running.remove(request)
        self.requests.pop(request.request_id, None)
        self.finished_req_ids.add(request.request_id)
