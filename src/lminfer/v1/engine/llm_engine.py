from __future__ import annotations

import itertools

import torch

from lminfer.config import LMInferConfig
from lminfer.sampling_params import SamplingParams
from lminfer.v1.core.kv_cache_utils import get_kv_cache_config
from lminfer.v1.core.sched.output import SchedulerOutput
from lminfer.v1.core.sched.scheduler import Scheduler
from lminfer.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from lminfer.v1.outputs import ModelRunnerOutput
from lminfer.v1.request import Request


class LLMEngine:
    def __init__(self, config: LMInferConfig) -> None:
        self.config = config
        kv_cache_config = self._make_kv_cache_config()
        self.scheduler = Scheduler(config, kv_cache_config)
        self.request_counter = itertools.count()

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        eos_token_id: int | None = None,
        request_id: str | None = None,
    ) -> str:
        if request_id is None:
            request_id = str(next(self.request_counter))
        self.scheduler.add_request(
            Request(request_id, prompt_token_ids, sampling_params, eos_token_id)
        )
        return request_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def step(self) -> list[Request]:
        scheduler_output = self.scheduler.schedule()
        output = self._simulate_execute(scheduler_output)
        if output is None:
            return []
        return self.scheduler.update_from_output(scheduler_output, output)

    def _simulate_execute(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:

        if scheduler_output.is_empty:
            return None
        req_ids: list[str] = []
        sampled_token_ids: list[list[int]] = []
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.scheduler.requests[req_id]
            req_ids.append(req_id)
            if request.num_computed_tokens >= request.num_tokens:
                # TODO: remove this.
                sampled_token_ids.append([0])

            else:
                sampled_token_ids.append([])
        return ModelRunnerOutput(req_ids, sampled_token_ids)

    def shutdown(self) -> None:
        pass

    def _make_kv_cache_config(self) -> KVCacheConfig:
        spec = FullAttentionSpec(
            block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
        )
        page_bytes = spec.page_size_bytes
        return get_kv_cache_config([{"layers.0": spec}], [1024 * page_bytes])
