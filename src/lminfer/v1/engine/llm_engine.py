from __future__ import annotations

import itertools

from lminfer.config import LMInferConfig
from lminfer.sampling_params import SamplingParams
from lminfer.v1.requests import Request, RequestStatus


class LLMEngine:
    def __init__(self, config: LMInferConfig) -> None:
        self.config = config
        self._requests: dict[str, Request] = {}
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
        self._requests[request_id] = Request(
            request_id,
            prompt_token_ids,
            sampling_params,
            eos_token_id,
        )
        return request_id

    def has_unfinished_requests(self) -> bool:
        return bool(self._requests)

    def step(self) -> list[Request]:
        done = list(self._requests.values())
        for request in done:
            request.status = RequestStatus.FINISHED_STOPPED
        self._requests.clear()
        return done

    def shutdown(self) -> None:
        pass
