from __future__ import annotations

import enum

from lminfer.sampling_params import SamplingParams
from lminfer.v1.core.kv_cache_utils import BlockHash


class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: RequestStatus) -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: RequestStatus) -> str | None:
        return _FINISHED_REASON.get(status)


_FINISHED_REASON = {
    RequestStatus.FINISHED_STOPPED: "stop",
    RequestStatus.FINISHED_LENGTH_CAPPED: "length",
    RequestStatus.FINISHED_ABORTED: "abort",
}


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        eos_token_id: int | None,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.eos_token_id = eos_token_id
        self.max_tokens = sampling_params.max_tokens
        self.status = RequestStatus.WAITING

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(prompt_token_ids)

        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = prompt_token_ids.copy()

        # num_computed_tokens:已经算完并且写进kvcache的token数
        self.num_computed_tokens = 0
        # 已经调度出去，但是采样结果还没到host的token数量
        self.num_output_placeholders = 0
        self.block_hashes: list[BlockHash] = []

    @property
    def num_tokens(self) -> int:
        # 已经确定的token总数
        return len(self._all_token_ids)

    @property
    def num_tokens_with_placeholders(self) -> int:
        # 所有已经调度的token(包括已经调度出去但是没得到结果的)
        return len(self._all_token_ids) + self.num_output_placeholders

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self._output_token_ids

    @property
    def all_token_ids(self) -> list[int]:
        return self._all_token_ids

    @property
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> str | None:
        return RequestStatus.get_finished_reason(self.status)

    def append_output_token_ids(self, token_id: int) -> None:
        self._output_token_ids.append(token_id)
        self._all_token_ids.append(token_id)
        if self.num_output_placeholders:
            # 之前占位的情况,现在真值到了,更新占位token数量
            self.num_output_placeholders -= 1
