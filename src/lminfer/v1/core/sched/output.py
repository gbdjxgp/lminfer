from __future__ import annotations

from dataclasses import dataclass

from lminfer.sampling_params import SamplingParams


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    # 单个request的block_id，按照kvcache group分组
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


@dataclass
class CachedRequestData:
    req_ids: list[str]
    # 被抢占之后恢复的请求。
    resumed_req_ids: set[str]
    # TODO: 什么是需要重同步的请求
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls([], set(), {}, [], [])

    def __len__(self) -> int:
        return len(self.req_ids)


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData

    num_scheduled_tokens: dict[str, int]
    # TODO:这个字段要干什么
    total_num_scheduled_tokens: int
    finished_req_ids: set[str]

    @property
    def is_empty(self) -> bool:
        return self.total_num_scheduled_tokens == 0
