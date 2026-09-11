from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelRunnerOutput:
    """
    req_ids:各行对应请求。顺序为InputBatch的行序。
    sampled_token_ids:内层列表为空表示没有效输出。
    """

    req_ids: list[str]
    sampled_token_ids: list[list[int]]
