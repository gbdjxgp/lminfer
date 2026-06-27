from __future__ import annotations

from functools import cache
from typing import Literal

import torch.distributed as dist
import torch


class DeviceInfo:
    def __init__(self) -> None:
        self._cuda_available = self.is_cuda_available()
        self._npu_available = self.is_npu_available()
        self._hip_available = self.is_hip_available()
        self._platform = self.platform()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

    @staticmethod
    @cache
    def get_world_size():
        return dist.get_world_size()

    @staticmethod
    @cache
    def is_cuda_available() -> bool:
        return torch.cuda.is_available() and torch.version.hip is None

    @staticmethod
    @cache
    def is_npu_available() -> bool:
        return hasattr(torch, "npu") and torch.npu.is_available()

    @staticmethod
    @cache
    def is_hip_available() -> bool:
        return torch.cuda.is_available() and torch.version.hip is not None

    @cache
    def platform(self, deviceId: int = None) -> str:
        if deviceId is not None:
            assert type(deviceId) == int
            deviceId = ":" + str(deviceId)
        else:
            deviceId = ""
        if self._cuda_available:
            return "cuda" + deviceId
        if self._npu_available:
            return "npu" + deviceId
        if self._hip_available:
            return "rocm" + deviceId
        assert deviceId is None
        return "cpu"

    @cache
    def device(self, deviceId: int | None = None) -> torch.device:
        return torch.device(self.platform(deviceId=deviceId))


deviceinfo = DeviceInfo()
