from __future__ import annotations

import torch

from lminfer.config import LMInferConfig
from lminfer.model_executor.model_loader import get_model


class GPUModelRunner:
    def __init__(self, config: LMInferConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.model_config = config.model_config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.is_driver = self.tp_size == 1
        self.model: torch.nn.Module | None = None

    def load_model(self) -> None:
        self.model = get_model(self.config, self.device)

    @torch.inference_mode()
    def eager_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        assert self.model is not None
        input_ids = input_ids.reshape(-1)
        hidden_states = self.model(input_ids, positions)
        last_hidden = hidden_states[-1:]
        return self.model.compute_logits(last_hidden)
