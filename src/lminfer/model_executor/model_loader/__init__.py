import torch
from torch import nn

from lminfer.config import LMInferConfig, set_current_config
from lminfer.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from lminfer.model_executor.models.registry import ModelRegistry

__all__ = ["get_model"]


def get_model(config: LMInferConfig, device: torch.device) -> nn.Module:
    model_config = config.model_config
    model_cls = ModelRegistry.resolve_model_cls(model_config.architecture)

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(model_config.dtype)
    torch.set_default_device(device)
    try:
        with set_current_config(config):
            model = model_cls(config)
            model.load_weights(safetensors_weights_iterator(model_config.model))
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
    return model.eval()
