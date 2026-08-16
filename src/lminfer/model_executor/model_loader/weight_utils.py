import os
from collections.abc import Iterator
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    assert param.data.shape == loaded_weight.shape, (
        f"权重shape不匹配！param:{param.data.shape} vs loaded_weight:{loaded_weight.shape}"
    )
    param.data.copy_(loaded_weight)


def safetensors_weights_iterator(
    model_path: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    assert files, f"{model_path}下面没safetensors文件"
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)
