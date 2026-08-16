import importlib

from torch import nn

_MODELS: dict[str, tuple[str, str]] = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM")
}


class ModelRegistry:
    @staticmethod
    def resolve_model_cls(architecture: str) -> type[nn.Module]:
        if architecture not in _MODELS:
            raise ValueError(
                f"不支持的模型架构{architecture!r},已经支持：{sorted(_MODELS)}"
            )
        module_name, cls_name = _MODELS[architecture]
        module = importlib.import_module(f"lminfer.model_executor.models.{module_name}")
        return getattr(module, cls_name)
