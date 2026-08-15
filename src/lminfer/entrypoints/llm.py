from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lminfer.config import (
    CacheConfig,
    CompilationConfig,
    LMInferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from lminfer.outputs import CompletionOutput, RequestOutput
from lminfer.sampling_params import SamplingParams
from lminfer.v1.engine.llm_engine import LLMEngine
from lminfer.v1.request import Request

_CONFIG_FIELDS = {
    "model_config": ModelConfig,
    "cache_config": CacheConfig,
    "parallel_config": ParallelConfig,
    "scheduler_config": SchedulerConfig,
    "compilation_config": CompilationConfig,
}


def make_config(model: str, **kwargs: Any) -> LMInferConfig:
    buckets: dict[str, dict[str, Any]] = {name: {} for name in _CONFIG_FIELDS}
    for key, value in kwargs.items():
        for name, cls in _CONFIG_FIELDS.items():
            if key in cls.__dataclass_fields__:
                buckets[name][key] = value
                break
        else:
            raise TypeError(f"未知的参数：{key!r}")
    buckets["model_config"]["model"] = model
    return LMInferConfig(
        **{name: cls(**buckets[name]) for name, cls in _CONFIG_FIELDS.items()}
    )


class LLM:
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.config = make_config(model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.llm_engine = LLMEngine(self.config)
        print(
            f"[lminfer] model={model} vocab={self.config.model_config.get_vocab_size()}"
            f"hidden={self.config.model_config.get_hidden_size()}"
            f"max_model_len={self.config.model_config.max_model_len}"
            f"tp={self.config.parallel_config.tensor_parallel_size}"
        )

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        eos_token_id = self.tokenizer.eos_token_id
        order: dict[str, int] = {}
        for i, (prompt, params) in enumerate(
            zip(prompts, sampling_params, strict=True)
        ):
            token_ids = (
                prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
            )
            req_id = self.llm_engine.add_request(token_ids, params, eos_token_id)
            order[req_id] = i

        outputs: list[RequestOutput | None] = [None] * len(prompts)
        pbar = (
            tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        while self.llm_engine.has_unfinished_requests():
            for request in self.llm_engine.step():
                outputs[order[request.request_id]] = self._make_output(request)
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()
        return outputs

    def _make_output(self, request: Request) -> RequestOutput:
        token_ids = request.output_token_ids
        return RequestOutput(
            request_id=request.request_id,
            prompt=None,
            prompt_token_ids=request.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=self.tokenizer.decode(token_ids, skip_special_tokens=True),
                    token_ids=token_ids,
                    finish_reason=request.get_finished_reason(),
                )
            ],
            finished=True,
        )
