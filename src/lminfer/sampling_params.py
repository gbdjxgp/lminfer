from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    # TODO: vllm中topk=0表示不做top-k截断
    top_k: int = -1
    max_tokens: int = 16
    ignore_eos: bool = False

    # temperature低于这个值退化为贪心
    _SAMPLING_EPS = 1e-5

    def __post_init__(self):
        assert self.temperature >= 0.0
        assert 0.0 < self.top_p <= 1.0
        assert self.top_k == -1 or self.top_k >= 1
        assert self.max_tokens >= 1
        if self.temperature < self._SAMPLING_EPS:
            # 贪心时这些值没有意义。
            self.temperature = 0.0
            self.top_p = 1.0
            self.top_k = -1

    @property
    def all_greedy(self) -> bool:
        return self.temperature == 0
