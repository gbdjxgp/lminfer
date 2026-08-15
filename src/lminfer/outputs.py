from dataclasses import dataclass, field


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = False
