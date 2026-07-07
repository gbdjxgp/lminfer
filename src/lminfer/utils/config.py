from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class LmInferConfig:
    activation_backend: str = "auto"


default_config = LmInferConfig()
_current_lminfer_config = None


@contextmanager
def set_current_lminfer_config(cfg: LmInferConfig):
    global _current_lminfer_config
    old = _current_lminfer_config
    _current_lminfer_config = cfg
    try:
        yield
    finally:
        _current_lminfer_config = old


def get_current_lminfer_config():
    return _current_lminfer_config
