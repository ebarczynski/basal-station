"""Backend profiles and hardware-aware selection."""

from .base import BACKENDS, BackendProfile, get_backend
from .manager import EngineError, EngineManager
from .registry import (
    RankedBackend,
    build_engine_args,
    plan_gguf_offload,
    rank_backends,
    rank_models_for_hardware,
)

__all__ = [
    "BACKENDS",
    "BackendProfile",
    "EngineError",
    "EngineManager",
    "RankedBackend",
    "build_engine_args",
    "get_backend",
    "plan_gguf_offload",
    "rank_backends",
    "rank_models_for_hardware",
]
