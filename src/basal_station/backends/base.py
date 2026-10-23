"""Static backend profile definitions.

Each profile maps to one `basal-serve --mode` invocation and one catalog
weight format. `requires` tags are checked against the HardwareProfile:
  darwin / linux  — platform gates
  cuda            — needs an NVIDIA GPU
  mlx             — needs Apple Silicon
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendProfile:
    key: str
    label: str
    mode: str        # basal-serve --mode value
    fmt: str         # catalog format key: gguf | mlx | fp8 | safetensors
    requires: tuple[str, ...]
    media: bool = False
    default_quant: str = "Q4_K_M"


BACKENDS: tuple[BackendProfile, ...] = (
    BackendProfile(
        key="mlx", label="MLX 8-bit · Apple Silicon", mode="mlx", fmt="mlx",
        requires=("darwin", "mlx"),
    ),
    BackendProfile(
        key="gguf-cuda", label="GGUF · CUDA (partial offload)", mode="gguf", fmt="gguf",
        requires=("linux", "cuda"),
    ),
    BackendProfile(
        key="gguf-metal", label="GGUF · Metal", mode="gguf", fmt="gguf",
        requires=("darwin",),
    ),
    BackendProfile(
        key="fp8-vllm", label="FP8 · vLLM (Hopper/Blackwell)", mode="vllm", fmt="fp8",
        requires=("cuda",),
    ),
    BackendProfile(
        key="eager", label="PyTorch eager · full media support", mode="eager",
        fmt="safetensors", requires=(), media=True,
    ),
    BackendProfile(
        key="mps", label="PyTorch MPS (macOS fallback)", mode="mps", fmt="safetensors",
        requires=("darwin",),
    ),
    BackendProfile(
        key="gguf-cpu", label="GGUF · CPU", mode="gguf", fmt="gguf",
        requires=(),
    ),
)


def get_backend(key: str) -> BackendProfile | None:
    return next((b for b in BACKENDS if b.key == key), None)
