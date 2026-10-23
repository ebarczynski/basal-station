"""Catalog of basal typed-decision models and their per-format Hugging Face repos.

Formats:
  safetensors — bf16 reference weights (basal-serve --mode eager / mps)
  fp8         — FP8 checkpoints for vLLM on Hopper/Blackwell NVIDIA GPUs
  gguf        — llama.cpp / Ollama / Metal; single-file quants inside the repo
  mlx         — Apple Silicon MLX ports (8-bit unless noted)
"""

from __future__ import annotations

from dataclasses import dataclass, field

FORMAT_LABELS = {
    "safetensors": "Safetensors (bf16)",
    "fp8": "FP8 (vLLM)",
    "gguf": "GGUF (llama.cpp / Metal / CPU)",
    "mlx": "MLX 8-bit (Apple Silicon)",
}

# Rough compressed bytes-per-parameter factors for VRAM planning.
GGUF_BYTES_PER_PARAM = {"Q4_K_M": 0.62, "Q5_K_M": 0.71, "Q6_K": 0.80, "Q8_0": 1.06}
DEFAULT_GGUF_QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
BF16_BYTES_PER_PARAM = 2.05
FP8_BYTES_PER_PARAM = 1.15
MLX8_BYTES_PER_PARAM = 1.06

R = "Remek"


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    family: str
    params_b: float | None          # billions of parameters; None = unknown
    media: bool                     # omni family: image/audio capable (eager mode)
    n_layers: int | None
    repos: dict[str, str] = field(default_factory=dict)  # format -> repo id

    @property
    def is_omni(self) -> bool:
        return self.family.startswith("omni")

    def repo(self, fmt: str) -> str | None:
        return self.repos.get(fmt)

    def formats(self) -> list[str]:
        order = ["gguf", "mlx", "fp8", "safetensors"]
        return [f for f in order if f in self.repos]


def _e(key: str, label: str, family: str, params_b: float | None, media: bool,
       n_layers: int | None, **repos: str | None) -> ModelEntry:
    return ModelEntry(
        key=key, label=label, family=family, params_b=params_b, media=media,
        n_layers=n_layers,
        repos={k: v for k, v in repos.items() if v},
    )


CATALOG: list[ModelEntry] = [
    # --- omni 1.0 (media-capable) ---
    _e("omni-1.0-max", "basal-omni-1.0-max (12B, media)", "omni-1.0", 12.0, True, 48,
        safetensors=f"{R}/basal-omni-1.0-max",
        fp8=f"{R}/basal-omni-1.0-max-FP8",
        gguf=f"{R}/basal-omni-1.0-max-GGUF",
        mlx=f"{R}/basal-omni-1.0-max-MLX"),
    _e("omni-1.0-mini", "basal-omni-1.0-mini (media)", "omni-1.0", 4.5, True, 34,
        safetensors=f"{R}/basal-omni-1.0-mini",
        fp8=f"{R}/basal-omni-1.0-mini-FP8",
        gguf=f"{R}/basal-omni-1.0-mini-GGUF",
        mlx=f"{R}/basal-omni-1.0-mini-MLX"),
    _e("omni-1.0-ultra", "basal-omni-1.0-ultra (media)", "omni-1.0", None, True, None,
        safetensors=f"{R}/basal-omni-1.0-ultra",
        fp8=f"{R}/basal-omni-1.0-ultra-FP8",
        gguf=f"{R}/basal-omni-1.0-ultra-GGUF"),
    _e("omni-1.1-max", "basal-omni-1.1-max (12B, media)", "omni-1.1", 12.0, True, 48,
        safetensors=f"{R}/basal-omni-1.1-max",
        mlx=f"{R}/basal-omni-1.1-max-MLX"),
    _e("omni-pico", "basal-omni-pico (edge, media)", "omni-pico", 1.0, True, 24,
        safetensors=f"{R}/basal-omni-pico",
        gguf=f"{R}/basal-omni-pico-GGUF"),
    # --- basal 2.0 ---
    _e("2.0-max", "basal-2.0-max (12B)", "2.0", 12.0, False, 48,
        safetensors=f"{R}/basal-2.0-max",
        fp8=f"{R}/basal-2.0-max-FP8",
        gguf=f"{R}/basal-2.0-max-GGUF",
        mlx=f"{R}/basal-2.0-max-MLX-8bit"),
    _e("2.0-4.5B", "basal-2.0-4.5B", "2.0", 4.5, False, 34,
        safetensors=f"{R}/basal-2.0-4.5B",
        fp8=f"{R}/basal-2.0-4.5B-FP8",
        gguf=f"{R}/basal-2.0-4.5B-GGUF",
        mlx=f"{R}/basal-2.0-4.5B-MLX-8bit"),
    _e("2.0-mini", "basal-2.0-mini", "2.0", 1.5, False, 26,
        safetensors=f"{R}/basal-2.0-mini",
        fp8=f"{R}/basal-2.0-mini-FP8",
        gguf=f"{R}/basal-2.0-mini-GGUF",
        mlx=f"{R}/basal-2.0-mini-MLX-8bit"),
    # --- basal 1.5 ---
    _e("1.5-max", "basal-1.5-max", "1.5", 12.0, False, 48,
        safetensors=f"{R}/basal-1.5-max",
        fp8=f"{R}/basal-1.5-max-FP8",
        gguf=f"{R}/basal-1.5-max-GGUF",
        mlx=f"{R}/basal-1.5-max-MLX-8bit"),
    _e("1.5-4.5B", "basal-1.5-4.5B", "1.5", 4.5, False, 34,
        safetensors=f"{R}/basal-1.5-4.5B",
        fp8=f"{R}/basal-1.5-4.5B-FP8",
        gguf=f"{R}/basal-1.5-4.5B-GGUF",
        mlx=f"{R}/basal-1.5-4.5B-MLX-8bit"),
    _e("1.5-mini", "basal-1.5-mini", "1.5", 1.5, False, 26,
        safetensors=f"{R}/basal-1.5-mini",
        fp8=f"{R}/basal-1.5-mini-FP8",
        gguf=f"{R}/basal-1.5-mini-GGUF",
        mlx=f"{R}/basal-1.5-mini-MLX-8bit"),
    # --- basal 1.0 ---
    _e("1.0-4.5B", "basal-1.0-4.5B", "1.0", 4.5, False, 34,
        safetensors=f"{R}/basal-1.0-4.5B",
        fp8=f"{R}/basal-1.0-4.5B-FP8"),
    _e("1.0-1.5B", "basal-1.0-1.5B", "1.0", 1.5, False, 26,
        safetensors=f"{R}/basal-1.0-1.5B",
        fp8=f"{R}/basal-1.0-1.5B-FP8"),
]

_BY_KEY = {m.key: m for m in CATALOG}


def get_model(key: str) -> ModelEntry | None:
    return _BY_KEY.get(key)


def list_models() -> list[ModelEntry]:
    return list(CATALOG)


def format_size_bytes(model: ModelEntry, fmt: str, quant: str | None = None) -> int | None:
    """Approximate on-disk / in-memory size in bytes, or None when unknown."""
    if model.params_b is None:
        return None
    p = model.params_b
    if fmt == "gguf":
        factor = GGUF_BYTES_PER_PARAM.get(quant or "Q4_K_M", 0.62)
        return int(p * factor * 1e9)
    if fmt == "fp8":
        return int(p * FP8_BYTES_PER_PARAM * 1e9)
    if fmt == "mlx":
        return int(p * MLX8_BYTES_PER_PARAM * 1e9)
    return int(p * BF16_BYTES_PER_PARAM * 1e9)  # safetensors bf16
