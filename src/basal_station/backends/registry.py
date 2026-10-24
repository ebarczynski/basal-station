"""Hardware-aware backend ranking and engine argument construction."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..catalog import (
    DEFAULT_GGUF_QUANTS,
    GGUF_BYTES_PER_PARAM,
    ModelEntry,
    format_size_bytes,
    list_models,
)
from ..hardware import GIB, HardwareProfile
from .base import BACKENDS, BackendProfile

# Fraction of VRAM usable for weights (rest: KV cache, CUDA context, display).
VRAM_USABLE_FRACTION = 0.90
# Estimated non-weight GPU memory at an 8k context.
CTX_OVERHEAD_BYTES = int(0.4 * GIB)


def plan_gguf_offload(
    model_bytes: int,
    n_layers: int,
    vram_bytes: int,
    ctx_overhead: int = CTX_OVERHEAD_BYTES,
    usable_fraction: float = VRAM_USABLE_FRACTION,
) -> int:
    """How many transformer layers to offload to GPU (-ngl) for partial offload."""
    if vram_bytes <= 0 or n_layers <= 0 or model_bytes <= 0:
        return 0
    usable = vram_bytes * usable_fraction - ctx_overhead
    if usable <= 0:
        return 0
    bytes_per_layer = model_bytes / n_layers
    return max(0, min(n_layers, int(usable // bytes_per_layer)))


@dataclass
class RankedBackend:
    profile: BackendProfile
    eligible: bool
    reason: str
    model_ref: str | None = None      # repo id passed as --model (tokenizer/identity)
    quant: str | None = None
    ngl: int | None = None            # display-only offload estimate (ADR-0002)
    est_vram_gb: float | None = None
    local: bool = False               # weights already in the HF cache
    gguf_file: str | None = None      # local .gguf file, passed as --gguf <file>
    extra_args: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "key": self.profile.key,
            "label": self.profile.label,
            "mode": self.profile.mode,
            "format": self.profile.fmt,
            "eligible": self.eligible,
            "reason": self.reason,
            "model_ref": self.model_ref,
            "quant": self.quant,
            "ngl": self.ngl,
            "est_vram_gb": self.est_vram_gb,
            "local": self.local,
            "gguf_file": self.gguf_file,
            "media": self.profile.media,
        }


def _gguf_file_for(local_formats: dict, quant: str | None) -> str | None:
    """Local downloaded .gguf file matching the quant, or None (ADR-0002)."""
    files = list(local_formats.get("gguf_files") or [])
    if not files:
        return None
    if quant:
        matches = [f for f in files if quant.lower() in f.lower()]
        files = matches or files
    gguf_dir = local_formats.get("gguf_dir")
    if not gguf_dir:
        return None
    return str(Path(gguf_dir) / files[0])


def _requirements_met(profile: BackendProfile, hw: HardwareProfile) -> str | None:
    for req in profile.requires:
        if req == "darwin" and hw.os != "Darwin":
            return "requires macOS"
        if req == "linux" and hw.os != "Linux":
            return "requires Linux"
        if req == "cuda" and hw.cuda_gpu is None:
            return "requires an NVIDIA GPU"
        if req == "mlx" and not hw.is_apple_silicon:
            return "requires Apple Silicon"
    return None


def _gguf_quant_for(hw: HardwareProfile, model: ModelEntry, vram_bytes: int) -> tuple[str, int]:
    """Pick the largest quant that makes sense, and compute the offload plan.

    On a small-VRAM card, prefer Q4_K_M for the best quality-per-byte while
    still being worth offloading; on big VRAM go up the ladder.
    """
    budget = vram_bytes * VRAM_USABLE_FRACTION - CTX_OVERHEAD_BYTES
    chosen = "Q4_K_M"
    for quant in DEFAULT_GGUF_QUANTS:
        size = format_size_bytes(model, "gguf", quant)
        if size is None:
            continue
        if size <= budget and budget > 0:
            chosen = quant
    size = format_size_bytes(model, "gguf", chosen)
    n_layers = model.n_layers or 32
    if size is None:
        # Unknown size: assume it fits fully if the raw repo fits in VRAM 3x over,
        # else partial with a conservative 12B-class estimate.
        size = int((model.params_b or 12.0) * GGUF_BYTES_PER_PARAM["Q4_K_M"] * 1e9)
    ngl = plan_gguf_offload(size, n_layers, vram_bytes)
    return chosen, ngl


def rank_backends(
    hw: HardwareProfile,
    model: ModelEntry,
    prefer_media: bool = False,
    local_formats: dict | None = None,
    gguf_ngl_override: str | int = "auto",
) -> list[RankedBackend]:
    """Rank backend profiles for this machine + model. Best first."""
    local_formats = local_formats or {}
    cuda = hw.cuda_gpu
    unified_budget = int(hw.ram_bytes * 0.75)
    ranked: list[RankedBackend] = []

    for profile in BACKENDS:
        req_issue = _requirements_met(profile, hw)
        if req_issue:
            ranked.append(RankedBackend(profile, False, req_issue))
            continue

        if profile.fmt not in model.repos:
            ranked.append(RankedBackend(profile, False, f"model has no {profile.fmt} weights"))
            continue

        if profile.key == "gguf-cuda":
            assert cuda is not None
            quant, ngl = _gguf_quant_for(hw, model, cuda.vram_bytes)
            if gguf_ngl_override != "auto" and isinstance(gguf_ngl_override, int):
                ngl = max(0, min(model.n_layers or 64, gguf_ngl_override))
            if ngl < 1:
                ranked.append(RankedBackend(profile, False, "not enough VRAM to offload any layer"))
                continue
            size = format_size_bytes(model, "gguf", quant)
            est = min(ngl / (model.n_layers or 32), 1.0) * (size or 0)
            offload_pct = round(100 * ngl / (model.n_layers or 32))
            ranked.append(
                RankedBackend(
                    profile, True,
                    f"offload ~{offload_pct}% of layers ({quant}) to {cuda.name} "
                    "(engine decides the real offload; estimate only)",
                    model_ref=model.repo("gguf"), quant=quant, ngl=ngl,
                    est_vram_gb=round(est / GIB, 1),
                    local=bool(local_formats.get("gguf")),
                    gguf_file=_gguf_file_for(local_formats, quant),
                )
            )
            continue

        if profile.key == "fp8-vllm":
            assert cuda is not None
            need = format_size_bytes(model, "fp8")
            if need is None:
                ranked.append(RankedBackend(profile, False, "model size unknown"))
                continue
            if cuda.vram_bytes < int(need * 1.25):
                ranked.append(
                    RankedBackend(
                        profile, False,
                        f"needs ~{need * 1.25 / GIB:.0f} GB VRAM (FP8 + vLLM headroom), "
                        f"GPU has {cuda.vram_bytes / GIB:.0f} GB",
                    )
                )
                continue
            ranked.append(
                RankedBackend(
                    profile, True, f"FP8 fits in {cuda.vram_bytes / GIB:.0f} GB VRAM",
                    model_ref=model.repo("fp8"),
                    est_vram_gb=round(need / GIB, 1),
                    local=bool(local_formats.get("fp8")),
                )
            )
            continue

        if profile.fmt in ("safetensors", "mlx"):
            need = format_size_bytes(model, profile.fmt)
            budget = unified_budget if hw.unified_memory else (cuda.vram_bytes if cuda else 0)
            if need is not None and need > budget:
                label = "unified memory" if hw.unified_memory else "VRAM"
                ranked.append(
                    RankedBackend(
                        profile, False,
                        f"needs ~{need / GIB:.0f} GB {label}, budget {budget / GIB:.0f} GB",
                    )
                )
                continue
            note = ""
            if profile.key == "eager" and model.media:
                note = " (images/audio enabled)"
            ranked.append(
                RankedBackend(
                    profile, True, f"{model.label} fits this machine{note}",
                    model_ref=model.repo(profile.fmt),
                    est_vram_gb=round((need or 0) / GIB, 1),
                    local=bool(local_formats.get(profile.fmt)),
                )
            )
            continue

        # gguf-cpu / gguf-metal
        suffix = "always available" if profile.key == "gguf-cpu" else "Metal unified memory"
        quant = "Q4_K_M"
        ranked.append(
            RankedBackend(
                profile, True, f"CPU/Metal path — {suffix}",
                model_ref=model.repo("gguf"), quant=quant,
                local=bool(local_formats.get("gguf")),
                gguf_file=_gguf_file_for(local_formats, quant),
            )
        )

    def score(rb: RankedBackend) -> tuple:
        if not rb.eligible:
            return (0, 0, 0)
        key = rb.profile.key
        if prefer_media and rb.profile.media:
            base = 100
        elif hw.is_apple_silicon and key == "mlx" or key == "fp8-vllm":
            base = 95
        elif key == "gguf-cuda":
            # The higher the offload fraction, the better.
            offload = (rb.ngl or 0) / (model.n_layers or 32)
            base = 80 + int(15 * offload)
        elif key == "mlx" and hw.is_apple_silicon:
            base = 90
        elif key in ("gguf-metal", "eager", "mps"):
            base = 70
        else:
            base = 50  # gguf-cpu: last resort
        # Slight bonus for already-downloaded weights.
        return (base + (2 if rb.local else 0), rb.ngl or 0, 0)

    ranked.sort(key=score, reverse=True)
    return ranked


def rank_models_for_hardware(hw: HardwareProfile) -> list[dict]:
    """Suggest catalog models ordered by fit for this machine."""
    suggestions = []
    for model in list_models():
        media_bonus = 5 if model.media else 0
        size = format_size_bytes(model, "gguf", "Q4_K_M")
        if size is None:
            fit, note = 30, "size unknown"
        else:
            cuda = hw.cuda_gpu
            budget = int(hw.ram_bytes * 0.75) if hw.unified_memory else (
                cuda.vram_bytes if cuda else 0
            )
            if size <= budget:
                fit, note = 90 + media_bonus, "fully offloadable on this machine"
            else:
                budget_cpu = int(hw.ram_bytes * 0.75)
                if size <= budget_cpu:
                    fit, note = 60 + media_bonus, "runs partially offloaded / mostly in RAM"
                else:
                    fit, note = 20, "exceeds memory — expect heavy swapping"
        recency = 10 if model.family.startswith("omni-1") else 5 if model.family >= "1.5" else 0
        suggestions.append({"model": model.key, "label": model.label, "fit": fit + recency, "note": note})
    suggestions.sort(key=lambda s: s["fit"], reverse=True)
    return suggestions


def resolve_engine_command(command: str) -> str:
    """Find basal-serve: explicit path > sibling of this venv > PATH."""
    p = Path(command).expanduser()
    if p.is_file() and os_access(p):
        return str(p)
    sibling = Path(sys.executable).parent / command
    if sibling.is_file():
        return str(sibling)
    found = shutil.which(command)
    if found:
        return found
    return command  # let subprocess fail with a clear error


def os_access(p: Path) -> bool:
    import os

    return os.access(p, os.X_OK)


def build_engine_args(
    hw: HardwareProfile,
    model: ModelEntry,
    profile: BackendProfile,
    ranked: RankedBackend,
    engine_host: str,
    engine_port: int,
    extra_args: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Assemble the basal-serve command line for a backend choice.

    ADR-0002: the engine's real GGUF interface is `--gguf <file>` (v2.3.5 and
    v2.4.1 define neither -ngl nor --threads). `--model` carries the repo id
    for tokenizer/identity; a resolved local .gguf file rides on --gguf.
    """
    args = [
        resolve_engine_command("basal-serve"),
        "--model",
        ranked.model_ref or model.repo(profile.fmt) or model.key,
        "--mode",
        profile.mode,
        "--host",
        engine_host,
        "--port",
        str(engine_port),
    ]
    info: dict = {
        "backend": profile.key,
        "model": model.key,
        "model_ref": ranked.model_ref,
        "mode": profile.mode,
    }
    if profile.fmt == "gguf" and ranked.quant:
        info["quant"] = ranked.quant
    if profile.fmt == "gguf" and ranked.gguf_file:
        args += ["--gguf", ranked.gguf_file]
        info["gguf_file"] = ranked.gguf_file
    if profile.media and model.media:
        args += ["--media", "on"]
        info["media"] = True
    if extra_args:
        args += list(extra_args)
    return args, info
