"""ADR-0002 regression: the station speaks the engine's real GGUF contract.

Engine v2.3.5/v2.4.1 argparse defines --gguf but neither -ngl nor --threads
(measured against rkinas/basal source). The station used to emit -ngl/--threads
and died at argparse before loading a single weight.
"""

import shutil
import subprocess

import pytest

from basal_station.api import localrepo
from basal_station.backends.registry import (
    _gguf_file_for,
    build_engine_args,
    rank_backends,
)
from basal_station.backends.base import get_backend
from basal_station.catalog import get_model
from basal_station.hardware import GIB, GpuInfo, HardwareProfile


def linux_cuda_3gb() -> HardwareProfile:
    return HardwareProfile(
        os="Linux", arch="x86_64", is_apple_silicon=False, chip=None,
        cpu_count=8, ram_bytes=16 * GIB,
        gpus=[GpuInfo("cuda", "NVIDIA GeForce RTX 3060", 3 * GIB)],
    )


def _ranked_gguf(model_key: str, local_formats=None):
    hw = linux_cuda_3gb()
    model = get_model(model_key)
    assert model is not None
    profile = get_backend("gguf-cuda")
    ranked = next(
        r for r in rank_backends(hw, model, local_formats=local_formats)
        if r.profile.key == "gguf-cuda"
    )
    return hw, model, profile, ranked


def test_build_args_use_gguf_file_when_cached(monkeypatch):
    monkeypatch.setattr(localrepo, "cached_repo_ids",
                        lambda: frozenset({"Remek/basal-2.0-4.5B-GGUF"}))
    monkeypatch.setattr(localrepo, "cached_gguf_payload", lambda repo: (
        "/home/u/.cache/hf/hub/models--Remek--basal-2.0-4.5B-GGUF/snapshots/abc",
        ("basal-2.0-4.5B-Q4_K_M.gguf", "basal-2.0-4.5B-Q8_0.gguf"),
    ))
    model = get_model("2.0-4.5B")
    lf = localrepo.local_formats(model)
    hw, model, profile, ranked = _ranked_gguf("2.0-4.5B", lf)
    args, info = build_engine_args(hw, model, profile, ranked, "127.0.0.1", 8000)

    assert "--gguf" in args
    assert args[args.index("--gguf") + 1].endswith("basal-2.0-4.5B-Q4_K_M.gguf")
    assert args[args.index("--model") + 1] == "Remek/basal-2.0-4.5B-GGUF"
    # the flags the engine does not define must never come back
    assert "-ngl" not in args
    assert "--threads" not in args
    assert info["gguf_file"].endswith(".gguf")


def test_build_args_without_local_file_omit_gguf():
    hw, model, profile, ranked = _ranked_gguf("2.0-4.5B")
    args, info = build_engine_args(hw, model, profile, ranked, "127.0.0.1", 8000)
    assert "--gguf" not in args
    assert "-ngl" not in args
    assert "--threads" not in args
    assert "gguf_file" not in info


def test_gguf_file_prefers_requested_quant():
    lf = {
        "gguf_dir": "/snap",
        "gguf_files": ("model-Q4_K_M.gguf", "model-Q8_0.gguf"),
    }
    assert _gguf_file_for(lf, "Q8_0") == "/snap/model-Q8_0.gguf"
    assert _gguf_file_for(lf, "Q4_K_M") == "/snap/model-Q4_K_M.gguf"
    assert _gguf_file_for(lf, None) == "/snap/model-Q4_K_M.gguf"  # sorted first
    assert _gguf_file_for({"gguf_files": ()}, "Q8_0") is None
    assert _gguf_file_for({"gguf_dir": "/snap", "gguf_files": ("a.gguf",)}, None) == "/snap/a.gguf"


def test_engine_help_exposes_gguf_flag():
    """Live contract guard against the installed engine; skipped without one."""
    engine = shutil.which("basal-serve") or shutil.which("basal")
    if engine is None:
        pytest.skip("basal engine not installed on this machine")
    out = subprocess.run([engine, "--help"], capture_output=True, text=True, timeout=30)
    helptext = out.stdout + out.stderr
    assert "--gguf" in helptext
    assert "-ngl" not in helptext.replace("--single", ""), (
        "engine grew an -ngl flag? revisit ADR-0002 before re-adding it"
    )
