from basal_station.backends.base import get_backend
from basal_station.backends.registry import (
    plan_gguf_offload,
    rank_backends,
    rank_models_for_hardware,
)
from basal_station.catalog import format_size_bytes, get_model
from basal_station.hardware import GIB, GpuInfo, HardwareProfile


def _linux_cuda_3gb() -> HardwareProfile:
    return HardwareProfile(
        os="Linux", arch="x86_64", is_apple_silicon=False, chip=None,
        cpu_count=8, ram_bytes=32 * GIB,
        gpus=[GpuInfo("cuda", "NVIDIA GeForce RTX 3060", 3 * GIB)],
    )


def _mac_m5() -> HardwareProfile:
    return HardwareProfile(
        os="Darwin", arch="arm64", is_apple_silicon=True, chip="Apple M5 Pro",
        cpu_count=14, ram_bytes=48 * GIB,
        gpus=[GpuInfo("metal", "Apple M5 Pro", 48 * GIB)],
    )


# ---------------- -ngl planning ----------------

def test_plan_gguf_offload_small_vram_partial():
    m = get_model("omni-1.0-max")
    size = format_size_bytes(m, "gguf", "Q4_K_M")
    ngl = plan_gguf_offload(size, m.n_layers, 3 * GIB)
    # (3 GB * 0.90 - 0.4 GB) / (7.4 GB / 48 layers) ≈ 15 layers
    assert 10 <= ngl <= 22
    assert ngl < m.n_layers


def test_plan_gguf_offload_full_when_fits():
    m = get_model("2.0-4.5B")
    size = format_size_bytes(m, "gguf", "Q4_K_M")  # ~2.8 GB
    ngl = plan_gguf_offload(size, m.n_layers, 8 * GIB)
    assert ngl == m.n_layers


def test_plan_gguf_offload_zero_when_no_vram():
    assert plan_gguf_offload(7 * GIB, 48, 0) == 0
    assert plan_gguf_offload(7 * GIB, 48, int(0.3 * GIB)) == 0


# ---------------- backend ranking ----------------

def test_mac_recommends_mlx_first():
    hw = _mac_m5()
    m = get_model("omni-1.0-max")
    ranked = rank_backends(hw, m)
    eligible = [r for r in ranked if r.eligible]
    assert eligible, ranked
    assert eligible[0].profile.key == "mlx"
    keys = {r.profile.key for r in ranked}
    assert "gguf-cuda" not in keys or not next(
        r for r in ranked if r.profile.key == "gguf-cuda"
    ).eligible
    # eager fits in 48GB unified memory (bf16 12B ≈ 25 GB)
    eager = next(r for r in ranked if r.profile.key == "eager")
    assert eager.eligible


def test_linux_3gb_recommends_gguf_cuda_with_partial_ngl():
    hw = _linux_cuda_3gb()
    m = get_model("omni-1.0-max")
    ranked = rank_backends(hw, m)
    gguf_cuda = next(r for r in ranked if r.profile.key == "gguf-cuda")
    assert gguf_cuda.eligible
    assert 1 <= gguf_cuda.ngl < m.n_layers
    best_eligible = next(r for r in ranked if r.eligible)
    assert best_eligible.profile.key == "gguf-cuda"
    # FP8 cannot fit on 3 GB
    fp8 = next(r for r in ranked if r.profile.key == "fp8-vllm")
    assert not fp8.eligible
    # eager bf16 12B needs ~25 GB VRAM on discrete — ineligible
    eager = next(r for r in ranked if r.profile.key == "eager")
    assert not eager.eligible


def test_linux_3gb_high_offload_for_small_model():
    hw = _linux_cuda_3gb()
    m = get_model("2.0-4.5B")
    ranked = rank_backends(hw, m)
    gguf_cuda = next(r for r in ranked if r.profile.key == "gguf-cuda")
    assert gguf_cuda.eligible
    # ~2.8 GB model on 3 GB card: most layers offloaded, remainder in RAM
    assert 20 <= gguf_cuda.ngl < m.n_layers
    assert gguf_cuda.ngl / m.n_layers > 0.7


def test_no_gpu_linux_falls_back_to_cpu():
    hw = HardwareProfile(
        os="Linux", arch="x86_64", is_apple_silicon=False, chip=None,
        cpu_count=8, ram_bytes=32 * GIB, gpus=[],
    )
    ranked = rank_backends(hw, get_model("omni-1.0-max"))
    eligible_keys = [r.profile.key for r in ranked if r.eligible]
    assert eligible_keys[0] == "gguf-cpu"
    assert "gguf-cuda" not in eligible_keys
    assert "mlx" not in eligible_keys


def test_ngl_override_fixed():
    hw = _linux_cuda_3gb()
    m = get_model("omni-1.0-max")
    ranked = rank_backends(hw, m, gguf_ngl_override=10)
    gguf_cuda = next(r for r in ranked if r.profile.key == "gguf-cuda")
    assert gguf_cuda.ngl == 10


# ---------------- model suggestions ----------------

def test_model_suggestions_prefer_small_for_3gb():
    hw = _linux_cuda_3gb()
    suggestions = rank_models_for_hardware(hw)
    top = [s["model"] for s in suggestions[:3]]
    assert "2.0-4.5B" in top
    assert suggestions[0]["fit"] >= suggestions[-1]["fit"]


def test_backend_lookup():
    assert get_backend("mlx").mode == "mlx"
    assert get_backend("nope") is None
