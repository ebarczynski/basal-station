from basal_station.hardware import GIB, GpuInfo, HardwareProfile, parse_nvidia_smi


def test_parse_nvidia_smi_single_gpu():
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 3070, 8192\n")
    assert len(gpus) == 1
    assert gpus[0].vendor == "cuda"
    assert gpus[0].name == "NVIDIA GeForce RTX 3070"
    assert gpus[0].vram_bytes == 8192 * 1024 * 1024


def test_parse_nvidia_smi_multi_and_garbage():
    out = "NVIDIA A100, 40960\n\n[N/A], N/A\nbadline\nNVIDIA RTX 3060, 12288"
    gpus = parse_nvidia_smi(out)
    assert [g.name for g in gpus] == ["NVIDIA A100", "NVIDIA RTX 3060"]


def _profile(**kw) -> HardwareProfile:
    defaults = dict(
        os="Linux", arch="x86_64", is_apple_silicon=False, chip=None,
        cpu_count=8, ram_bytes=32 * GIB,
        gpus=[GpuInfo("cuda", "NVIDIA GeForce RTX 3060", 3 * GIB)],
    )
    defaults.update(kw)
    return HardwareProfile(**defaults)


def test_cuda_property():
    hw = _profile()
    assert hw.cuda_gpu is not None
    assert hw.cuda_gpu.vram_bytes == 3 * GIB
    assert not hw.unified_memory
    assert hw.os_label == "Linux"

    mac = _profile(
        os="Darwin", arch="arm64", is_apple_silicon=True, chip="Apple M5 Pro",
        gpus=[GpuInfo("metal", "Apple M5 Pro", 24 * GIB)],
    )
    assert mac.unified_memory
    assert mac.cuda_gpu is None
