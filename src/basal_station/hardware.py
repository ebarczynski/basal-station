"""Hardware detection: platform, chip, RAM, NVIDIA VRAM."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

GIB = 1024**3
MIB = 1024**2


@dataclass
class GpuInfo:
    vendor: str  # "cuda" | "metal" | "cpu"
    name: str
    vram_bytes: int


@dataclass
class HardwareProfile:
    os: str
    arch: str
    is_apple_silicon: bool
    chip: str | None
    cpu_count: int
    ram_bytes: int
    gpus: list[GpuInfo] = field(default_factory=list)

    @property
    def unified_memory(self) -> bool:
        return self.is_apple_silicon

    @property
    def cuda_gpu(self) -> GpuInfo | None:
        return next((g for g in self.gpus if g.vendor == "cuda"), None)

    @property
    def os_label(self) -> str:
        return {"Darwin": "macOS", "Linux": "Linux"}.get(self.os, self.os)

    def summary(self) -> dict:
        return {
            "os": self.os_label,
            "arch": self.arch,
            "chip": self.chip,
            "apple_silicon": self.is_apple_silicon,
            "unified_memory": self.unified_memory,
            "cpu_count": self.cpu_count,
            "ram_gb": round(self.ram_bytes / GIB, 1),
            "gpus": [
                {
                    "vendor": g.vendor,
                    "name": g.name,
                    "vram_gb": round(g.vram_bytes / GIB, 1),
                }
                for g in self.gpus
            ],
        }


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _linux_ram_bytes() -> int:
    try:
        text = Path("/proc/meminfo").read_text()
        m = re.search(r"MemTotal:\s+(\d+)\s*kB", text)
        if m:
            return int(m.group(1)) * 1024
    except Exception:
        pass
    return 0


def parse_nvidia_smi(output: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`.

    memory is reported in MiB.
    """
    gpus: list[GpuInfo] = []
    for line in output.strip().splitlines():
        if not line.strip() or "[N/A]" in line or "N/A" in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram = int(float(parts[1])) * MIB
        except ValueError:
            continue
        if vram <= 0:
            continue
        gpus.append(GpuInfo("cuda", name, vram))
    return gpus


def _query_nvidia() -> list[GpuInfo]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    try:
        out = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return parse_nvidia_smi(out.stdout)
    except Exception:
        return []


def _detect() -> HardwareProfile:
    os_name = platform.system()
    arch = platform.machine()
    is_as = os_name == "Darwin" and arch == "arm64"
    chip = _sysctl("machdep.cpu.brand_string") if is_as else None
    cpu_count = os_cpu_count()
    ram = int(_sysctl("hw.memsize") or 0) if os_name == "Darwin" else _linux_ram_bytes()

    gpus: list[GpuInfo] = []
    if is_as:
        gpus.append(GpuInfo("metal", chip or "Apple GPU", ram))
    gpus.extend(_query_nvidia())
    return HardwareProfile(
        os=os_name,
        arch=arch,
        is_apple_silicon=is_as,
        chip=chip,
        cpu_count=cpu_count,
        ram_bytes=ram,
        gpus=gpus,
    )


def os_cpu_count() -> int:
    return os.cpu_count() or 1


@lru_cache(maxsize=1)
def _cached() -> HardwareProfile:
    return _detect()


def detect(refresh: bool = False) -> HardwareProfile:
    if refresh:
        _cached.cache_clear()
    return _cached()
