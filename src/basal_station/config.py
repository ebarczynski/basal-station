"""Configuration: YAML file + env overrides, deep-merged."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONFIG_FILENAME = "basal-station.yaml"


def default_config_paths() -> list[Path]:
    return [
        Path.cwd() / CONFIG_FILENAME,
        Path.home() / ".config" / "basal-station" / "config.yaml",
    ]


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8800


class EngineConfig(BaseModel):
    command: str = "basal-serve"
    host: str = "127.0.0.1"
    port: int = 8000
    startup_timeout: float = 300.0
    extra_args: list[str] = Field(default_factory=list)
    gguf_ngl: str | int = "auto"  # display-only offload estimate (ADR-0002)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class KnowledgeConfig(BaseModel):
    enabled: bool = True
    command: list[str] = Field(default_factory=lambda: ["kp", "mcp"])
    env: dict[str, str] = Field(default_factory=dict)
    default_top_k: int = 5
    timeout_s: float = 30.0


class AppSettings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    default_model: str = "omni-1.0-max"
    auto_backend: bool = True


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def env_overrides() -> dict[str, Any]:
    patch: dict[str, Any] = {}

    def section(name: str) -> dict[str, Any]:
        return patch.setdefault(name, {})

    if v := os.getenv("BASAL_STATION_SERVER_PORT"):
        section("server")["port"] = int(v)
    if v := os.getenv("BASAL_STATION_ENGINE_PORT"):
        section("engine")["port"] = int(v)
    if v := os.getenv("BASAL_STATION_ENGINE_COMMAND"):
        section("engine")["command"] = v
    if v := os.getenv("KP_COMMAND"):
        section("knowledge")["command"] = shlex.split(v)
    if v := os.getenv("KNOWLEDGE_PALACE_COMMAND"):
        section("knowledge")["command"] = shlex.split(v)
    return patch


def find_config(explicit: str | os.PathLike | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for p in default_config_paths():
        if p.is_file():
            return p
    return None


def load_settings(explicit: str | os.PathLike | None = None) -> AppSettings:
    data: dict[str, Any] = {}
    path = find_config(explicit)
    if path:
        data = yaml.safe_load(path.read_text()) or {}
    merged = _deep_merge(data, env_overrides())
    return AppSettings(**merged)
