"""Which model formats are already in the local Hugging Face cache."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..catalog import ModelEntry, format_size_bytes


@lru_cache(maxsize=1)
def cached_repo_ids() -> frozenset[str]:
    try:
        from huggingface_hub import scan_cache_dir

        return frozenset(r.repo_id for r in scan_cache_dir().repos)
    except Exception:
        return frozenset()


@lru_cache(maxsize=16)
def cached_gguf_payload(repo_id: str) -> tuple[str, tuple[str, ...]] | None:
    """(snapshot_dir, relative .gguf file names) for a cached GGUF repo, or None."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                root = Path(repo.repo_path)
                files = tuple(sorted(str(p.relative_to(root)) for p in root.rglob("*.gguf")))
                return (str(root), files)
    except Exception:
        return None
    return None


def local_formats(model: ModelEntry) -> dict[str, bool | str | tuple[str, ...]]:
    cached = cached_repo_ids()
    out: dict[str, bool | str | tuple[str, ...]] = {
        fmt: repo in cached for fmt, repo in model.repos.items()
    }
    # ADR-0002: the engine takes GGUF weights as a file (--gguf <file>), so
    # expose where the downloaded single-file quants actually live.
    gguf_repo = model.repo("gguf")
    if gguf_repo and out.get("gguf"):
        payload = cached_gguf_payload(gguf_repo)
        if payload:
            out["gguf_dir"], out["gguf_files"] = payload
    return out


def model_payload(model: ModelEntry) -> dict:
    cached = cached_repo_ids()
    formats = []
    for fmt, repo in model.repos.items():
        size = format_size_bytes(model, fmt)
        formats.append(
            {
                "format": fmt,
                "repo": repo,
                "local": repo in cached,
                "size_gb": round(size / 1024**3, 1) if size else None,
            }
        )
    return {
        "key": model.key,
        "label": model.label,
        "family": model.family,
        "params_b": model.params_b,
        "media": model.media,
        "formats": formats,
    }
