"""Which model formats are already in the local Hugging Face cache."""

from __future__ import annotations

from functools import lru_cache

from ..catalog import ModelEntry, format_size_bytes


@lru_cache(maxsize=1)
def cached_repo_ids() -> frozenset[str]:
    try:
        from huggingface_hub import scan_cache_dir

        return frozenset(r.repo_id for r in scan_cache_dir().repos)
    except Exception:
        return frozenset()


def local_formats(model: ModelEntry) -> dict[str, bool]:
    cached = cached_repo_ids()
    return {fmt: repo in cached for fmt, repo in model.repos.items()}


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
