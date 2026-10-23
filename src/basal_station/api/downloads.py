"""Hugging Face download jobs with progress tracking."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from ..catalog import ModelEntry

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


class DownloadError(RuntimeError):
    pass


def job_key(model_key: str, fmt: str, quant: str | None) -> str:
    return f"{model_key}:{fmt}:{quant or 'default'}"


def jobs_status() -> dict[str, dict[str, Any]]:
    with _lock:
        return {k: dict(v) for k, v in _jobs.items()}


def get_job(key: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(key)
        return dict(job) if job else None


def _update(key: str, **fields: Any) -> None:
    with _lock:
        if key in _jobs:
            _jobs[key].update(fields)


def submit(model: ModelEntry, fmt: str, quant: str | None) -> str:
    repo = model.repo(fmt)
    if repo is None:
        raise DownloadError(f"model '{model.key}' has no {fmt} repo")
    key = job_key(model.key, fmt, quant)
    with _lock:
        existing = _jobs.get(key)
        if existing and existing["status"] in ("queued", "downloading"):
            raise DownloadError(f"download already running: {key}")
        _jobs[key] = {
            "key": key,
            "model": model.key,
            "format": fmt,
            "quant": quant,
            "repo": repo,
            "status": "queued",
            "done": 0,
            "total": 0,
            "pct": 0.0,
            "error": None,
            "path": None,
            "started_at": time.time(),
        }
    return key


def _make_tqdm(key: str):
    """Build a tqdm subclass that reports into the job table (best effort)."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None

    class TrackingTqdm(tqdm):
        def update(self, n=1):
            _update(key, done=self.n, total=self.total or 0,
                    pct=round(100 * self.n / self.total, 1) if self.total else 0.0,
                    status_line=str(self))
            return super().update(n)

    return TrackingTqdm


def _pick_gguf_files(repo: str, quant: str | None) -> list[str]:
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    if not files:
        raise DownloadError(f"no .gguf files in {repo}")
    if quant:
        matches = [f for f in files if quant.lower() in f.lower()]
        files = matches or files
    return files


def _download_sync(key: str, model: ModelEntry, fmt: str, quant: str | None) -> str:
    from huggingface_hub import snapshot_download

    repo = model.repo(fmt)
    assert repo is not None
    tqdm_class = _make_tqdm(key)
    kwargs: dict[str, Any] = {}
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class

    _update(key, status="downloading")

    if fmt == "gguf":
        files = _pick_gguf_files(repo, quant)
        # snapshot the chosen quant file(s) + small metadata, skip other quants
        allow = list(files) + ["*.json", "*.txt", "*.md", "*.jinja"]
        try:
            path = snapshot_download(repo, allow_patterns=allow, **kwargs)
        except TypeError:  # older huggingface_hub without tqdm_class
            path = snapshot_download(repo, allow_patterns=allow)
        return path

    try:
        path = snapshot_download(repo, **kwargs)
    except TypeError:
        path = snapshot_download(repo)
    return path


async def run_download(key: str, model: ModelEntry, fmt: str, quant: str | None) -> None:
    def _task() -> str:
        return _download_sync(key, model, fmt, quant)

    try:
        path = await asyncio.to_thread(_task)
        _update(key, status="done", pct=100.0, path=path)
    except Exception as exc:
        _update(key, status="error", error=str(exc)[:500])


def submit_and_schedule(model: ModelEntry, fmt: str, quant: str | None) -> str:
    key = submit(model, fmt, quant)
    asyncio.get_running_loop().create_task(run_download(key, model, fmt, quant))
    return key
