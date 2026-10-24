"""EngineManager: lifecycle of the basal-serve subprocess + decision proxy."""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from datetime import UTC, datetime

import httpx

from ..config import AppSettings
from ..hardware import detect
from .registry import build_engine_args, rank_backends


class EngineError(RuntimeError):
    pass


def _compare_identity(ident: dict, model) -> tuple[bool, bool]:
    """(verified, mismatch) between what the engine reports and the request.

    Comparison runs on separator-stripped names ("basal-omni-1.0-max-MLX" →
    "basalomni10maxmlx") with substring containment both ways — token sets
    cannot tell 1.0 from 1.1. The HF repo id is excluded on purpose (the
    author prefix never appears in engine reports). Unknown identity counts
    as verified=False but never as a mismatch (ADR-0003: report only facts).
    """
    verified_model = ident.get("model")
    if not verified_model:
        return False, False
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    key, engine = norm(model.key), norm(str(verified_model))
    return True, key not in engine and engine not in key


class EngineManager:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._proc: asyncio.subprocess.Process | None = None
        self._pump_tasks: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._logs: deque[str] = deque(maxlen=500)
        self._state: dict = {
            "running": False,
            "backend": None,
            "model": None,
            "mode": None,
            "pid": None,
            "started_at": None,
            "command": None,
            "info": {},
            "last_error": None,
        }
        self._client = httpx.AsyncClient(timeout=120.0)

    # ---------- lifecycle ----------

    async def start(
        self,
        backend_key: str,
        model_key: str,
        quant: str | None = None,
        extra_args: list[str] | None = None,
    ) -> dict:
        from ..catalog import get_model
        from .base import get_backend

        model = get_model(model_key)
        profile = get_backend(backend_key)
        if model is None:
            raise EngineError(f"unknown model '{model_key}'")
        if profile is None:
            raise EngineError(f"unknown backend '{backend_key}'")

        async with self._lock:
            await self._stop_locked()
            hw = detect()

            # An engine may already be serving on the configured port (another
            # station/MCP/CLI client started it). Adopt it instead of spawning
            # a duplicate that would fight over the port — but only report
            # what the engine verifies about itself (ADR-0003).
            if await self._external_healthy():
                ident = await self._external_identity()
                verified, mismatch = _compare_identity(ident, model)
                info: dict = {
                    "adopted": True,
                    "verified": verified,
                    "requested": {"backend": backend_key, "model": model_key},
                }
                if mismatch:
                    info["warning"] = (
                        "adopted engine serves a different model than requested"
                    )
                self._state.update(
                    running=True,
                    backend=None,
                    model=ident.get("model"),
                    mode=ident.get("mode"),
                    pid=None,
                    started_at=datetime.now(UTC).isoformat(),
                    command=["(adopted engine already listening on port)"],
                    info=info,
                    last_error=None,
                )
                self._log(
                    f"adopted engine already serving on {self.settings.engine.base_url}: "
                    + (
                        f"model={ident.get('model')} mode={ident.get('mode')} (verified)"
                        if verified
                        else "identity unverified (/v1/models gave nothing usable)"
                    )
                    + (f" — WARNING: {info['warning']}" if mismatch else "")
                )
                return self.status()

            ranked_list = rank_backends(
                hw,
                model,
                gguf_ngl_override=self.settings.engine.gguf_ngl,
            )
            ranked = next((r for r in ranked_list if r.profile.key == backend_key), None)
            if ranked is None or not ranked.eligible:
                reason = ranked.reason if ranked else "backend unavailable on this machine"
                raise EngineError(f"backend '{backend_key}' not usable here: {reason}")
            if quant:
                ranked.quant = quant

            args, info = build_engine_args(
                hw, model, profile, ranked,
                engine_host=self.settings.engine.host,
                engine_port=self.settings.engine.port,
                extra_args=list(self.settings.engine.extra_args) + list(extra_args or []),
            )

            self._log(f"$ {' '.join(args)}")
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except FileNotFoundError as exc:
                raise EngineError(
                    f"cannot launch '{args[0]}': {exc}. Install the basal engine "
                    "(see scripts/setup-*.sh or README)."
                ) from exc

            self._pump_tasks.append(asyncio.create_task(self._pump(self._proc)))
            self._state.update(
                running=True,
                backend=backend_key,
                model=model_key,
                mode=profile.mode,
                pid=self._proc.pid,
                started_at=datetime.now(UTC).isoformat(),
                command=args,
                info=info,
                last_error=None,
            )

            ok = await self._wait_health(self.settings.engine.startup_timeout)
            if not ok:
                tail = "\n".join(list(self._logs)[-15:])
                await self._stop_locked()
                self._state["last_error"] = "engine did not become healthy in time"
                raise EngineError(
                    f"engine did not become healthy within "
                    f"{self.settings.engine.startup_timeout:.0f}s.\nLast logs:\n{tail}"
                )
            self._log("engine healthy")
            return self.status()

    async def _external_healthy(self) -> bool:
        """True when *something* answers /health on the engine port."""
        try:
            resp = await self._client.get(f"{self.settings.engine.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _external_identity(self) -> dict:
        """Best-effort parse of the engine's /v1/models (ADR-0003).

        The real engine answers {"models": [{"name": ..., "mode": ...}]};
        generic OpenAI-ish shapes are tolerated. Returns {} when nothing can
        be extracted — the adopted state then stays "unverified" instead of
        echoing the request or claiming a verification that found nothing.
        """
        try:
            resp = await self._client.get(
                f"{self.settings.engine.base_url}/v1/models"
            )
            if resp.status_code != 200:
                return {}
            data = resp.json()
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        model = data.get("model") or data.get("model_id") or data.get("id")
        mode = data.get("mode")
        models = data.get("models")
        if not model and isinstance(models, list) and models:
            first = models[0]
            if isinstance(first, dict):
                model = first.get("id") or first.get("model") or first.get("name")
                mode = mode or first.get("mode")
            elif isinstance(first, str):
                model = first
        if not model:
            return {}
        return {"model": model, "mode": mode}

    async def stop(self) -> dict:
        async with self._lock:
            await self._stop_locked()
            return self.status()

    async def _stop_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            if self._state.get("info", {}).get("adopted"):
                self._log("detached from adopted engine (it keeps serving on its port)")
            self._state.update(running=False, pid=None)
            return
        self._log("stopping engine…")
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (TimeoutError, ProcessLookupError):
            proc.kill()
            await proc.wait()
        for t in self._pump_tasks:
            t.cancel()
        self._pump_tasks.clear()
        self._state.update(running=False, pid=None)
        self._log("engine stopped")

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        stdout = proc.stdout
        if stdout is None:
            return
        while True:
            line = await stdout.readline()
            if not line:
                break
            self._log(line.decode(errors="replace").rstrip())
        if self._proc is proc and proc.returncode not in (0, None):
            self._state["running"] = False
            self._state["last_error"] = f"engine exited with code {proc.returncode}"
            self._log(f"[engine exited with code {proc.returncode}]")

    async def _wait_health(self, timeout: float) -> bool:
        url = f"{self.settings.engine.base_url}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.returncode is not None:
                # Our child died — most likely the port was claimed by a
                # concurrent starter (station + MCP + CLI racing). Succeed if
                # another engine ends up serving, otherwise fail fast. The
                # adopted engine's identity is whatever it verifies (ADR-0003).
                if await self._external_healthy():
                    ident = await self._external_identity()
                    self._state.update(
                        pid=None,
                        model=ident.get("model"),
                        mode=ident.get("mode"),
                        info={
                            **self._state.get("info", {}),
                            "adopted": True,
                            "verified": bool(ident),
                        },
                    )
                    return True
                return False
            try:
                resp = await self._client.get(url)
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
        return False

    # ---------- decisions ----------

    async def healthy(self) -> bool:
        if not self._state["running"]:
            return False
        try:
            resp = await self._client.get(f"{self.settings.engine.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def decide(self, payload: dict) -> dict:
        # /health is the source of truth (covers own child and engines
        # started by the service, MCP or any other client on this port).
        if not await self._external_healthy():
            raise EngineError("engine is not running")
        resp = await self._client.post(
            f"{self.settings.engine.base_url}/v1/systemone", json=payload
        )
        if resp.status_code != 200:
            raise EngineError(f"engine returned {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    # ---------- introspection ----------

    def _log(self, line: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._logs.append(f"{stamp} {line}")

    def recent_logs(self, n: int = 100) -> list[str]:
        return list(self._logs)[-n:]

    def status(self) -> dict:
        return {**self._state, "engine_url": self.settings.engine.base_url}


def create_manager(settings: AppSettings) -> EngineManager:
    return EngineManager(settings)


async def ensure_running(manager: EngineManager, settings: AppSettings) -> dict:
    """Start the configured default model on the best eligible backend unless
    an engine is already healthy on the port. Raises EngineError with the real
    reason when nothing can start.

    Shared by the HTTP API (/api/decide auto-start) and the MCP server, so
    both entry paths behave the same way.
    """
    if await manager.healthy():
        return manager.status()
    from ..api.localrepo import local_formats  # function-level: avoids api<-backends cycle
    from ..catalog import get_model
    from ..hardware import detect

    model = get_model(settings.default_model)
    if model is None:
        raise EngineError(
            f"default_model '{settings.default_model}' is not in the catalog"
        )
    ranked = rank_backends(
        detect(), model,
        local_formats=local_formats(model),
        gguf_ngl_override=settings.engine.gguf_ngl,
    )
    best = next((r for r in ranked if r.eligible), None)
    if best is None:
        raise EngineError("no eligible backend on this machine for the default model")
    return await manager.start(best.profile.key, model.key)
