"""Queue item 6: /api/decide auto-starts the engine once, then decides.

Mirrors the MCP server behavior (README's "booted on demand") for the HTTP
path. When auto_backend is false or start fails, the caller still gets a
503 with the actionable reason.
"""

import asyncio
from types import SimpleNamespace

from basal_station.api.routes import _ground_and_decide
from basal_station.backends.manager import EngineError
from basal_station.config import AppSettings, EngineConfig, KnowledgeConfig


class FakeEngine:
    def __init__(self, healthy: bool):
        self.healthy_state = healthy
        self.starts: list[tuple] = []
        self.decides: list[dict] = []

    async def healthy(self):
        return self.healthy_state

    async def start(self, backend_key, model_key, quant=None, extra_args=None):
        self.starts.append((backend_key, model_key))
        self.healthy_state = True
        return {"backend": backend_key, "model": model_key}

    async def decide(self, payload):
        if not self.healthy_state:
            raise EngineError("engine is not running")
        self.decides.append(payload)
        return {"answers": {}}


def _state(engine: FakeEngine, auto_backend: bool = True):
    return SimpleNamespace(
        settings=AppSettings(
            engine=EngineConfig(host="127.0.0.1", port=8021),
            knowledge=KnowledgeConfig(enabled=False),
            auto_backend=auto_backend,
        ),
        knowledge=None,
        engine=engine,
    )


def test_decide_autostarts_unhealthy_engine():
    engine = FakeEngine(healthy=False)
    result, info = asyncio.run(_ground_and_decide(
        _state(engine), "state text", {"q": {"type": "noul"}}
    ))
    assert engine.starts, "unhealthy engine must be auto-started"
    assert engine.decides, "decision must proceed after auto-start"
    assert result["answers"] == {}
    assert info == {"used": False}


def test_decide_skips_autostart_when_healthy():
    engine = FakeEngine(healthy=True)
    asyncio.run(_ground_and_decide(_state(engine), "s", {"q": {"type": "noul"}}))
    assert engine.starts == []
    assert len(engine.decides) == 1


def test_decide_respects_auto_backend_off():
    engine = FakeEngine(healthy=False)

    async def run():
        return await _ground_and_decide(
            _state(engine, auto_backend=False), "s", {"q": {"type": "noul"}}
        )

    try:
        asyncio.run(run())
        raised = False
    except Exception as exc:  # HTTPException from routes
        raised = True
        assert "engine is not running" in str(exc)
    assert raised
    assert engine.starts == []


def test_decide_surfaces_start_failure_reason():
    engine = FakeEngine(healthy=False)

    async def failing_start(backend_key, model_key, quant=None, extra_args=None):
        raise EngineError("no eligible backend on this machine for the default model")

    engine.start = failing_start

    async def run():
        return await _ground_and_decide(_state(engine), "s", {"q": {"type": "noul"}})

    try:
        asyncio.run(run())
        detail = ""
    except Exception as exc:
        detail = str(exc)
    assert "no eligible backend" in detail
