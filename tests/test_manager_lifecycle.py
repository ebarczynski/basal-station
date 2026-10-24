"""ADR-0003 regression: adoption reports only what the engine verifies.

Used to stamp the REQUESTED backend/model into status the moment anything
answered /health — "mlx / omni-1.0-max running" on a Linux box against an
arbitrary server. Now backend/mode stay unknown and model comes from the
engine's own /v1/models, with the request preserved under info.requested.
"""

import asyncio

from basal_station.backends.manager import EngineManager, ensure_running
from basal_station.config import AppSettings, EngineConfig, KnowledgeConfig


def _manager() -> EngineManager:
    settings = AppSettings(
        engine=EngineConfig(host="127.0.0.1", port=8021, startup_timeout=3),
        knowledge=KnowledgeConfig(enabled=False),
    )
    return EngineManager(settings)


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    def __init__(self, responses: dict[str, _FakeResp]):
        self._responses = responses

    def get(self, url):
        return _async(self._responses.get("health", _FakeResp(200)) if url.endswith("/health")
                      else self._responses.get("models", _FakeResp(404)))


async def _async(value):
    return value


def test_adoption_unverified_reports_nothing_as_fact():
    mgr = _manager()
    mgr._client = _FakeClient({"health": _FakeResp(200), "models": _FakeResp(404)})

    async def run():
        return await mgr.start(backend_key="mlx", model_key="omni-1.0-max")

    st = asyncio.run(run())
    assert st["running"] is True
    assert st["backend"] is None          # requested backend is NOT reported
    assert st["model"] is None            # requested model is NOT reported
    assert st["pid"] is None
    assert st["info"]["adopted"] is True
    assert st["info"]["verified"] is False
    assert st["info"]["requested"] == {"backend": "mlx", "model": "omni-1.0-max"}


def test_adoption_reports_engine_identity_when_verified():
    mgr = _manager()
    mgr._client = _FakeClient({
        "health": _FakeResp(200),
        "models": _FakeResp(200, {"model": "Remek/basal-1.5-4.5B", "mode": "fast"}),
    })

    async def run():
        return await mgr.start(backend_key="mlx", model_key="omni-1.0-max")

    st = asyncio.run(run())
    assert st["model"] == "Remek/basal-1.5-4.5B"
    assert st["mode"] == "fast"
    assert st["backend"] is None
    assert st["info"]["verified"] is True


def test_external_identity_parses_shape_variants():
    cases = [
        ({"model": "m1"}, {"model": "m1", "mode": None}),
        ({"model_id": "m2"}, {"model": "m2", "mode": None}),
        ({"models": [{"id": "m3"}]}, {"model": "m3", "mode": None}),
        ({"models": ["m4"]}, {"model": "m4", "mode": None}),
        # the real basal-serve shape (measured): name + per-model mode
        (
            {"models": [{"name": "basal-omni-1.0-max-MLX", "mode": "mlx"}]},
            {"model": "basal-omni-1.0-max-MLX", "mode": "mlx"},
        ),
        # a 200 with no usable identity must be unverified, not "verified: None"
        ({"models": []}, {}),
        ({"models": [{"release_date": "2026-10-22"}]}, {}),
        ([1, 2], {}),
        (None, {}),
    ]
    for payload, expected in cases:
        mgr = _manager()
        mgr._client = _FakeClient({"models": _FakeResp(200, payload)})

        async def run(mgr=mgr):
            return await mgr._external_identity()

        assert asyncio.run(run()) == expected, payload


def test_external_identity_bad_status_or_error_is_unverified():
    mgr = _manager()
    mgr._client = _FakeClient({"models": _FakeResp(500)})

    async def run():
        return await mgr._external_identity()

    assert asyncio.run(run()) == {}

    mgr = _manager()
    mgr._client = _FakeClient({"models": _FakeResp(200, RuntimeError("boom"))})
    assert asyncio.run(run()) == {}


def test_adoption_mismatch_gets_warning():
    from basal_station.backends.manager import _compare_identity
    from basal_station.catalog import get_model

    model = get_model("omni-1.0-max")
    # the real engine name for this model - token match, no warning
    verified, mismatch = _compare_identity({"model": "basal-omni-1.0-max-MLX"}, model)
    assert verified and not mismatch
    # a different model on the port - flagged
    verified, mismatch = _compare_identity({"model": "basal-1.5-mini"}, model)
    assert verified and mismatch
    # unknown identity - never a mismatch (only silence)
    verified, mismatch = _compare_identity({}, model)
    assert not verified and not mismatch


def test_ensure_running_skips_when_healthy(monkeypatch):
    mgr = _manager()
    started = []

    async def healthy():
        return True

    async def start(*a, **kw):
        started.append((a, kw))
        return {}

    monkeypatch.setattr(mgr, "healthy", healthy)
    monkeypatch.setattr(mgr, "start", start)
    settings = AppSettings(knowledge=KnowledgeConfig(enabled=False))
    asyncio.run(ensure_running(mgr, settings))
    assert started == []


def test_ensure_running_starts_best_backend(monkeypatch):
    mgr = _manager()
    started = []

    async def healthy():
        return False

    async def start(backend_key, model_key, quant=None, extra_args=None):
        started.append((backend_key, model_key))
        return {"backend": backend_key}

    monkeypatch.setattr(mgr, "healthy", healthy)
    monkeypatch.setattr(mgr, "start", start)
    settings = AppSettings(knowledge=KnowledgeConfig(enabled=False))
    status = asyncio.run(ensure_running(mgr, settings))
    assert started, "engine must be started when nothing is healthy"
    backend, model = started[0]
    assert model == settings.default_model
    assert backend  # best eligible backend for this machine (gguf-cpu on a bare Linux box)
    assert status["backend"] == backend


def test_ensure_running_raises_when_no_backend(monkeypatch):
    import pytest

    from basal_station.backends.manager import EngineError

    mgr = _manager()

    async def healthy():
        return False

    monkeypatch.setattr(mgr, "healthy", healthy)
    settings = AppSettings(knowledge=KnowledgeConfig(enabled=False),
                           default_model="no-such-model")
    with pytest.raises(EngineError):
        asyncio.run(ensure_running(mgr, settings))
