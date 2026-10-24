"""ADR-0001 regression: grounding without a verifiable `context` tool fails visibly.

The real knowledge-palace (mcp 1.x server) exposes search/similar/ask only and
answers unknown tools with plain text — which used to be ingested as retrieval
hits, poisoning every grounded decision state with "Unknown tool: context"
while the response reported used: true. These tests pin the fix.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from basal_station.config import KnowledgeConfig
from basal_station.knowledge import KnowledgeBaseUnavailable, KnowledgeService
from basal_station.knowledge.enrich import ground_state

STUB = str(Path(__file__).parent / "stub_kp_server.py")
STATE = "PEP screening hit 71% fuzzy match on a business account."


def _service(*flags) -> KnowledgeService:
    cfg = KnowledgeConfig(
        enabled=True,
        command=[sys.executable, STUB, *flags],
        timeout_s=20.0,
    )
    return KnowledgeService(cfg)


class _FakeConn:
    def __init__(self, tools: list[str]):
        self._tools = tools
        self.calls: list[tuple] = []

    def needs_restart(self) -> bool:
        return False

    def error(self) -> str | None:
        return None

    async def list_tools(self) -> list[str]:
        return list(self._tools)

    async def call(self, tool, arguments, timeout=None):
        self.calls.append((tool, arguments))
        return {"results": [{"text": "policy says X", "source": "s", "score": 0.9}]}


def _service_with_conn(tools: list[str]) -> KnowledgeService:
    svc = KnowledgeService.__new__(KnowledgeService)  # test seam: skip MCP startup
    svc.config = KnowledgeConfig(enabled=True)  # noqa: SLF001
    svc._conn = _FakeConn(tools)  # noqa: SLF001
    return svc


def test_context_refused_without_tool():
    """The exact real-KP shape: search/ask/similar but no context."""
    svc = _service_with_conn(["search", "similar", "ask"])

    async def run():
        await svc.context("PEP screening", 5)

    with pytest.raises(KnowledgeBaseUnavailable) as exc:
        asyncio.run(run())
    assert "'context'" in str(exc.value)
    assert "ADR-0001" in str(exc.value)
    assert svc._conn.calls == []  # the tool was never invoked


def test_context_allowed_when_listed():
    svc = _service_with_conn(["search", "similar", "ask", "context"])

    async def run():
        return await svc.context("q", 5)

    hits = asyncio.run(run())
    assert hits == {"results": [{"text": "policy says X", "source": "s", "score": 0.9}]}
    assert svc._conn.calls[0][0] == "context"


def test_poisoning_regression_against_real_stub_protocol():
    """End to end over stdio MCP: no `context` tool -> refusal, never hits.

    Before the fix this returned the literal text 'Unknown tool: context' as a
    retrieval hit and ground_state appended it to the state as knowledge.
    """
    svc = _service()

    async def run():
        await svc.start()
        try:
            hits = await svc.context("PEP screening", 5)
            # if the guard ever regresses, prove the poisoning instead of lying:
            grounded, included = ground_state(STATE, hits, "PEP screening")
            assert "Unknown tool: context" not in grounded, (
                "grounding ingested an unknown-tool reply as knowledge"
            )
            assert not included
        finally:
            await svc.close()

    with pytest.raises(KnowledgeBaseUnavailable):
        asyncio.run(run())


def test_structured_context_round_trip():
    """With the tool present, grounding gets real structured hits."""
    svc = _service("--with-context")

    async def run():
        await svc.start()
        try:
            return await svc.context("PEP screening", 5)
        finally:
            await svc.close()

    hits = asyncio.run(run())
    grounded, included = ground_state(STATE, hits, "PEP screening")
    assert included, "structured hits must flow through"
    assert "EDD team" in grounded
    assert "Unknown tool" not in grounded


def test_route_level_grounding_failure_is_visible_and_ungrounded():
    """_ground_and_decide reports used:false + error and calls the engine ungrounded."""
    from basal_station.api.routes import _ground_and_decide
    from basal_station.api.schemas import KnowledgeOptions

    class ExplodingKnowledge:
        def available(self):
            return True

        async def context(self, query, limit):
            raise KnowledgeBaseUnavailable("knowledge-palace lacks the 'context' tool")

    class FakeEngine:
        def __init__(self):
            self.payloads = []

        async def decide(self, payload):
            self.payloads.append(payload)
            return {"answers": {}}

    state = SimpleNamespace(
        settings=SimpleNamespace(knowledge=KnowledgeConfig(enabled=True)),
        knowledge=ExplodingKnowledge(),
        engine=FakeEngine(),
    )
    result, info = asyncio.run(_ground_and_decide(
        state, STATE, {"routing": {"type": "choice", "instructions": "who?"}},
        KnowledgeOptions(enabled=True),
    ))
    assert info["used"] is False
    assert "context" in info["error"]
    # the engine still saw the RAW state — grounding failure must not edit it
    assert result["grounded_state"] is None
    assert state.engine.payloads[0]["state"] == STATE
    json.dumps(info)  # the audit trail must stay serializable
