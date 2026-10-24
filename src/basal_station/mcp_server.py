"""MCP server exposing basal-station to MCP clients (e.g. ZCode).

Run with the project venv interpreter:
    python -m basal_station.mcp_server

Tools:
    decide          — typed decision, optionally grounded in knowledge-palace
    knowledge_search / knowledge_ask — query the knowledge base directly
    engine_status / engine_start    — inspect or boot the basal-serve engine

The first `decide` (or `engine_start`) launches basal-serve with the best
backend for this machine and the configured default model; that takes up to
a couple of minutes on cold start. Later calls hit the warm engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from .backends import EngineError
from .backends.registry import rank_backends
from .catalog import get_model
from .config import load_settings
from .hardware import detect
from .knowledge import KnowledgeService, auto_query, ground_state, hits_summary

log = logging.getLogger(__name__)

server = MCPServer(
    "basal-station",
    instructions=(
        "Typed-decision engine: instead of generating prose, a 12B model returns "
        "calibrated probabilities over answer options you define. Use `decide` for "
        "routing / yes-no / scoring / classification questions over a state "
        "(message, document, transcript). Set ground_with_knowledge=true to retrieve "
        "supporting passages from the user's knowledge-palace first."
    ),
)

_settings = load_settings()
_engine = None
_knowledge: KnowledgeService | None = None
_lock = asyncio.Lock()


def _engine_manager():
    global _engine
    if _engine is None:
        from .backends import EngineManager

        _engine = EngineManager(_settings)
    return _engine


async def _knowledge_service() -> KnowledgeService:
    global _knowledge
    if _knowledge is None:
        _knowledge = KnowledgeService(_settings.knowledge)
        await _knowledge.start()
    return _knowledge


async def _ensure_engine() -> None:
    manager = _engine_manager()
    from .backends.manager import ensure_running

    await ensure_running(manager, _settings)


def _clean_questions(questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {k: v for k, v in spec.items() if v not in (None, {}, "")}
        for name, spec in questions.items()
    }


@server.tool(
    description=(
        "Ask the basal typed-decision model. Provide a state (the message/document/"
        "transcript to judge) and questions like "
        '{"routing": {"type": "choice", "instructions": "...", "criteria": '
        '{"edd_team": "enhanced due diligence", "auto_clear": "false positive"}}, '
        '"needs_review": {"type": "noul", "instructions": "..."}}. '
        "Types: choice, noul (yes/no), score, multi, act, tool. Returns per-question "
        "probabilities, confidence and the chosen option."
    )
)
async def decide(
    state: str,
    questions: dict[str, dict[str, Any]],
    ground_with_knowledge: bool = True,
    top_k: int = 5,
    knowledge_query: str | None = None,
    auto_start_engine: bool = True,
) -> str:
    if not state.strip():
        return json.dumps({"error": "state must not be empty"})
    if not questions:
        return json.dumps({"error": "provide at least one question"})

    payload: dict[str, Any] = {"state": state, "questions": _clean_questions(questions)}

    knowledge_info: dict[str, Any] = {"used": False}
    if ground_with_knowledge and _settings.knowledge.enabled:
        try:
            kb = await _knowledge_service()
            query = knowledge_query or auto_query(state, questions)
            hits = await kb.context(query, top_k)
            grounded, included = ground_state(state, hits, query)
            payload["state"] = grounded
            knowledge_info = {"used": True, "query": query, "included": len(included),
                              "hits": hits_summary(hits)}
        except Exception as exc:
            knowledge_info = {"used": False, "error": str(exc)[:300]}

    manager = _engine_manager()
    async with _lock:
        try:
            if auto_start_engine:
                await _ensure_engine()
            result = await manager.decide(payload)
        except EngineError as exc:
            return json.dumps({"error": str(exc)[:500], "knowledge": knowledge_info})

    return json.dumps(
        {**result, "knowledge": knowledge_info},
        ensure_ascii=False,
        default=str,
    )


@server.tool(description="Search the knowledge-palace knowledge base.")
async def knowledge_search(query: str, limit: int = 5) -> str:
    kb = await _knowledge_service()
    return json.dumps(await kb.search(query, limit), ensure_ascii=False, default=str)


@server.tool(description="Ask the knowledge-palace knowledge base a question.")
async def knowledge_ask(question: str, limit: int = 5) -> str:
    kb = await _knowledge_service()
    return json.dumps(await kb.ask(question, limit), ensure_ascii=False, default=str)


@server.tool(description="Show basal engine status (backend, model, uptime).")
async def engine_status() -> str:
    manager = _engine_manager()
    status = manager.status()
    status["healthy"] = await manager.healthy()
    return json.dumps(status, ensure_ascii=False, default=str)


@server.tool(
    description=(
        "Start the basal-serve engine. Defaults to the configured default model and "
        "the best backend for this machine. Cold start loads ~13 GB of weights: "
        "expect up to two minutes before the first call answers."
    )
)
async def engine_start(model: str | None = None, backend: str | None = None) -> str:
    model_key = model or _settings.default_model
    if get_model(model_key) is None:
        return json.dumps({"error": f"unknown model '{model_key}'"})
    manager = _engine_manager()
    async with _lock:
        try:
            if backend is None:
                entry = get_model(model_key)
                assert entry is not None
                best = next(
                    (r for r in rank_backends(detect(), entry) if r.eligible), None
                )
                if best is None:
                    return json.dumps({"error": "no eligible backend on this machine"})
                backend = best.profile.key
            status = await manager.start(backend, model_key)
        except EngineError as exc:
            return json.dumps({"error": str(exc)[:500]})
    return json.dumps({"started": True, **{k: status[k] for k in
                     ("backend", "model", "pid")}}, ensure_ascii=False)


def main() -> int:
    # Logs must go to stderr; stdout carries the MCP stdio transport.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
