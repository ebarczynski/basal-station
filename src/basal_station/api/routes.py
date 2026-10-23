"""HTTP API routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from .. import __version__
from ..backends import EngineError
from ..backends.registry import (
    plan_gguf_offload,
    rank_backends,
    rank_models_for_hardware,
)
from ..catalog import format_size_bytes, get_model, list_models
from ..hardware import detect
from ..knowledge import KnowledgeBaseUnavailable, auto_query, ground_state, hits_summary
from . import downloads
from .localrepo import cached_repo_ids, local_formats, model_payload
from .schemas import (
    AskRequest,
    ContextRequest,
    DecideRequest,
    DownloadRequest,
    EngineStartRequest,
    SearchRequest,
    SimilarRequest,
)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------- system

@router.get("/system")
async def system(request: Request):
    state = request.app.state
    hw = detect()
    knowledge_status = await state.knowledge.status()
    engine = state.engine
    suggested = rank_models_for_hardware(hw)
    return {
        "app": "basal-station",
        "version": __version__,
        "hardware": hw.summary(),
        "engine": engine.status(),
        "engine_healthy": await engine.healthy(),
        "knowledge": knowledge_status,
        "suggested_models": suggested,
        "default_model": state.settings.default_model,
        "auto_backend": state.settings.auto_backend,
    }


# ---------------------------------------------------------------- models

@router.get("/models")
async def models(refresh: bool = False):
    if refresh:
        cached_repo_ids.cache_clear()
    return [model_payload(m) for m in list_models()]


@router.post("/models/download")
async def download_model(body: DownloadRequest):
    model = get_model(body.model)
    if model is None:
        raise HTTPException(404, f"unknown model '{body.model}'")
    if body.format not in model.repos:
        raise HTTPException(400, f"model '{body.model}' has no {body.format} repo")
    try:
        key = downloads.submit_and_schedule(model, body.format, body.quant)
    except downloads.DownloadError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job": key, "status": "queued"}


@router.get("/models/downloads")
async def download_status():
    return downloads.jobs_status()


# ---------------------------------------------------------------- backends

@router.get("/backends")
async def backends(request: Request, model: str = "", prefer_media: bool = False):
    state = request.app.state
    m = get_model(model or state.settings.default_model)
    if m is None:
        raise HTTPException(404, f"unknown model '{model}'")
    hw = detect()
    ranked = rank_backends(
        hw, m, prefer_media=prefer_media,
        local_formats=local_formats(m),
        gguf_ngl_override=state.settings.engine.gguf_ngl,
    )
    rec = next((r for r in ranked if r.eligible), None)
    return {
        "model": m.key,
        "recommended": rec.profile.key if rec else None,
        "backends": [r.as_dict() for r in ranked],
    }


# ---------------------------------------------------------------- engine

@router.post("/engine/start")
async def engine_start(request: Request, body: EngineStartRequest):
    state = request.app.state
    try:
        return await state.engine.start(
            body.backend, body.model, quant=body.quant, extra_args=body.extra_args
        )
    except EngineError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/engine/stop")
async def engine_stop(request: Request):
    return await request.app.state.engine.stop()


@router.get("/engine/status")
async def engine_status(request: Request):
    engine = request.app.state.engine
    return {**engine.status(), "healthy": await engine.healthy()}


@router.get("/engine/logs")
async def engine_logs(request: Request, n: int = 120):
    return {"lines": request.app.state.engine.recent_logs(n)}


# ---------------------------------------------------------------- decide

@router.post("/decide")
async def decide(request: Request, body: DecideRequest):
    state = request.app.state
    settings = state.settings
    raw_questions = body.model_dump()["questions"]
    payload: dict = {
        "state": body.state,
        "questions": {
            name: {k: v for k, v in q.items() if v not in (None, {}, "")}
            for name, q in raw_questions.items()
        },
    }
    payload.update(body.model_dump(exclude_none=True, exclude={"state", "questions", "knowledge"}))

    knowledge_info: dict = {"used": False}
    if body.knowledge and body.knowledge.enabled and settings.knowledge.enabled:
        if not state.knowledge.available():
            knowledge_info = {"used": False, "error": "knowledge base not connected"}
        else:
            t0 = time.monotonic()
            query = body.knowledge.query or auto_query(body.state, raw_questions)
            try:
                hits = await state.knowledge.context(query, body.knowledge.top_k)
            except Exception as exc:
                knowledge_info = {"used": False, "error": str(exc)[:300]}
            else:
                grounded, included = ground_state(body.state, hits, query)
                payload["state"] = grounded
                knowledge_info = {
                    "used": True,
                    "query": query,
                    "hits": hits_summary(hits),
                    "included": len(included),
                    "latency_ms": round(1000 * (time.monotonic() - t0)),
                }

    try:
        result = await state.engine.decide(payload)
    except EngineError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {**result, "knowledge": knowledge_info,
            "grounded_state": payload["state"] if knowledge_info.get("used") else None}


@router.post("/decide/raw")
async def decide_raw(request: Request, payload: dict):
    """Passthrough straight to the engine's /v1/systemone."""
    try:
        return await request.app.state.engine.decide(payload)
    except EngineError as exc:
        raise HTTPException(503, str(exc)) from exc


# ---------------------------------------------------------------- knowledge

@router.get("/knowledge/status")
async def knowledge_status(request: Request):
    return await request.app.state.knowledge.status()


@router.post("/knowledge/search")
async def knowledge_search(request: Request, body: SearchRequest):
    try:
        return await request.app.state.knowledge.search(body.query, body.limit, body.mode)
    except KnowledgeBaseUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/knowledge/ask")
async def knowledge_ask(request: Request, body: AskRequest):
    try:
        return await request.app.state.knowledge.ask(body.question, body.limit)
    except KnowledgeBaseUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/knowledge/context")
async def knowledge_context(request: Request, body: ContextRequest):
    try:
        return await request.app.state.knowledge.context(body.query, body.limit)
    except KnowledgeBaseUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/knowledge/similar")
async def knowledge_similar(request: Request, body: SimilarRequest):
    try:
        return await request.app.state.knowledge.similar(body.text, body.limit)
    except KnowledgeBaseUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


# ---------------------------------------------------------------- planning helper

@router.get("/plan")
async def plan(request: Request, model: str, quant: str = "Q4_K_M"):
    """Pure VRAM plan for a model+quant — no engine needed."""
    m = get_model(model)
    if m is None:
        raise HTTPException(404, f"unknown model '{model}'")
    hw = detect()
    size = format_size_bytes(m, "gguf", quant)
    cuda = hw.cuda_gpu
    ngl = plan_gguf_offload(
        size or 0, m.n_layers or 32, cuda.vram_bytes if cuda else 0
    ) if size else 0
    return {
        "model": m.key,
        "quant": quant,
        "size_gb": round(size / 1024**3, 1) if size else None,
        "vram_gb": round(cuda.vram_bytes / 1024**3, 1) if cuda else 0,
        "ngl": ngl,
        "layers": m.n_layers,
        "offload_pct": round(100 * ngl / (m.n_layers or 32)) if m.n_layers else None,
    }
