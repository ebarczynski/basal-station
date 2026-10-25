"""Turn knowledge-palace hits into a grounded decision state."""

from __future__ import annotations

import json
from typing import Any

MAX_BLOCK_CHARS = 6000


def auto_query(state: str, questions: dict[str, Any]) -> str:
    """Natural-language retrieval query: question instructions + the state head.

    Measured against multi-query retrieval: pipe-joined question names and
    option keys ("route | technical | account | Ticket …") return zero hits
    where an instruction sentence returns several. Instructions are natural
    language by design; bare keys add noise, so they are dropped. The state
    head anchors the query to the concrete case.
    """
    parts: list[str] = []
    for spec in list(questions.values())[:4]:
        if not isinstance(spec, dict):
            continue
        instructions = str(spec.get("instructions") or "").strip()
        if len(instructions) > 8:
            parts.append(instructions.rstrip("?.") + "?")
    state_head = " ".join((state or "").split())[:400]
    if state_head:
        parts.append(state_head)
    return " ".join(parts)[:800]


def state_query(state: str) -> str:
    """The state itself as a retrieval query (zero-hit retry path)."""
    return " ".join((state or "").split())[:400]


def _normalize_hits(payload: Any) -> list[dict[str, Any]]:
    """Accept list / {'results': [...]} / plain text shapes."""
    if payload is None:
        return []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("results", "hits", "passages", "chunks"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            items = [payload]
    else:
        return [{"text": str(payload)}]

    hits: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("chunk") or ""
            hit = {k: v for k, v in item.items() if k != "text" and v is not None}
            hit["text"] = str(text)
            hits.append(hit)
        else:
            hits.append({"text": str(item)})
    return hits


def format_hits_block(query: str, payload: Any) -> tuple[str, list[dict[str, Any]]]:
    hits = _normalize_hits(payload)
    if not hits:
        return "", []
    header = f"Reference knowledge (from personal knowledge base; retrieval query: {query!r})"
    lines = [header]
    used = len(header)  # the header counts toward the budget (long queries!)
    kept = 0
    for i, hit in enumerate(hits, 1):
        source = hit.get("source") or hit.get("title") or hit.get("author") or hit.get("id") or ""
        text = " ".join(str(hit.get("text", "")).split())
        if not text:
            continue
        prefix = f"[{i}] {f'({source}) ' if source else ''}"
        budget = MAX_BLOCK_CHARS - used - len(prefix) - 1
        if budget <= 0:
            break
        chunk = text[:budget]
        lines.append(f"{prefix}{chunk}")
        used += len(prefix) + len(chunk) + 1  # +1 = the joining newline
        kept += 1
        if used >= MAX_BLOCK_CHARS:
            break
    return "\n".join(lines), hits[:kept]


def ground_state(state: str, payload: Any, query: str) -> tuple[str, list[dict[str, Any]]]:
    """Append a reference-knowledge block to the raw state.

    Returns (grounded_state, hits_actually_included).
    """
    block, hits = format_hits_block(query, payload)
    if not block:
        return state, []
    grounded = state.rstrip() + "\n\n" + block
    return grounded, hits


def hits_summary(payload: Any, limit: int = 8) -> list[dict[str, Any]]:
    return _normalize_hits(payload)[:limit]


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
