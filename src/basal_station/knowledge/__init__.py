"""knowledge-palace MCP integration (RAG grounding for decisions)."""

from .client import KnowledgeBaseUnavailable, KnowledgeService
from .enrich import auto_query, format_hits_block, ground_state, hits_summary, state_query

__all__ = [
    "KnowledgeBaseUnavailable",
    "KnowledgeService",
    "auto_query",
    "format_hits_block",
    "ground_state",
    "hits_summary",
    "state_query",
]
