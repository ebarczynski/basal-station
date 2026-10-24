"""Pydantic schemas mirroring the basal /v1/systemone contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionType = Literal["choice", "noul", "score", "multi", "act", "tool"]


class QuestionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: QuestionType = "choice"
    instructions: str = ""
    criteria: dict[str, str] = Field(default_factory=dict)
    max_error: float | None = None


class KnowledgeOptions(BaseModel):
    enabled: bool = True
    query: str | None = None       # explicit retrieval query; auto-derived when None
    top_k: int | None = None


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    state: str
    questions: dict[str, QuestionSpec] = Field(min_length=1)
    knowledge: KnowledgeOptions | None = None
    policy: dict[str, Any] | None = None  # applied by basal-station, not the engine
    media: list[dict[str, Any]] | None = None  # eager backend only


class EvalExample(BaseModel):
    state: str
    labels: dict[str, Any]
    knowledge: KnowledgeOptions | None = None


class EvalRequest(BaseModel):
    questions: dict[str, QuestionSpec] = Field(min_length=1)
    examples: list[EvalExample] = Field(min_length=1)
    threshold: float = 0.5


class DownloadRequest(BaseModel):
    model: str
    format: Literal["gguf", "mlx", "fp8", "safetensors"]
    quant: str | None = None       # gguf only, e.g. Q4_K_M


class EngineStartRequest(BaseModel):
    backend: str
    model: str
    quant: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    limit: int | None = None
    mode: str | None = None


class AskRequest(BaseModel):
    question: str
    limit: int | None = None


class ContextRequest(BaseModel):
    query: str
    limit: int | None = None


class SimilarRequest(BaseModel):
    text: str
    limit: int = 5
