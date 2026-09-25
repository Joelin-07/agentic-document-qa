"""Pydantic models used at every boundary: MCP tools, LLM output, traces, evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- documents


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: Literal["pdf", "txt", "md"]
    source_path: str
    content_hash: str
    num_chunks: int = Field(ge=0)
    num_chars: int = Field(ge=0)
    ingested_at: datetime = Field(default_factory=utcnow)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    page: int | None = None


class RetrievedChunk(Chunk):
    score: float


# --------------------------------------------------------------------------- MCP tool I/O


class IngestResult(BaseModel):
    status: Literal["added", "updated", "unchanged"]
    document: DocumentInfo


class SearchResult(BaseModel):
    query: str
    results: list[RetrievedChunk]


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class DocumentText(BaseModel):
    doc_id: str
    filename: str
    text: str
    truncated: bool


class DeleteResult(BaseModel):
    doc_id: str
    deleted: bool


# --------------------------------------------------------------------------- LLM answers


class LLMAnswer(BaseModel):
    """Schema the LLM must fill in. Passed to Ollama as a JSON schema, then validated."""

    answer: str = Field(min_length=1, description="One or two complete sentences grounded in the context")
    found: bool = Field(description="True if the context contains the answer")
    citations: list[int] = Field(default_factory=list, description="Numbers of the context passages used")
    confidence: Literal["high", "medium", "low"] = "medium"


class Citation(BaseModel):
    filename: str
    chunk_id: str
    page: int | None = None
    score: float
    snippet: str


class Answer(BaseModel):
    """Final validated answer returned to the UI/CLI."""

    answer: str
    found: bool
    confidence: Literal["high", "medium", "low"]
    citations: list[Citation] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = wrong, 5 = fully correct")
    reason: str


# --------------------------------------------------------------------------- observability


class Span(BaseModel):
    node: str
    started_at: datetime
    duration_ms: float
    status: Literal["ok", "error"] = "ok"
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Trace(BaseModel):
    trace_id: str
    thread_id: str
    question: str
    started_at: datetime = Field(default_factory=utcnow)
    total_ms: float = 0.0
    status: Literal["ok", "error"] = "ok"
    route: list[str] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    answer_found: bool | None = None
    error: str | None = None


# --------------------------------------------------------------------------- evaluation


class EvalItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_source: str | None = None
    answerable: bool = True
    context: list[str] = Field(default_factory=list, description="Earlier questions asked in the same thread (tests memory)")


class EvalItemResult(BaseModel):
    id: str
    question: str
    answer: str
    expected_answer: str
    answerable: bool
    found: bool
    retrieval_hit: bool | None
    citation_hit: bool | None
    keyword_score: float
    judge_score: int | None = None
    correct: bool
    latency_ms: float
    trace_id: str


class EvalReport(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=utcnow)
    chat_model: str
    embed_model: str
    num_items: int
    accuracy: float
    retrieval_hit_rate: float
    citation_accuracy: float
    refusal_accuracy: float
    mean_keyword_score: float
    mean_judge_score: float | None
    avg_latency_ms: float
    results: list[EvalItemResult]
