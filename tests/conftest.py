"""Shared fixtures. Everything runs against the deterministic FakeLLM, so no Ollama is needed."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from docqa.config import PROJECT_ROOT, Settings
from docqa.document_service import DocumentService
from docqa.graph import DocQAAgent
from docqa.llm import FakeLLM
from docqa.observability import Tracer

DEMO_DOCS = PROJECT_ROOT / "demo_data" / "docs"
EVAL_SET = PROJECT_ROOT / "demo_data" / "eval" / "eval_set.jsonl"


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Hashed bag-of-words similarities are low, so the fake backend uses a low cutoff.
    return Settings(_env_file=None, llm_backend="fake", data_dir=tmp_path / "data", min_score=0.15)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(settings, fake_llm) -> DocumentService:
    svc = DocumentService(settings, fake_llm)
    for path in sorted(DEMO_DOCS.iterdir()):
        svc.ingest(path)
    return svc


@pytest.fixture
def agent(settings, fake_llm, service) -> DocQAAgent:
    """Agent wired to the in-process document service (MCP is covered in test_mcp.py / test_e2e.py)."""
    return DocQAAgent(settings, fake_llm, docs=service, checkpointer=InMemorySaver(), tracer=Tracer(settings.traces_file))
