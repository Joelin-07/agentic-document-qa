"""Application settings, validated by Pydantic.

Values come from (in order of priority) environment variables prefixed with
``DOCQA_``, a local ``.env`` file, and the defaults below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCQA_", env_file=".env", extra="ignore")

    # LLM backend ("fake" is a deterministic stand-in used by tests; no Ollama needed)
    llm_backend: Literal["ollama", "fake"] = "ollama"
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    request_timeout: float = Field(120.0, gt=0)

    # Retrieval
    chunk_size: int = Field(800, ge=100, le=4000, description="Max characters per chunk")
    chunk_overlap: int = Field(120, ge=0, description="Characters carried over between chunks")
    top_k: int = Field(4, ge=1, le=20)
    min_score: float = Field(0.55, ge=-1.0, le=1.0, description="Cosine similarity cutoff for a relevant chunk")

    # Memory: how many previous messages are shown to the LLM
    history_messages: int = Field(6, ge=0, le=50)

    # Storage
    data_dir: Path = PROJECT_ROOT / "data"
    demo_docs_dir: Path = PROJECT_ROOT / "demo_data" / "docs"
    eval_dataset: Path = PROJECT_ROOT / "demo_data" / "eval" / "eval_set.jsonl"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.sqlite"

    @property
    def traces_file(self) -> Path:
        return self.data_dir / "traces.jsonl"

    @property
    def eval_reports_dir(self) -> Path:
        return self.data_dir / "eval_reports"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.index_dir, self.uploads_dir, self.eval_reports_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
