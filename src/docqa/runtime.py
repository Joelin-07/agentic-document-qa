"""Wires settings -> LLM -> MCP client -> agent. Shared by the UI, CLI and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docqa.config import Settings
from docqa.graph import DocQAAgent
from docqa.llm import LLMClient, get_llm
from docqa.mcp_client import MCPDocumentClient
from docqa.models import IngestResult


@dataclass
class Runtime:
    settings: Settings
    llm: LLMClient
    mcp: MCPDocumentClient
    agent: DocQAAgent

    def ingest_folder(self, folder: Path) -> list[IngestResult]:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
        return [self.mcp.ingest(str(p)) for p in files]

    def close(self) -> None:
        self.mcp.close()
        self.agent.close()


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or Settings()
    settings.ensure_dirs()
    llm = get_llm(settings)
    mcp = MCPDocumentClient(settings).start()
    agent = DocQAAgent(settings, llm, docs=mcp)
    return Runtime(settings, llm, mcp, agent)
