"""Synchronous MCP client for the document server.

The agent (LangGraph nodes), Streamlit and the CLI are synchronous, while the MCP SDK is
async. This client runs one long-lived stdio session on a background event loop and
exposes blocking, Pydantic-typed methods. One server subprocess serves many calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import Future
from typing import Any, TypeVar

from mcp import Client, StdioServerParameters
from pydantic import BaseModel

from docqa.config import Settings
from docqa.models import DeleteResult, DocumentInfo, DocumentList, DocumentText, IngestResult, SearchResult

T = TypeVar("T", bound=BaseModel)


class MCPToolError(RuntimeError):
    pass


class MCPDocumentClient:
    def __init__(self, settings: Settings, timeout: float = 300.0):
        self.settings = settings
        self.timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Client | None = None
        self._stop: asyncio.Event | None = None
        self._runner: Future | None = None

    # ------------------------------------------------------------------ lifecycle

    def server_params(self) -> StdioServerParameters:
        # The server subprocess reads the same DOCQA_* settings as this process.
        env = dict(os.environ)
        env.update(
            {
                "DOCQA_LLM_BACKEND": self.settings.llm_backend,
                "DOCQA_OLLAMA_HOST": self.settings.ollama_host,
                "DOCQA_EMBED_MODEL": self.settings.embed_model,
                "DOCQA_DATA_DIR": str(self.settings.data_dir),
                "DOCQA_CHUNK_SIZE": str(self.settings.chunk_size),
                "DOCQA_CHUNK_OVERLAP": str(self.settings.chunk_overlap),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        return StdioServerParameters(command=sys.executable, args=["-m", "docqa.mcp_server"], env=env)

    def start(self) -> "MCPDocumentClient":
        if self._session is not None:
            return self
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        ready: Future = Future()
        self._runner = asyncio.run_coroutine_threadsafe(self._run(ready), self._loop)
        ready.result(timeout=60)
        return self

    async def _run(self, ready: Future) -> None:
        self._stop = asyncio.Event()
        try:
            async with Client(self.server_params(), read_timeout_seconds=self.timeout) as client:
                self._session = client
                ready.set_result(True)
                await self._stop.wait()
        except BaseException as exc:  # surface startup failures to start()
            if not ready.done():
                ready.set_exception(exc)
            else:
                raise
        finally:
            self._session = None

    def close(self) -> None:
        if self._loop is None:
            return
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._runner is not None:
            try:
                self._runner.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()
        self._loop = None

    def __enter__(self) -> "MCPDocumentClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ calls

    def _submit(self, coro) -> Any:
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP client is not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=self.timeout)

    def list_tools(self) -> list[dict]:
        result = self._submit(self._session.list_tools())
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in result.tools
        ]

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        result = self._submit(self._session.call_tool(name, arguments or {}))
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            raise MCPToolError(f"{name} failed: {text}")
        if result.structured_content is not None:
            return result.structured_content
        return json.loads(text)

    def _typed(self, model: type[T], name: str, arguments: dict | None = None) -> T:
        return model.model_validate(self.call_tool(name, arguments))

    # Typed wrappers: MCP results are validated with the same Pydantic models the server uses.

    def ingest(self, path: str) -> IngestResult:
        return self._typed(IngestResult, "ingest_document", {"path": str(path)})

    def list_documents(self) -> list[DocumentInfo]:
        return self._typed(DocumentList, "list_documents").documents

    def search(self, query: str, top_k: int) -> SearchResult:
        return self._typed(SearchResult, "search_documents", {"query": query, "top_k": top_k})

    def read_document(self, doc_id: str, max_chars: int = 4000) -> DocumentText:
        return self._typed(DocumentText, "read_document", {"doc_id": doc_id, "max_chars": max_chars})

    def delete_document(self, doc_id: str) -> DeleteResult:
        return self._typed(DeleteResult, "delete_document", {"doc_id": doc_id})
