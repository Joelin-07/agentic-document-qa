"""MCP server exposing the document tools over stdio.

Run standalone:            python -m docqa.mcp_server
Inspect with MCP Inspector: mcp dev src/docqa/mcp_server.py

Tool arguments are validated by Pydantic (via MCPServer, the SDK's high-level server) and results are returned as
typed structured content, which the client validates again with the same models.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from docqa.config import Settings
from docqa.document_service import DocumentService
from docqa.llm import get_llm
from docqa.models import DeleteResult, DocumentList, DocumentText, IngestResult, SearchResult

# stdout is the MCP transport, so logs must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[mcp-server] %(levelname)s %(message)s")
log = logging.getLogger("docqa.mcp_server")
logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = MCPServer(
    "docqa-documents",
    instructions="Tools to ingest local PDF/TXT/MD documents and search them semantically.",
)


@lru_cache
def service() -> DocumentService:
    settings = Settings()
    log.info("backend=%s index=%s", settings.llm_backend, settings.index_dir)
    return DocumentService(settings, get_llm(settings))


@contextmanager
def expected_errors():
    """Report user errors (bad path, unknown id, unsupported file) to the client as tool errors.

    MCPServer hides the message of any other exception, so real crashes stay generic.
    """
    try:
        yield
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ToolError(exc.args[0] if exc.args else repr(exc)) from exc


@mcp.tool()
def ingest_document(
    path: Annotated[str, Field(min_length=1, description="Absolute or relative path to a .pdf, .txt or .md file")],
) -> IngestResult:
    """Load, chunk and embed a document into the local index. Re-ingesting an unchanged file is a no-op."""
    with expected_errors():
        result = service().ingest(path)
    log.info("ingest %s -> %s (%d chunks)", result.document.filename, result.status, result.document.num_chunks)
    return result


@mcp.tool()
def list_documents() -> DocumentList:
    """List all indexed documents with their ids and chunk counts."""
    return service().list_documents()


@mcp.tool()
def search_documents(
    query: Annotated[str, Field(min_length=1, description="Natural-language search query")],
    top_k: Annotated[int, Field(ge=1, le=20, description="Number of chunks to return")] = 4,
) -> SearchResult:
    """Semantic search over all indexed document chunks (cosine similarity, highest first)."""
    result = service().search(query, top_k)
    log.info("search %r -> %s", query, [r.score for r in result.results])
    return result


@mcp.tool()
def read_document(
    doc_id: Annotated[str, Field(min_length=1)],
    max_chars: Annotated[int, Field(ge=100, le=50_000)] = 4000,
) -> DocumentText:
    """Return the (possibly truncated) full text of one document."""
    with expected_errors():
        return service().read_document(doc_id, max_chars)


@mcp.tool()
def delete_document(doc_id: Annotated[str, Field(min_length=1)]) -> DeleteResult:
    """Remove a document and its chunks from the index."""
    return service().delete_document(doc_id)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
