"""Document tool logic (ingest / list / search / read / delete).

The MCP server exposes these methods as MCP tools. Tests can also use the service
directly (in-process) through the same ``DocumentTools`` interface the agent expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from docqa.config import Settings
from docqa.ingestion import build_chunks, content_hash, doc_id_for, file_type, load_pages
from docqa.llm import LLMClient
from docqa.models import (
    DeleteResult,
    DocumentInfo,
    DocumentList,
    DocumentText,
    IngestResult,
    SearchResult,
)
from docqa.vector_store import VectorStore


class DocumentTools(Protocol):
    """What the LangGraph agent needs from the document layer (implemented over MCP)."""

    def search(self, query: str, top_k: int) -> SearchResult: ...


class DocumentService:
    def __init__(self, settings: Settings, llm: LLMClient):
        settings.ensure_dirs()
        self.settings = settings
        self.llm = llm
        self.store = VectorStore(settings.index_dir)

    def ingest(self, path: str | Path) -> IngestResult:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        kind = file_type(path)
        digest = content_hash(path)
        doc_id = doc_id_for(path.name)

        existing = self.store.get_document(doc_id)
        if existing and existing.content_hash == digest:
            return IngestResult(status="unchanged", document=existing)

        chunks, num_chars = build_chunks(path, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError(f"No extractable text in {path.name} (scanned PDFs are not supported)")
        vectors = self.llm.embed([c.text for c in chunks], kind="document")
        doc = DocumentInfo(
            doc_id=doc_id,
            filename=path.name,
            file_type=kind,
            source_path=str(path),
            content_hash=digest,
            num_chunks=len(chunks),
            num_chars=num_chars,
        )
        self.store.upsert(doc, chunks, vectors)
        return IngestResult(status="updated" if existing else "added", document=doc)

    def list_documents(self) -> DocumentList:
        return DocumentList(documents=self.store.list_documents())

    def search(self, query: str, top_k: int) -> SearchResult:
        [vector] = self.llm.embed([query], kind="query")
        return SearchResult(query=query, results=self.store.search(vector, top_k))

    def read_document(self, doc_id: str, max_chars: int = 4000) -> DocumentText:
        doc = self.store.get_document(doc_id)
        if doc is None:
            raise KeyError(f"Unknown doc_id '{doc_id}'. Use list_documents to see available ids.")
        source = Path(doc.source_path)
        if source.is_file():
            text = "\n\n".join(t for _, t in load_pages(source))
        else:  # original file moved: fall back to the indexed chunks
            text = "\n\n".join(c.text for c in self.store.document_chunks(doc_id))
        return DocumentText(
            doc_id=doc_id, filename=doc.filename, text=text[:max_chars], truncated=len(text) > max_chars
        )

    def delete_document(self, doc_id: str) -> DeleteResult:
        return DeleteResult(doc_id=doc_id, deleted=self.store.delete(doc_id))
