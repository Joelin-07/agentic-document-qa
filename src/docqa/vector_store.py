"""A tiny persistent vector store: NumPy matrix + JSON metadata, cosine similarity search.

For a handful of documents this is all that is needed - no vector database.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from docqa.models import Chunk, DocumentInfo, RetrievedChunk


class VectorStore:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.meta_path = self.index_dir / "chunks.json"
        self.vec_path = self.index_dir / "vectors.npy"
        self._lock = threading.RLock()
        self._mtime: float | None = None
        self.documents: dict[str, DocumentInfo] = {}
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        if not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.documents = {d["doc_id"]: DocumentInfo.model_validate(d) for d in meta["documents"]}
        self.chunks = [Chunk.model_validate(c) for c in meta["chunks"]]
        self.vectors = np.load(self.vec_path) if self.vec_path.exists() else np.zeros((0, 0), dtype=np.float32)
        self._mtime = self.meta_path.stat().st_mtime

    def _reload_if_changed(self) -> None:
        """Pick up writes made by another process (e.g. a second MCP server instance)."""
        if self.meta_path.exists() and self.meta_path.stat().st_mtime != self._mtime:
            self._load()

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp_vec = self.vec_path.with_suffix(".tmp.npy")
        np.save(tmp_vec, self.vectors)
        os.replace(tmp_vec, self.vec_path)
        meta = {
            "documents": [d.model_dump(mode="json") for d in self.documents.values()],
            "chunks": [c.model_dump(mode="json") for c in self.chunks],
        }
        tmp_meta = self.meta_path.with_suffix(".tmp")
        tmp_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(tmp_meta, self.meta_path)
        self._mtime = self.meta_path.stat().st_mtime

    # ------------------------------------------------------------------ writes

    def upsert(self, doc: DocumentInfo, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            self._reload_if_changed()
            self._drop(doc.doc_id)
            new = _normalize(np.asarray(vectors, dtype=np.float32).reshape(len(vectors), -1))
            if self.vectors.size and new.size and self.vectors.shape[1] != new.shape[1]:
                raise ValueError(
                    f"Embedding dimension changed ({self.vectors.shape[1]} -> {new.shape[1]}). "
                    "Delete the data/index folder and re-ingest after switching embedding models."
                )
            self.vectors = new if not self.vectors.size else np.vstack([self.vectors, new])
            self.chunks.extend(chunks)
            self.documents[doc.doc_id] = doc
            self._save()

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            self._reload_if_changed()
            if doc_id not in self.documents:
                return False
            self._drop(doc_id)
            self._save()
            return True

    def _drop(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        if len(keep) != len(self.chunks):
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self.documents.pop(doc_id, None)

    # ------------------------------------------------------------------ reads

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        with self._lock:
            self._reload_if_changed()
            if not self.chunks:
                return []
            q = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
            scores = self.vectors @ q
            order = np.argsort(-scores)[:top_k]
            return [
                RetrievedChunk(**self.chunks[i].model_dump(), score=round(float(scores[i]), 4))
                for i in order
            ]

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            self._reload_if_changed()
            return sorted(self.documents.values(), key=lambda d: d.filename.lower())

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        with self._lock:
            self._reload_if_changed()
            return self.documents.get(doc_id)

    def document_chunks(self, doc_id: str) -> list[Chunk]:
        with self._lock:
            self._reload_if_changed()
            return [c for c in self.chunks if c.doc_id == doc_id]


def _normalize(m: np.ndarray) -> np.ndarray:
    if not m.size:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
