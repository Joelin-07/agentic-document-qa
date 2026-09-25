"""Load PDF/TXT/MD files and split them into overlapping, paragraph-aware chunks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from docqa.models import Chunk

SUPPORTED_TYPES = {".pdf": "pdf", ".txt": "txt", ".md": "md", ".markdown": "md"}


def file_type(path: Path) -> str:
    try:
        return SUPPORTED_TYPES[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: pdf, txt, md") from None


def load_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return ``(page_number, text)`` pairs. Text files are a single page with number None."""
    kind = file_type(path)
    if kind == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_id_for(filename: str) -> str:
    """Stable id per filename, so re-ingesting a file replaces its previous version."""
    return hashlib.sha1(filename.lower().encode()).hexdigest()[:10]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks of at most ``chunk_size`` characters.

    Paragraphs longer than ``chunk_size`` are split on sentence/word boundaries.
    Each new chunk starts with the last ``overlap`` characters of the previous one.
    """
    text = text.replace("\r\n", "\n")
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    pieces: list[str] = []
    for para in filter(None, paragraphs):
        pieces.extend(_split_long(para, chunk_size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = _tail(current, overlap)
        current = f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) + 2 <= chunk_size else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    parts: list[str] = []
    current = ""
    for word in re.split(r"(?<=[.!?])\s+|\s+", paragraph):
        candidate = f"{current} {word}" if current else word
        if len(candidate) > size and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return "" if overlap <= 0 else text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space >= 0 else tail


def build_chunks(path: Path, chunk_size: int, overlap: int) -> tuple[list[Chunk], int]:
    """Load ``path`` and return its chunks plus the total character count."""
    doc_id = doc_id_for(path.name)
    chunks: list[Chunk] = []
    total_chars = 0
    for page, text in load_pages(path):
        total_chars += len(text)
        for piece in chunk_text(text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{len(chunks)}",
                    doc_id=doc_id,
                    filename=path.name,
                    index=len(chunks),
                    text=piece,
                    page=page,
                )
            )
    return chunks, total_chars
