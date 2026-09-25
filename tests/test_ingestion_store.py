import pytest

from docqa.ingestion import build_chunks, chunk_text, doc_id_for, load_pages
from docqa.llm import FakeLLM
from docqa.models import Chunk, DocumentInfo
from docqa.vector_store import VectorStore

from .conftest import DEMO_DOCS


def test_chunk_text_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # the start of each chunk repeats the end of the previous one
    assert chunks[0][-20:].strip().split()[-1] in chunks[1][:80]


def test_chunk_text_splits_long_paragraphs():
    chunks = chunk_text("word " * 500, chunk_size=200, overlap=0)
    assert len(chunks) >= 10 and all(len(c) <= 200 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("  \n\n  ", 200, 20) == []


@pytest.mark.parametrize("name", ["employee_handbook.md", "product_faq.txt", "release_notes.md", "security_policy.pdf"])
def test_demo_documents_load(name):
    chunks, chars = build_chunks(DEMO_DOCS / name, 800, 120)
    assert chunks and chars > 500
    assert {c.filename for c in chunks} == {name}
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_pdf_pages_and_text():
    pages = load_pages(DEMO_DOCS / "security_policy.pdf")
    assert [p for p, _ in pages] == [1, 2]
    assert "14 characters" in pages[0][1]
    chunks, _ = build_chunks(DEMO_DOCS / "security_policy.pdf", 800, 120)
    assert {c.page for c in chunks} == {1, 2}


def test_unsupported_file_type(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b")
    with pytest.raises(ValueError, match="Unsupported"):
        load_pages(f)


def _doc(name: str, n: int) -> DocumentInfo:
    return DocumentInfo(doc_id=doc_id_for(name), filename=name, file_type="txt", source_path=name,
                        content_hash="h", num_chunks=n, num_chars=10)


def test_vector_store_upsert_search_delete_and_persist(tmp_path):
    llm = FakeLLM()
    store = VectorStore(tmp_path / "idx")
    for name, texts in {"a.txt": ["apples are red", "bananas are yellow"], "b.txt": ["the sky is blue"]}.items():
        chunks = [Chunk(chunk_id=f"{name}-{i}", doc_id=doc_id_for(name), filename=name, index=i, text=t)
                  for i, t in enumerate(texts)]
        store.upsert(_doc(name, len(chunks)), chunks, llm.embed(texts))

    hits = store.search(llm.embed(["yellow bananas"], "query")[0], top_k=2)
    assert hits[0].text == "bananas are yellow"
    assert hits[0].score > hits[1].score

    # persisted and reloadable
    reloaded = VectorStore(tmp_path / "idx")
    assert [d.filename for d in reloaded.list_documents()] == ["a.txt", "b.txt"]
    assert len(reloaded.chunks) == 3

    # re-upserting a document replaces its chunks
    new = [Chunk(chunk_id="a-0", doc_id=doc_id_for("a.txt"), filename="a.txt", index=0, text="cherries")]
    reloaded.upsert(_doc("a.txt", 1), new, llm.embed(["cherries"]))
    assert len(reloaded.chunks) == 2

    assert reloaded.delete(doc_id_for("b.txt")) is True
    assert reloaded.delete("missing") is False
    assert [c.text for c in VectorStore(tmp_path / "idx").chunks] == ["cherries"]


def test_vector_store_empty_search(tmp_path):
    assert VectorStore(tmp_path / "idx").search([0.1, 0.2], 3) == []


def test_document_service_ingest_is_idempotent(service):
    path = DEMO_DOCS / "product_faq.txt"
    assert service.ingest(path).status == "unchanged"
    assert len(service.list_documents().documents) == 4


def test_document_service_update_and_read(settings, fake_llm, tmp_path):
    from docqa.document_service import DocumentService

    svc = DocumentService(settings, fake_llm)
    f = tmp_path / "note.md"
    f.write_text("# Note\n\nThe office cat is called Pixel.")
    assert svc.ingest(f).status == "added"
    f.write_text("# Note\n\nThe office cat is called Byte.")
    assert svc.ingest(f).status == "updated"
    doc = svc.list_documents().documents[0]
    assert "Byte" in svc.read_document(doc.doc_id).text
    assert svc.search("office cat name", 1).results[0].filename == "note.md"
    with pytest.raises(KeyError):
        svc.read_document("nope")
    with pytest.raises(FileNotFoundError):
        svc.ingest(tmp_path / "missing.txt")
