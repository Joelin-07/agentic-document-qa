# Local AI Document Assistant (POC)

Ask questions about your own PDF / TXT / Markdown files, fully on your machine.
No cloud services, no paid APIs, no vector database.

| Technology | What it does here |
|---|---|
| **LangGraph** | 6-node workflow with conditional routing, a validation-retry loop and a SQLite checkpointer |
| **MCP** | All document access goes through an MCP server (stdio) with 5 tools; the agent is an MCP client |
| **Pydantic** | Settings, MCP tool inputs/outputs, the LLM's JSON answer, traces and the eval dataset are all typed and validated |
| **Ollama** | Local chat model (`llama3.2:3b`) with JSON-schema structured output + local embeddings (`nomic-embed-text`) |
| **Memory** | Conversation state persisted per thread in SQLite; follow-up questions are rewritten using history |
| **Evaluation** | 15-question dataset, retrieval/citation/keyword/refusal metrics, optional LLM-as-judge |
| **Observability** | Per-node spans (latency, tokens, retrieval scores, errors) written to `data/traces.jsonl` and shown in the UI |
| **Streamlit** | Minimal local UI on http://localhost:8501 |

See **[DEMO.md](DEMO.md)** for exact setup commands and a 5-10 minute demo script.

## Architecture

```
 Streamlit UI (app.py)  /  debug CLI (docqa ...)
            │
            ▼
   DocQAAgent ── LangGraph StateGraph ─────────────────────────────────────────┐
   │                                                                           │
   │  START → contextualize → retrieve → grade ─┬─ relevant ─→ generate → validate → END
   │           (memory)       (MCP)     (cutoff) │                 ▲   retry ──┘
   │                                             └─ nothing ─→ refuse → END
   │                                                                           │
   │  SqliteSaver checkpointer (data/memory.sqlite)  ◄── conversation memory   │
   │  Tracer (data/traces.jsonl)                     ◄── spans for every node  │
   └───────────────┬───────────────────────────────────────────────────────────┘
                   │ MCP over stdio (subprocess)
                   ▼
   MCP server "docqa-documents"  (src/docqa/mcp_server.py)
     tools: ingest_document · list_documents · search_documents · read_document · delete_document
                   │
                   ▼
   DocumentService → loaders (pypdf / text) → chunker → Ollama embeddings
                   → NumPy vector store (data/index/chunks.json + vectors.npy)
```

**Question flow**

1. **contextualize**: if the thread has history, the LLM rewrites a follow-up ("does *it* include storage?") into a standalone question.
2. **retrieve**: calls the MCP tool `search_documents` (cosine similarity over chunk embeddings).
3. **grade**: keeps chunks with score ≥ `DOCQA_MIN_SCORE`. If none remain, it routes to **refuse** and returns a fixed "not in the documents" answer (no LLM call, no hallucination).
4. **generate**: Ollama answers from numbered passages and must return JSON matching the `LLMAnswer` Pydantic schema.
5. **validate**: Pydantic validates the JSON and maps citation numbers to real chunks. If it is invalid, the error is fed back and generation is retried once.

## Project layout

```
app.py                     Streamlit UI (primary interface)
src/docqa/
  config.py                Pydantic settings (DOCQA_* env vars / .env)
  models.py                Pydantic models: documents, MCP I/O, LLM answer, traces, evaluation
  llm.py                   Ollama client + deterministic FakeLLM for tests
  ingestion.py             PDF/TXT/MD loading, paragraph-aware chunking
  vector_store.py          NumPy cosine-similarity store persisted to disk
  document_service.py      Tool logic (ingest/search/list/read/delete)
  mcp_server.py            MCP server exposing the tools (stdio)
  mcp_client.py            Sync MCP client (persistent session on a background loop)
  graph.py                 LangGraph workflow, memory, DocQAAgent facade
  prompts.py               Prompt templates
  observability.py         Tracer, JSONL traces, metrics summary
  evaluation.py            Evaluation runner + metrics + reports
  runtime.py               Wires settings → LLM → MCP client → agent
  cli.py                   Debug CLI (docqa ...)
demo_data/docs/            4 sample documents (MD, TXT, PDF) about a fictional company
demo_data/eval/            eval_set.jsonl (15 questions with expected answers)
scripts/                   make_sample_pdf.py, run_eval.py
tests/                     unit, MCP, graph, eval and end-to-end tests (incl. Streamlit AppTest)
data/                      created at runtime: index, memory DB, traces, eval reports
```

## Quick start (Windows, native Ollama)

```powershell
# 1. Ollama (native Windows app) + models
winget install Ollama.Ollama
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Python 3.11-3.13 environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
copy .env.example .env

# 3. Check and run
docqa doctor
streamlit run app.py          # → http://localhost:8501
```

On macOS/Linux, use `python3 -m venv .venv && source .venv/bin/activate` and install Ollama from https://ollama.com.

## Debug CLI

| Command | Purpose |
|---|---|
| `docqa doctor` | Show settings, check Ollama and models |
| `docqa ingest [PATH...]` | Ingest files/folders via MCP (default: `demo_data/docs`) |
| `docqa docs` | List indexed documents |
| `docqa ask "question" --thread t1` | One question (memory kept per thread) |
| `docqa chat --thread t1` | Interactive chat (`/history`, `/clear`, `/exit`) |
| `docqa eval [--judge] [--limit N]` | Run the evaluation and write a report |
| `docqa traces --last 10` | Recent traces and aggregate metrics |
| `docqa graph` | Print the LangGraph workflow as Mermaid |

## Tests and evaluation

```powershell
pytest                          # full suite with the fake LLM (no Ollama needed)
pytest -m ollama                # live test against Ollama (skipped if Ollama/models are missing)
docqa eval                      # 15-question evaluation with the real models
python scripts/run_eval.py --judge
```

Tests use `FakeLLM`, a deterministic stand-in (hashed bag-of-words embeddings and a
sentence-overlap "answerer"). It exercises every node, route, the MCP subprocess, the CLI
and the Streamlit app without a model.

## Configuration

All settings are in `src/docqa/config.py` and can be overridden with `DOCQA_*` environment
variables or a `.env` file (see `.env.example`). The most useful ones:

| Variable | Default | Notes |
|---|---|---|
| `DOCQA_CHAT_MODEL` | `llama3.2:3b` | Any Ollama chat model, e.g. `qwen2.5:7b` for better answers |
| `DOCQA_EMBED_MODEL` | `nomic-embed-text` | If you change it, delete `data/index` and re-ingest |
| `DOCQA_MIN_SCORE` | `0.55` | Relevance cutoff; lower it if the assistant refuses too often |
| `DOCQA_TOP_K` | `4` | Chunks retrieved per question |
| `DOCQA_LLM_BACKEND` | `ollama` | `fake` runs without Ollama (tests) |

## Docker (optional)

Native Python + native Ollama is the recommended setup (faster, and it can use your GPU).
Docker is provided as a fallback:

```powershell
docker compose up --build                   # app in Docker, Ollama native on the host
docker compose --profile ollama up --build  # app + Ollama both in Docker (CPU only)
```

With the `ollama` profile, set `DOCQA_OLLAMA_HOST=http://ollama:11434` and pull the models
inside the container (`docker compose exec ollama ollama pull llama3.2:3b`, same for
`nomic-embed-text`).

## Design decisions

- **No vector database.** A few documents produce a few dozen chunks. A NumPy matrix with
  cosine similarity is exact, instant and has nothing to install. The store is behind a small
  class, so it could be swapped for Chroma/Qdrant later.
- **Refusal before generation.** The `grade` node refuses when no chunk passes the cutoff, so
  off-topic questions never reach the LLM. The LLM can also decline with `found=false`.
- **Structured output + validation.** Small models sometimes return malformed JSON. Ollama's
  JSON-schema `format` makes that rare, and the validate → retry loop handles the rest.
- **MCP as the only data path.** The agent never imports the vector store. Everything goes
  through MCP tools, so the same server works with MCP Inspector or any other MCP host.
- **Memory = LangGraph checkpointer.** No custom memory code. The graph state (including
  messages) is saved per `thread_id` in SQLite and survives restarts.

### Why PySpark is not included

PySpark was optional and is left out on purpose:

- **No real benefit at this scale.** The corpus is a handful of documents. Parsing and
  chunking take milliseconds in plain Python, and the slow step (embedding) is bound by
  the local Ollama server, which Spark cannot speed up.
- **Heavy dependencies.** PySpark needs a Java runtime (not installed on this machine) and,
  on Windows, `winutils.exe`/Hadoop settings. That would add more setup than the rest of
  the POC combined.
- **Slower and harder to follow.** JVM startup adds seconds to every run, and Spark's
  execution model would hide the simple ingest → chunk → embed flow the POC is meant to show.

If the corpus grew to thousands of documents, batch ingestion is where Spark (or a simpler
multiprocessing pool) could be added, inside `DocumentService.ingest`.

## Limitations

- Scanned (image-only) PDFs are not supported (no OCR).
- A 3B model is fast on CPU but less accurate than larger models; switch `DOCQA_CHAT_MODEL` for quality.
- Single-user, local POC: no authentication, no concurrent-write safety beyond a process-local lock.
