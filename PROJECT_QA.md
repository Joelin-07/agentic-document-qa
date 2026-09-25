# Project Q&A: Demo & Interview Prep

Short answers you can say out loud. For setup commands see [DEMO.md](DEMO.md); for architecture see [README.md](README.md).

---

## 1. The project in one minute

**Q: What is this project?**
A local AI assistant that answers questions about your own documents (PDF, TXT, Markdown). You upload files, ask questions in plain English, and get answers with sources.

**Q: What makes it different from just using ChatGPT?**
Everything runs on your own machine. No cloud, no paid API, no data leaves the laptop. It answers only from your documents, shows its sources, and says "I don't know" when the answer isn't there.

**Q: What are the main building blocks?**
- **Streamlit:** the web UI.
- **LangGraph:** the step-by-step workflow.
- **MCP:** the document tools.
- **Ollama:** the local AI models.
- **Pydantic:** data validation.
- **SQLite:** conversation memory.
- **JSONL traces:** observability.
- **Eval script:** quality measurement.

**Q: What is RAG, and is this RAG?**
Yes. RAG means Retrieval-Augmented Generation. We first *retrieve* the relevant document pieces, then the LLM *generates* an answer using only those pieces.

---

## 2. End-to-end flow

**Q: Walk me through what happens from upload to answer.**

1. **Upload:** the user adds a file in the sidebar. It is saved to `data/uploads/`.
2. **Ingest (via MCP):** the app calls the MCP tool `ingest_document`.
3. **Processing:** the text is extracted (pypdf for PDFs), split into ~800-character chunks with overlap, and each chunk is turned into a vector by Ollama's `nomic-embed-text`.
4. **Index:** the vectors and text are saved to `data/index/` (a NumPy file plus JSON).
5. **Question:** the user types a question in the Chat tab.
6. **LangGraph workflow starts:**
   - **contextualize:** if there is chat history, rewrite a follow-up into a standalone question.
   - **retrieve:** call the MCP tool `search_documents` to get the top 4 chunks by similarity.
   - **grade:** keep chunks scoring ≥ 0.55. If none pass, go to **refuse**.
   - **generate:** Ollama (`llama3.2:3b`) writes an answer as JSON, using only those chunks.
   - **validate:** Pydantic checks the JSON. If it's broken, retry once.
7. **Response:** the answer, confidence and citations are shown in the chat.
8. **Memory:** the question and answer are saved in SQLite under the conversation's thread id.
9. **Observability:** a trace of every step (time, tokens, scores) is appended to `data/traces.jsonl`.
10. **Evaluation (separate):** a 15-question test set measures accuracy on demand.

**Q: Which parts use the LLM?**
Three places:
- the query rewrite (only when there is history)
- the answer
- the optional judge in evaluation

Embeddings use a separate small model.

**Q: Which parts do NOT use the LLM?**
Chunking, search, grading, refusal, validation, memory storage and tracing. These are plain code, so they are fast and predictable.

---

## 3. The Streamlit UI

**Q: What is Streamlit and why use it?**
A Python library that turns a Python script into a web app. We used it because it gives a clean demo UI in one file (`app.py`) with no HTML or JavaScript. It runs at `http://localhost:8501`.

**Q: What happens if we remove it?**
The system still works through the CLI (`docqa ask`, `docqa chat`). Streamlit is only the presentation layer.

### Sidebar

| Element | What it is | Why it exists |
|---|---|---|
| **System** (🟢/🔴 models) | Health check of Ollama and both models | Shows immediately if the AI backend is down or a model is missing |
| **Add files / Ingest uploads** | File uploader for PDF/TXT/MD | Adds your own documents to the index |
| **Load demo docs** | Ingests the 4 sample files | Quick start for demos |
| **Document list + ✕** | Indexed files with chunk counts; ✕ deletes one | Shows what the assistant knows; lets you remove files |
| **Memory: thread id** | Name of the current conversation | Each thread has its own memory |
| **"N turns stored"** | Count of saved question/answer pairs | Proves memory is working and persisted |
| **New chat / Clear memory** | Start a fresh thread / wipe the current one | Shows memory isolation and reset |

### Tabs

| Tab | What it shows | Why it exists |
|---|---|---|
| **💬 Chat** | Conversation, answers, confidence badge, route, "memory rewrite" note, **Sources** expander | The main feature: Q&A with proof of where the answer came from |
| **🔀 LangGraph trace** | Workflow diagram with the path taken in green, node timings, retrieved chunks with scores and kept/dropped | Makes the "agent" transparent: you can see every step it took |
| **📊 Evaluation** | Run the test set; accuracy, retrieval, citation, refusal and judge scores; per-question table | Proves quality with numbers instead of "it looks right" |
| **📈 Observability** | Requests, avg/p95 latency, not-found rate, errors/retries, tokens, time per node, recent traces | Shows how the system behaves over time: speed, failures, cost |
| **🧰 MCP tools** | The 5 MCP tools with their input schemas, plus a form to call any tool manually | Proves the agent uses a real tool protocol, and shows Pydantic validation live |

**Q: What does the confidence badge mean?**
🟢 high, 🟡 medium, 🔴 low. The model reports it, and we force it to "low" if the model claims an answer but cites no source.

**Q: What does "memory rewrite" mean in the chat?**
The follow-up question was rewritten using earlier messages. For example, "Can I carry *them* over?" becomes "Can 25 days of paid time off be carried over?"

---

## 4. LangGraph

**Q: What is LangGraph?**
A library for building AI workflows as a graph: each step is a *node*, and *edges* (including if/else branches) connect them. It also has built-in memory (checkpointing).

**Q: Why do we need it?**
Our process isn't a single LLM call. It has decisions (refuse or answer?) and loops (retry if the JSON is invalid). LangGraph makes that flow explicit, testable and visible.

**Q: How is it implemented here?**
`src/docqa/graph.py` has a `StateGraph` with 6 nodes: `contextualize → retrieve → grade → generate → validate`, plus `refuse`. It has two conditional edges:
- after `grade`: relevant → `generate`, nothing relevant → `refuse`
- after `validate`: invalid → back to `generate` (once), valid → END

**Q: What is the "state"?**
A dictionary that flows through the nodes: question, rewritten question, retrieved chunks, raw answer, final answer, trace id and the message history.

**Q: What happens if we remove it?**
We'd write the same logic as nested if/else code. It would work, but we'd lose the visual graph, the built-in memory, and the clean separation of steps.

**Q: Where can I see it in the demo?**
The **LangGraph trace** tab. Compare a normal question (ends in `validate`) with an off-topic one (ends in `refuse`).

---

## 5. MCP (Model Context Protocol)

**Q: What is MCP?**
An open standard for connecting AI applications to tools and data. A *server* exposes tools, and a *client* (the AI app) discovers and calls them in a standard format.

**Q: Why do we need it?**
It separates "the brain" (agent) from "the hands" (document tools). The agent never touches files or the index directly. It just calls tools. The same server could be plugged into any MCP-compatible app.

**Q: What tools does our server have?**
`ingest_document`, `list_documents`, `search_documents`, `read_document`, `delete_document`.

**Q: How is it implemented?**
- `mcp_server.py` defines the tools with the official MCP Python SDK (`MCPServer`).
- `mcp_client.py` starts the server as a background subprocess and talks to it over stdin/stdout (the "stdio" transport).
- The actual logic lives in `document_service.py`.

**Q: What happens if we remove it?**
The agent could call the Python functions directly, which is simpler. But we'd lose the standard tool interface, tool discovery, schema validation at the boundary, and reusability by other AI clients.

**Q: How do I prove it's really MCP?**
- **MCP tools** tab: the tool list comes live from the server.
- In the trace, the `retrieve` step shows `tool: mcp:search_documents`.
- You can run the server alone: `python -m docqa.mcp_server`.

---

## 6. Pydantic

**Q: What is Pydantic?**
A Python library that defines data shapes as classes and automatically validates data against them (types, required fields, ranges).

**Q: Why do we need it?**
LLMs and external inputs are unreliable. Pydantic guarantees that every piece of data crossing a boundary has the right shape, or fails with a clear error.

**Q: Where is it used?**
- **Settings** (`config.py`): e.g. `top_k` must be 1–20, overlap must be smaller than chunk size.
- **MCP tools**: inputs validated on the server, outputs validated again on the client.
- **LLM answer** (`LLMAnswer`): its JSON schema is sent to Ollama, and the reply is validated.
- **Traces and evaluation data**: every trace line and eval question is a typed model.

**Q: What happens if the LLM returns bad JSON?**
The `validate` node catches it, sends the error back to the model, and retries once. If it fails again, the user gets a polite "please rephrase" message instead of a crash.

**Q: Quick live demo?**
In the **MCP tools** tab, call `search_documents` with `{"query": "refund", "top_k": 99}`. It is rejected: *top_k must be ≤ 20*.

---

## 7. Ollama

**Q: What is Ollama?**
A free tool that runs open-source AI models locally, with a simple API on `localhost:11434`.

**Q: Which models and why?**
- `llama3.2:3b` for answers: small (2 GB), runs on CPU, good enough for short factual answers.
- `nomic-embed-text` for embeddings: small and fast, and turns text into vectors for search.

**Q: What are embeddings, simply?**
A list of numbers that represents the *meaning* of a text. Similar meanings give similar numbers, so we can find relevant chunks by comparing them (cosine similarity).

**Q: How does Ollama return valid JSON?**
We pass our Pydantic schema as Ollama's `format` parameter, which forces the model to output JSON of that shape.

**Q: What happens if Ollama is down?**
The sidebar shows 🔴 / "not reachable", and questions show an error. Nothing else breaks. Start it with `ollama serve`.

**Q: Can we use a better model?**
Yes. Change `DOCQA_CHAT_MODEL` in `.env` (e.g. `qwen2.5:7b`). No code change is needed.

---

## 8. Memory

**Q: What is memory here?**
The assistant remembers the conversation, so follow-up questions work ("And how much storage does *it* include?").

**Q: How is it implemented?**
LangGraph's **SQLite checkpointer** saves the full graph state, including the messages, per `thread_id` in `data/memory.sqlite`. On each new question, the `contextualize` node reads the recent history and rewrites the follow-up into a standalone question.

**Q: Why rewrite the question instead of just sending the history?**
Search works on the question text. "Does *it* include storage?" would retrieve nothing useful. The rewritten "Does the Pro plan include storage?" finds the right chunk.

**Q: Does memory survive a restart?**
Yes, because it's stored in SQLite, not in RAM. Restart the app, use the same thread id, and the history is there.

**Q: What happens if we remove it?**
Every question is treated as new. Follow-ups with "it", "that" or "them" fail.

---

## 9. Evaluation

**Q: What is the evaluation?**
A fixed set of 15 questions with expected answers (`demo_data/eval/eval_set.jsonl`): 12 answerable, 2 unanswerable, and 1 follow-up that needs memory.

**Q: What metrics do we measure?**

| Metric | Meaning |
|---|---|
| Accuracy | Answer correct (or correctly refused) |
| Retrieval hit rate | The right document was among the retrieved chunks |
| Citation accuracy | The answer cited the right document |
| Refusal accuracy | Unanswerable questions were declined |
| Keyword score | Expected key facts (e.g. "25", "90 days") appear in the answer |
| Judge score | Optional: the local LLM grades each answer 1–5 against the expected answer |
| Latency | Time per question |

**Q: Why do we need it?**
To measure quality objectively and catch regressions. For example, after changing the prompt, the model, or `min_score`, re-run it and compare.

**Q: Current results?**
15/15 correct, 100% retrieval, 100% refusal, 92% citation accuracy, judge 5/5, about 6–13 s per question on CPU.

**Q: How do I run it?**
Use the **Evaluation** tab, or run `docqa eval` (add `--judge` for LLM grading). Reports are saved in `data/eval_reports/`.

---

## 10. Observability

**Q: What is observability here?**
Recording what happened inside every request, so we can debug and monitor it.

**Q: What is recorded?**
For each question, one **trace**:
- the route taken
- each node's duration
- tokens used
- retrieval scores
- the rewritten question
- validation results
- errors

It is saved as one JSON line in `data/traces.jsonl`.

**Q: How is it implemented?**
A small `Tracer` in `observability.py` wraps each LangGraph node with a decorator that times it and collects details. No external service is needed.

**Q: Why do we need it?**
To answer "why was this answer wrong or slow?" For example, the trace showed that one wrong citation happened because the model picked passage [2] instead of [1].

**Q: What happens if we remove it?**
The app still works, but it's a black box. There's no latency data, no debugging trail and no metrics.

**Q: Where can I see it?**
The **Observability** tab (dashboard), the **LangGraph trace** tab (one request), or `docqa traces` in the terminal.

---

## 11. Docker

**Q: Why is Docker included?**
As an optional, repeatable way to run the app on any machine without installing Python.

**Q: Is it required?**
No. The primary setup is native Python plus native Ollama on Windows, which is faster and can use the GPU.

**Q: What does the Docker setup contain?**
- A `Dockerfile` for the app.
- `docker-compose.yml`:
  - `docker compose up`: app in Docker, Ollama on the host.
  - `--profile ollama`: Ollama in Docker too (CPU only).

---

## 12. Other design choices

**Q: Why no vector database (Chroma, Pinecone…)?**
A few documents produce only a few dozen chunks. A NumPy matrix gives exact, instant search with nothing to install. It can be swapped out later.

**Q: Why no PySpark?**
Spark is for big data. Our corpus is a few files, so Spark would add Java, slow startup and complexity for zero benefit.

**Q: What is chunking and why overlap?**
We split documents into ~800-character pieces so search returns focused passages. Consecutive chunks share ~120 characters, so a sentence cut at a boundary isn't lost.

**Q: What is `min_score` (0.55)?**
The minimum similarity for a chunk to count as relevant. We picked it from measured scores: real questions scored ≥ 0.62, and off-topic ones 0.46–0.52.

**Q: How does it avoid hallucination?**
There are two layers:
1. If no chunk passes `min_score`, the agent refuses without calling the LLM.
2. The prompt tells the LLM to use only the given passages and return `found=false` otherwise.

**Q: How are tests possible without Ollama?**
A `FakeLLM` gives deterministic answers and embeddings, so all 43 tests run in about 20 seconds. One extra test runs against real Ollama when it's available.

---

## 13. Likely team-lead questions

**Q: Show me it doesn't make things up.**
Ask *"What is the weather in Paris?"*: it refuses via the `refuse` route. Ask *"What is the stock price of Nimbus Labs?"*: the documents look related, but the model still says "I don't know".

**Q: Show me memory working.**
Ask *"How many days of PTO do employees get?"*, then *"Can I carry any of them over?"*. The chat shows the rewritten question, and the sidebar counter increases.

**Q: How do I know the answer is correct?**
Open **Sources**: it shows the exact chunk, file and similarity score the answer came from.

**Q: How long does a question take, and where does the time go?**
About 6–13 s on CPU. The Observability tab shows that `generate` (the LLM) takes most of it. Retrieval takes milliseconds.

**Q: What if I upload a new document?**
It is chunked, embedded and searchable immediately. Re-uploading an unchanged file does nothing; a changed file replaces the old version.

**Q: What breaks with a scanned PDF?**
There is no extractable text, so the app shows "No extractable text". OCR is out of scope for this POC.

**Q: How would you make this production-ready?**
- A bigger model or a GPU.
- A real vector DB for many documents.
- Authentication and multi-user support.
- OCR.
- Deployment as a service.
- Tracing sent to a dashboard (e.g. Langfuse or Phoenix, both self-hostable).

**Q: How do you know a change didn't make it worse?**
Run `pytest` (logic) and `docqa eval` (quality) before and after, then compare the reports.

**Q: What was the hardest part?**
Getting reliable output from a small 3B model. It was solved with a JSON schema, Pydantic validation plus retry, and a similarity cutoff tuned on real scores.

**Q: Is any data sent outside the machine?**
No. The only network calls go to Ollama on `localhost`. Streamlit usage statistics are turned off in `.streamlit/config.toml`.
