"""Streamlit UI - the primary demo interface.

Run:  streamlit run app.py   ->  http://localhost:8501
"""

from __future__ import annotations

import json
import uuid

import streamlit as st

from docqa.config import Settings
from docqa.evaluation import latest_report, load_dataset, run_evaluation, save_report, summary_metrics
from docqa.graph import graph_dot
from docqa.llm import OllamaLLM
from docqa.mcp_client import MCPToolError
from docqa.observability import load_traces, summarize
from docqa.runtime import Runtime, build_runtime

st.set_page_config(page_title="Local Document Assistant", layout="wide")


@st.cache_resource(show_spinner="Starting the MCP document server...")
def get_runtime() -> Runtime:
    return build_runtime(Settings())


@st.cache_data(ttl=15, show_spinner=False)
def ollama_status(host: str, chat_model: str, embed_model: str) -> dict:
    try:
        return OllamaLLM(Settings(ollama_host=host, chat_model=chat_model, embed_model=embed_model)).health()
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


rt = get_runtime()
settings = rt.settings
ss = st.session_state
ss.setdefault("thread_id", "demo")
ss.setdefault("meta", {})  # (thread_id, assistant message index) -> response details
ss.setdefault("last", None)  # last AgentResponse


# ============================================================================ sidebar

with st.sidebar:
    st.header("Local Document Assistant")

    st.subheader("System")
    if settings.llm_backend == "fake":
        st.warning("Fake LLM backend (tests/demo without Ollama)")
    else:
        status = ollama_status(settings.ollama_host, settings.chat_model, settings.embed_model)
        if not status.get("reachable"):
            st.error(f"Ollama not reachable at {settings.ollama_host}. Start it with `ollama serve`.")
        else:
            for key in ("chat_model", "embed_model"):
                ok = status[f"{key}_ok"]
                st.markdown(f"{'🟢' if ok else '🔴'} `{status[key]}`" + ("" if ok else f" - run `ollama pull {status[key]}`"))

    st.subheader("Documents")
    uploads = st.file_uploader("Add PDF / TXT / MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    col_a, col_b = st.columns(2)
    if col_a.button("Ingest uploads", disabled=not uploads, width="stretch"):
        with st.spinner("Chunking and embedding via MCP..."):
            for f in uploads:
                target = settings.uploads_dir / f.name
                target.write_bytes(f.getbuffer())
                try:
                    r = rt.mcp.ingest(str(target))
                    st.toast(f"{r.document.filename}: {r.status} ({r.document.num_chunks} chunks)")
                except MCPToolError as exc:
                    st.error(str(exc))
    if col_b.button("Load demo docs", width="stretch"):
        with st.spinner("Ingesting demo_data/docs via MCP..."):
            try:
                for r in rt.ingest_folder(settings.demo_docs_dir):
                    st.toast(f"{r.document.filename}: {r.status}")
            except MCPToolError as exc:
                st.error(str(exc))

    documents = rt.mcp.list_documents()
    if not documents:
        st.info("No documents yet. Upload files or load the demo docs.")
    for d in documents:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d.filename}**  \n<small>{d.num_chunks} chunks · {d.file_type}</small>", unsafe_allow_html=True)
        if c2.button("✕", key=f"del-{d.doc_id}", help="Delete from index"):
            rt.mcp.delete_document(d.doc_id)
            st.rerun()

    st.subheader("Memory")
    ss.thread_id = st.text_input("Conversation (thread id)", value=ss.thread_id).strip() or "demo"
    history = rt.agent.history(ss.thread_id)
    turns = sum(m["role"] == "user" for m in history)
    st.markdown(f"🧠 **{turns} turns** stored for `{ss.thread_id}`  \n<small>SQLite checkpointer: `{settings.memory_db.name}`</small>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    if col_c.button("New chat", width="stretch"):
        ss.thread_id = f"chat-{uuid.uuid4().hex[:6]}"
        st.rerun()
    if col_d.button("Clear memory", width="stretch"):
        rt.agent.clear(ss.thread_id)
        st.rerun()


# ============================================================================ tabs

tab_chat, tab_trace, tab_eval, tab_obs, tab_mcp = st.tabs(
    ["💬 Chat", "🔀 LangGraph trace", "📊 Evaluation", "📈 Observability", "🧰 MCP tools"]
)

# ---------------------------------------------------------------------------- chat
with tab_chat:
    messages = st.container()
    question = st.chat_input("Ask a question about your documents")
    if question:
        with st.spinner("Running LangGraph workflow..."):
            try:
                resp = rt.agent.ask(question, thread_id=ss.thread_id)
                assistant_index = len(rt.agent.history(ss.thread_id)) - 1
                ss.meta[(ss.thread_id, assistant_index)] = resp
                ss.last = resp
                st.rerun()  # refresh sidebar memory counter and trace tab
            except Exception as exc:
                st.error(f"Request failed: {exc}. Is Ollama running and are the models pulled?")

    with messages:
        if not history:
            st.caption("Try: *How many days of PTO do employees get?* then follow up with *Can I carry any of them over?*")
        for i, m in enumerate(history):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                resp = ss.meta.get((ss.thread_id, i))
                if resp is None:
                    continue
                a = resp.answer
                badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}[a.confidence]
                info = f"{badge} confidence **{a.confidence}** · {resp.trace.total_ms / 1000:.1f}s · route `{' → '.join(resp.trace.route)}`"
                if resp.standalone_question != resp.question:
                    info += f"  \n🧠 memory rewrite: *{resp.standalone_question}*"
                st.caption(info)
                if a.citations:
                    with st.expander(f"Sources ({len(a.citations)})"):
                        for c in a.citations:
                            page = f", page {c.page}" if c.page else ""
                            st.markdown(f"**{c.filename}**{page} · `{c.chunk_id}` · similarity {c.score:.2f}")
                            st.caption(c.snippet + ("..." if len(c.snippet) >= 300 else ""))

# ---------------------------------------------------------------------------- trace
with tab_trace:
    last = ss.last
    if last is None:
        st.info("Ask a question to see how it flows through the LangGraph workflow.")
        st.graphviz_chart(graph_dot())
    else:
        t = last.trace
        st.markdown(f"**Question:** {last.question}")
        if last.standalone_question != last.question:
            st.markdown(f"**Standalone (from memory):** {last.standalone_question}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total latency", f"{t.total_ms / 1000:.2f} s")
        c2.metric("Nodes executed", len(t.route))
        c3.metric("Tokens (prompt / completion)", f"{t.prompt_tokens} / {t.completion_tokens}")
        c4.metric("Answer found", "yes" if last.answer.found else "no")
        st.graphviz_chart(graph_dot(t.route))
        st.markdown("**Node spans**")
        st.dataframe(
            [{"#": i + 1, "node": s.node, "ms": s.duration_ms, "status": s.status, "detail": json.dumps(s.detail, default=str)}
             for i, s in enumerate(t.spans)],
            hide_index=True,
        )
        st.markdown(f"**Retrieved chunks** (MCP `search_documents`, cutoff `min_score={settings.min_score}`)")
        st.dataframe(
            [{"file": c.filename, "chunk": c.chunk_id, "score": c.score, "kept": c.score >= settings.min_score, "text": c.text[:160]}
             for c in last.retrieved],
            hide_index=True,
        )
        with st.expander("Mermaid diagram source (LangGraph `draw_mermaid`)"):
            st.code(rt.agent.mermaid(), language="text")

# ---------------------------------------------------------------------------- evaluation
with tab_eval:
    items = load_dataset(settings.eval_dataset)
    st.markdown(f"Dataset `{settings.eval_dataset.name}`: **{len(items)} questions** "
                f"({sum(not i.answerable for i in items)} unanswerable, {sum(bool(i.context) for i in items)} memory follow-ups)")
    c1, c2, c3 = st.columns([2, 2, 3])
    limit = c1.number_input("Questions to run", 1, len(items), len(items))
    use_judge = c2.checkbox("LLM-as-judge (slower)")
    if c3.button("▶ Run evaluation", type="primary"):
        rt.ingest_folder(settings.demo_docs_dir)
        bar = st.progress(0.0, text="Starting...")
        log = st.empty()
        lines: list[str] = []

        def progress(i, n, r):
            lines.append(f"{'✅' if r.correct else '❌'} {r.id}: {r.answer[:100]}")
            bar.progress(i / n, text=f"{i}/{n}")
            log.markdown("  \n".join(lines[-6:]))

        try:
            report = run_evaluation(rt.agent, items[: int(limit)], use_judge=use_judge, progress=progress)
            save_report(report, settings.eval_reports_dir)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    report = latest_report(settings.eval_reports_dir)
    if report:
        st.markdown(f"**Latest report** `{report.run_id}` · {report.created_at:%Y-%m-%d %H:%M} UTC · `{report.chat_model}`")
        cols = st.columns(4)
        m = summary_metrics(report)
        cols[0].metric("Accuracy", f"{m['accuracy']:.0%}")
        cols[1].metric("Retrieval hit rate", f"{m['retrieval_hit_rate']:.0%}")
        cols[2].metric("Citation accuracy", f"{m['citation_accuracy']:.0%}")
        cols[3].metric("Refusal accuracy", f"{m['refusal_accuracy']:.0%}")
        cols = st.columns(4)
        cols[0].metric("Keyword score", f"{m['mean_keyword_score']:.2f}")
        cols[1].metric("Judge score (1-5)", m["mean_judge_score"] or "-")
        cols[2].metric("Avg latency", f"{m['avg_latency_ms'] / 1000:.1f} s")
        st.dataframe(
            [r.model_dump(exclude={"trace_id"}) for r in report.results], hide_index=True
        )

# ---------------------------------------------------------------------------- observability
with tab_obs:
    traces = load_traces(settings.traces_file)
    summary = summarize(traces)
    if not traces:
        st.info(f"No traces yet. Every question is logged to `{settings.traces_file}`.")
    else:
        c = st.columns(5)
        c[0].metric("Requests", summary["requests"])
        c[1].metric("Avg latency", f"{summary['avg_latency_ms'] / 1000:.2f} s")
        c[2].metric("p95 latency", f"{summary['p95_latency_ms'] / 1000:.2f} s")
        c[3].metric("Not-found rate", f"{summary['not_found_rate']:.0%}")
        c[4].metric("Errors / retries", f"{summary['errors']} / {summary['retries']}")
        st.caption(f"Tokens: {summary['prompt_tokens']} prompt, {summary['completion_tokens']} completion")
        left, right = st.columns(2)
        left.markdown("**Average time per node (ms)**")
        left.bar_chart(summary["avg_node_ms"], horizontal=True)
        right.markdown("**Latency of recent requests (ms)**")
        right.line_chart([t.total_ms for t in traces[-50:]])
        st.markdown("**Recent traces**")
        st.dataframe(
            [{"trace": t.trace_id, "time": f"{t.started_at:%H:%M:%S}", "thread": t.thread_id, "question": t.question,
              "route": " → ".join(t.route), "ms": t.total_ms, "tokens": t.prompt_tokens + t.completion_tokens,
              "found": t.answer_found, "status": t.status} for t in reversed(traces[-30:])],
            hide_index=True,
        )
        st.download_button("Download traces.jsonl", settings.traces_file.read_bytes(), "traces.jsonl")

# ---------------------------------------------------------------------------- MCP
with tab_mcp:
    st.markdown("The agent never touches the index directly. All document access goes through this "
                "**MCP server** (stdio subprocess). Tool inputs are validated with Pydantic on the server.")
    tools = rt.mcp.list_tools()
    for tool in tools:
        with st.expander(f"`{tool['name']}` - {tool['description']}"):
            st.json(tool["input_schema"])
    st.markdown("**Call a tool manually**")
    name = st.selectbox("Tool", [t["name"] for t in tools], index=[t["name"] for t in tools].index("search_documents"))
    default_args = {"search_documents": {"query": "refund policy", "top_k": 3}, "read_document": {"doc_id": documents[0].doc_id if documents else ""}}
    args = st.text_area("Arguments (JSON)", json.dumps(default_args.get(name, {})), key=f"args-{name}")
    if st.button("Call tool"):
        try:
            st.json(rt.mcp.call_tool(name, json.loads(args or "{}")))
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        except MCPToolError as exc:
            st.error(str(exc))
