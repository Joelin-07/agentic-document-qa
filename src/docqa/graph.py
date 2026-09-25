"""The LangGraph workflow.

    START -> contextualize -> retrieve -> grade --(relevant)--> generate -> validate --> END
                                            \\                      ^          |
                                             \\                     +-(retry)--+
                                              +--(nothing relevant)--> refuse -----> END

* contextualize: uses conversation memory to rewrite follow-ups into standalone questions
* retrieve:      calls the MCP ``search_documents`` tool
* grade:         keeps chunks above the similarity cutoff and routes
* generate:      Ollama answer with JSON-schema structured output
* validate:      Pydantic validation; one retry with the error fed back to the model
* refuse:        deterministic "not in the documents" answer

Memory: a SQLite checkpointer stores the full state (incl. messages) per ``thread_id``.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError

from docqa import prompts
from docqa.config import Settings
from docqa.document_service import DocumentTools
from docqa.llm import LLMClient
from docqa.models import Answer, Citation, LLMAnswer, RetrievedChunk, Trace
from docqa.observability import Tracer

MAX_GENERATE_ATTEMPTS = 2
NODES = ["contextualize", "retrieve", "grade", "generate", "validate", "refuse"]
EDGES = [
    ("START", "contextualize", ""),
    ("contextualize", "retrieve", ""),
    ("retrieve", "grade", ""),
    ("grade", "generate", "relevant"),
    ("grade", "refuse", "nothing relevant"),
    ("generate", "validate", ""),
    ("validate", "generate", "retry"),
    ("validate", "END", "valid"),
    ("refuse", "END", ""),
]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation memory (persisted)
    question: str
    standalone_question: str
    retrieved: list[dict]  # all chunks returned by MCP search
    chunks: list[dict]  # chunks that passed grading
    raw_answer: str
    attempts: int
    validation_error: str | None
    answer: dict | None
    trace_id: str


class AgentResponse(BaseModel):
    thread_id: str
    question: str
    standalone_question: str
    answer: Answer
    retrieved: list[RetrievedChunk]
    trace: Trace


def build_graph(settings: Settings, llm: LLMClient, docs: DocumentTools, tracer: Tracer):
    node = tracer.node

    @node("contextualize")
    def contextualize(state: AgentState) -> dict:
        question = state["question"]
        history = state["messages"][:-1]  # everything before the current question
        if not history or settings.history_messages == 0:
            return {"standalone_question": question, "_trace": {"skipped": "no history"}}
        history_text = prompts.format_history(history, settings.history_messages)
        res = llm.chat(
            [
                {"role": "system", "content": prompts.CONTEXTUALIZE_SYSTEM},
                {"role": "user", "content": prompts.contextualize_prompt(history_text, question)},
            ]
        )
        rewritten = res.content.strip().strip('"').splitlines()[0].strip() if res.content.strip() else ""
        if not rewritten or len(rewritten) > 3 * len(question) + 150:
            rewritten = question  # the model misbehaved; fall back to the raw question
        return {
            "standalone_question": rewritten,
            "_trace": {
                "history_messages": len(history),
                "rewritten": rewritten,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            },
        }

    @node("retrieve")
    def retrieve(state: AgentState) -> dict:
        result = docs.search(state["standalone_question"], settings.top_k)
        retrieved = [r.model_dump() for r in result.results]
        return {
            "retrieved": retrieved,
            "_trace": {
                "tool": "mcp:search_documents",
                "query": state["standalone_question"],
                "hits": [{"file": r.filename, "chunk": r.chunk_id, "score": r.score} for r in result.results],
            },
        }

    @node("grade")
    def grade(state: AgentState) -> dict:
        relevant = [c for c in state["retrieved"] if c["score"] >= settings.min_score]
        return {
            "chunks": relevant,
            "_trace": {
                "min_score": settings.min_score,
                "kept": len(relevant),
                "dropped": len(state["retrieved"]) - len(relevant),
            },
        }

    @node("generate")
    def generate(state: AgentState) -> dict:
        history_text = prompts.format_history(state["messages"][:-1], settings.history_messages)
        prompt = prompts.answer_prompt(
            history_text, state["chunks"], state["standalone_question"], state.get("validation_error")
        )
        res = llm.chat(
            [{"role": "system", "content": prompts.ANSWER_SYSTEM}, {"role": "user", "content": prompt}],
            schema=LLMAnswer.model_json_schema(),
        )
        attempt = state.get("attempts", 0) + 1
        return {
            "raw_answer": res.content,
            "attempts": attempt,
            "_trace": {
                "attempt": attempt,
                "model": res.model,
                "passages": len(state["chunks"]),
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            },
        }

    @node("validate")
    def validate(state: AgentState) -> dict:
        try:
            parsed = LLMAnswer.model_validate_json(state["raw_answer"])
        except ValidationError as exc:
            error = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'json'}: {e['msg']}" for e in exc.errors()[:3])
            if state.get("attempts", 0) < MAX_GENERATE_ATTEMPTS:
                return {"validation_error": error, "_trace": {"valid": False, "error": error, "action": "retry"}}
            answer = Answer(answer=prompts.INVALID_ANSWER, found=False, confidence="low")
            return _final(answer, {"valid": False, "error": error, "action": "give up"})

        chunks = state["chunks"]
        used = [n for n in dict.fromkeys(parsed.citations) if 1 <= n <= len(chunks)]
        citations = [
            Citation(
                filename=chunks[n - 1]["filename"],
                chunk_id=chunks[n - 1]["chunk_id"],
                page=chunks[n - 1].get("page"),
                score=chunks[n - 1]["score"],
                snippet=chunks[n - 1]["text"][:300],
            )
            for n in used
        ] if parsed.found else []
        confidence = parsed.confidence if (citations or not parsed.found) else "low"
        answer = Answer(answer=parsed.answer, found=parsed.found, confidence=confidence, citations=citations)
        return _final(
            answer,
            {"valid": True, "found": parsed.found, "citations": used, "invalid_citations": len(parsed.citations) - len(used)},
        )

    @node("refuse")
    def refuse(state: AgentState) -> dict:
        answer = Answer(answer=prompts.NOT_FOUND_ANSWER, found=False, confidence="high")
        best = max((c["score"] for c in state["retrieved"]), default=None)
        return _final(answer, {"reason": "no chunk above min_score", "best_score": best})

    def _final(answer: Answer, detail: dict) -> dict:
        return {
            "answer": answer.model_dump(),
            "validation_error": None,
            "messages": [AIMessage(content=answer.answer)],
            "_trace": detail,
        }

    def route_after_grade(state: AgentState) -> Literal["generate", "refuse"]:
        return "generate" if state["chunks"] else "refuse"

    def route_after_validate(state: AgentState) -> Literal["generate", "__end__"]:
        return "generate" if state.get("answer") is None else END

    g = StateGraph(AgentState)
    for name, fn in [
        ("contextualize", contextualize),
        ("retrieve", retrieve),
        ("grade", grade),
        ("generate", generate),
        ("validate", validate),
        ("refuse", refuse),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", route_after_validate, {"generate": "generate", END: END})
    g.add_edge("refuse", END)
    return g


def sqlite_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.ensure_dirs()
    conn = sqlite3.connect(settings.memory_db, check_same_thread=False)
    return SqliteSaver(conn)


class DocQAAgent:
    """Facade used by the UI, CLI, evaluation and tests."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        docs: DocumentTools,
        checkpointer: BaseCheckpointSaver | None = None,
        tracer: Tracer | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.tracer = tracer or Tracer(settings.traces_file)
        self.checkpointer = checkpointer if checkpointer is not None else sqlite_checkpointer(settings)
        self.graph = build_graph(settings, llm, docs, self.tracer).compile(checkpointer=self.checkpointer)

    def ask(self, question: str, thread_id: str = "default") -> AgentResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        trace = self.tracer.start(question, thread_id)
        turn: AgentState = {  # per-turn keys are reset; `messages` accumulates across turns
            "messages": [HumanMessage(content=question)],
            "question": question,
            "standalone_question": question,
            "retrieved": [],
            "chunks": [],
            "raw_answer": "",
            "attempts": 0,
            "validation_error": None,
            "answer": None,
            "trace_id": trace.trace_id,
        }
        try:
            final = self.graph.invoke(turn, config={"configurable": {"thread_id": thread_id}})
        except Exception as exc:
            self.tracer.finish(trace.trace_id, error=repr(exc))
            raise
        answer = Answer.model_validate(final["answer"])
        trace = self.tracer.finish(trace.trace_id, found=answer.found)
        return AgentResponse(
            thread_id=thread_id,
            question=question,
            standalone_question=final["standalone_question"],
            answer=answer,
            retrieved=[RetrievedChunk.model_validate(c) for c in final["retrieved"]],
            trace=trace,
        )

    def history(self, thread_id: str) -> list[dict]:
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = (state.values or {}).get("messages", [])
        return [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages]

    def clear(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        conn = getattr(self.checkpointer, "conn", None)
        if conn is not None:
            conn.close()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()


def graph_dot(route: list[str] | None = None) -> str:
    """Graphviz DOT of the workflow, highlighting the nodes/edges visited in ``route``."""
    route = route or []
    path = ["START", *route, "END"] if route else []
    visited_edges = set(zip(path, path[1:]))
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fontname=Helvetica, fillcolor="#f2f2f2"];']
    for n in ["START", *NODES, "END"]:
        shape = "ellipse" if n in ("START", "END") else "box"
        fill = "#8fd19e" if n in path else "#f2f2f2"
        count = route.count(n)
        label = f"{n} x{count}" if count > 1 else n
        lines.append(f'"{n}" [label="{label}", shape={shape}, fillcolor="{fill}"];')
    for a, b, label in EDGES:
        style = 'color="#2e8b57", penwidth=2.5' if (a, b) in visited_edges else 'color="#999999"'
        lines.append(f'"{a}" -> "{b}" [label="{label}", fontsize=10, {style}];')
    lines.append("}")
    return "\n".join(lines)
