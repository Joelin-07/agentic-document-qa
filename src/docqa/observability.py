"""Lightweight tracing: per-node spans for each question, written as JSON lines.

Every question produces one ``Trace`` (route taken, per-node latency, token usage,
retrieval scores, errors), appended to ``data/traces.jsonl``. ``summarize`` turns
that file into the metrics shown in the UI and CLI.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from docqa.models import Span, Trace, utcnow

log = logging.getLogger("docqa.trace")


class Tracer:
    def __init__(self, traces_file: Path | None):
        self.traces_file = traces_file
        self._active: dict[str, Trace] = {}
        self._lock = threading.Lock()

    def start(self, question: str, thread_id: str) -> Trace:
        trace = Trace(trace_id=uuid.uuid4().hex[:12], thread_id=thread_id, question=question)
        with self._lock:
            self._active[trace.trace_id] = trace
        return trace

    def record(self, trace_id: str, span: Span, tokens: tuple[int, int] = (0, 0)) -> None:
        trace = self._active.get(trace_id)
        if trace is None:
            return
        trace.spans.append(span)
        trace.route.append(span.node)
        trace.prompt_tokens += tokens[0]
        trace.completion_tokens += tokens[1]
        log.info("trace=%s node=%s %.0fms %s", trace_id, span.node, span.duration_ms, span.status)

    def finish(self, trace_id: str, *, found: bool | None = None, error: str | None = None) -> Trace:
        with self._lock:
            trace = self._active.pop(trace_id)
        trace.total_ms = round((utcnow() - trace.started_at).total_seconds() * 1000, 1)
        trace.answer_found = found
        if error:
            trace.status, trace.error = "error", error
        if self.traces_file:
            self.traces_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.traces_file.open("a", encoding="utf-8") as f:
                f.write(trace.model_dump_json() + "\n")
        return trace

    def node(self, name: str) -> Callable:
        """Decorator for LangGraph nodes.

        Times the node and records a span. A node may return an extra ``_trace`` dict
        (details for the span, e.g. scores or token counts); it is removed from the
        state update before LangGraph sees it.
        """

        def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
            @wraps(fn)
            def wrapper(state: dict) -> dict:
                started, t0 = utcnow(), time.perf_counter()
                try:
                    update = fn(state) or {}
                except Exception as exc:
                    self.record(
                        state.get("trace_id", ""),
                        Span(node=name, started_at=started, duration_ms=_ms(t0), status="error", error=repr(exc)),
                    )
                    raise
                detail: dict[str, Any] = update.pop("_trace", {})
                tokens = (detail.pop("prompt_tokens", 0), detail.pop("completion_tokens", 0))
                if any(tokens):
                    detail["tokens"] = {"prompt": tokens[0], "completion": tokens[1]}
                self.record(
                    state.get("trace_id", ""),
                    Span(node=name, started_at=started, duration_ms=_ms(t0), detail=detail),
                    tokens,
                )
                return update

            return wrapper

        return decorator


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


# --------------------------------------------------------------------------- reading / metrics


def load_traces(traces_file: Path, last: int | None = None) -> list[Trace]:
    if not traces_file.exists():
        return []
    lines = [ln for ln in traces_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if last:
        lines = lines[-last:]
    traces = []
    for ln in lines:
        try:
            traces.append(Trace.model_validate_json(ln))
        except ValueError:
            continue  # skip corrupted lines rather than failing the dashboard
    return traces


def summarize(traces: list[Trace]) -> dict[str, Any]:
    if not traces:
        return {"requests": 0}
    latencies = sorted(t.total_ms for t in traces)
    per_node: dict[str, list[float]] = {}
    for t in traces:
        for s in t.spans:
            per_node.setdefault(s.node, []).append(s.duration_ms)
    return {
        "requests": len(traces),
        "errors": sum(t.status == "error" for t in traces),
        "not_found_rate": round(sum(t.answer_found is False for t in traces) / len(traces), 3),
        "avg_latency_ms": round(statistics.fmean(latencies), 1),
        "p95_latency_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "prompt_tokens": sum(t.prompt_tokens for t in traces),
        "completion_tokens": sum(t.completion_tokens for t in traces),
        "retries": sum(t.route.count("generate") > 1 for t in traces),
        "avg_node_ms": {n: round(statistics.fmean(v), 1) for n, v in per_node.items()},
    }

