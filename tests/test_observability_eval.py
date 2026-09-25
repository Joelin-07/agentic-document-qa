from docqa.evaluation import keyword_score, latest_report, load_dataset, report_markdown, run_evaluation, save_report
from docqa.models import Span, Trace, utcnow
from docqa.observability import Tracer, load_traces, summarize

from .conftest import EVAL_SET


def test_tracer_node_decorator_records_spans_and_strips_trace_key(tmp_path):
    tracer = Tracer(tmp_path / "t.jsonl")
    trace = tracer.start("q", "thread")

    @tracer.node("step")
    def step(state):
        return {"x": 1, "_trace": {"info": "hi", "prompt_tokens": 5, "completion_tokens": 2}}

    assert step({"trace_id": trace.trace_id}) == {"x": 1}
    done = tracer.finish(trace.trace_id, found=True)
    assert done.route == ["step"] and done.prompt_tokens == 5
    assert done.spans[0].detail == {"info": "hi", "tokens": {"prompt": 5, "completion": 2}}
    assert load_traces(tmp_path / "t.jsonl")[0].trace_id == trace.trace_id


def test_tracer_records_errors(tmp_path):
    tracer = Tracer(None)
    trace = tracer.start("q", "t")

    @tracer.node("boom")
    def boom(state):
        raise RuntimeError("bad")

    try:
        boom({"trace_id": trace.trace_id})
    except RuntimeError:
        pass
    done = tracer.finish(trace.trace_id, error="bad")
    assert done.status == "error" and done.spans[0].status == "error"


def test_summarize():
    def t(ms, route, found):
        return Trace(trace_id="x", thread_id="t", question="q", total_ms=ms, route=route, answer_found=found,
                     spans=[Span(node=n, started_at=utcnow(), duration_ms=10) for n in route])

    s = summarize([t(100, ["a", "generate"], True), t(300, ["a", "generate", "generate"], False)])
    assert s["requests"] == 2 and s["avg_latency_ms"] == 200 and s["retries"] == 1
    assert s["not_found_rate"] == 0.5 and s["avg_node_ms"] == {"a": 10, "generate": 10}
    assert summarize([]) == {"requests": 0}


def test_load_traces_skips_corrupt_lines(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text("{not json}\n" + Trace(trace_id="ok", thread_id="t", question="q").model_dump_json() + "\n")
    assert [t.trace_id for t in load_traces(f)] == ["ok"]


def test_keyword_score():
    assert keyword_score("It is 25 days.", ["25"]) == 1.0
    assert keyword_score("It is 25 days.", ["25", "PTO"]) == 0.5
    assert keyword_score("anything", []) == 1.0


def test_run_evaluation_with_fake_llm(agent, settings):
    items = load_dataset(EVAL_SET)
    seen = []
    report = run_evaluation(agent, items, use_judge=True, progress=lambda i, n, r: seen.append(i))
    assert seen == list(range(1, len(items) + 1))
    assert report.num_items == len(items)
    assert report.retrieval_hit_rate >= 0.8  # even bag-of-words retrieval finds the right document
    assert report.accuracy >= 0.6
    assert report.mean_judge_score == 4  # fake judge always returns 4
    q13 = next(r for r in report.results if r.id == "q13")
    assert q13.correct and not q13.found  # unanswerable question refused

    json_path, md_path = save_report(report, settings.eval_reports_dir)
    assert json_path.exists() and "| accuracy |" in md_path.read_text()
    assert latest_report(settings.eval_reports_dir).run_id == report.run_id
    assert "q01" in report_markdown(report)
