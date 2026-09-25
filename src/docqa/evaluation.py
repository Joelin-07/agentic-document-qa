"""Evaluation over a small JSONL dataset.

Metrics (per item, then averaged):
  * retrieval_hit   - expected source file is among the retrieved chunks (answerable items)
  * citation_hit    - expected source file is among the cited chunks (answerable items)
  * keyword_score   - fraction of expected keywords present in the answer
  * correct         - answerable: found and keyword_score >= threshold; unanswerable: declined
  * judge_score     - optional 1-5 grade from the local LLM (LLM-as-judge)
  * latency_ms      - end-to-end time for the question
"""

from __future__ import annotations

import statistics
import uuid
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from docqa import prompts
from docqa.graph import DocQAAgent
from docqa.llm import LLMClient
from docqa.models import EvalItem, EvalItemResult, EvalReport, JudgeVerdict

KEYWORD_THRESHOLD = 0.5


def load_dataset(path: Path) -> list[EvalItem]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                items.append(EvalItem.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path}:{n}: invalid eval item\n{exc}") from exc
    return items


def keyword_score(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)


def judge(llm: LLMClient, item: EvalItem, answer: str) -> int | None:
    res = llm.chat(
        [
            {"role": "system", "content": prompts.JUDGE_SYSTEM},
            {"role": "user", "content": prompts.judge_prompt(item.question, item.expected_answer, answer)},
        ],
        schema=JudgeVerdict.model_json_schema(),
    )
    try:
        return JudgeVerdict.model_validate_json(res.content).score
    except ValidationError:
        return None


def run_evaluation(
    agent: DocQAAgent,
    items: list[EvalItem],
    *,
    use_judge: bool = False,
    progress: Callable[[int, int, EvalItemResult], None] | None = None,
) -> EvalReport:
    run_id = uuid.uuid4().hex[:8]
    results: list[EvalItemResult] = []
    for i, item in enumerate(items, start=1):
        thread = f"eval-{run_id}-{item.id}"  # fresh memory per item
        for earlier in item.context:
            agent.ask(earlier, thread_id=thread)
        resp = agent.ask(item.question, thread_id=thread)
        answer = resp.answer
        retrieved_files = {c.filename for c in resp.retrieved}
        cited_files = {c.filename for c in answer.citations}
        kw = keyword_score(answer.answer, item.expected_keywords)
        if item.answerable:
            retrieval_hit = item.expected_source in retrieved_files if item.expected_source else None
            citation_hit = item.expected_source in cited_files if item.expected_source else None
            correct = answer.found and kw >= KEYWORD_THRESHOLD
        else:
            retrieval_hit = citation_hit = None
            correct = not answer.found
        result = EvalItemResult(
            id=item.id,
            question=item.question,
            answer=answer.answer,
            expected_answer=item.expected_answer,
            answerable=item.answerable,
            found=answer.found,
            retrieval_hit=retrieval_hit,
            citation_hit=citation_hit,
            keyword_score=round(kw, 3),
            judge_score=judge(agent.llm, item, answer.answer) if use_judge else None,
            correct=correct,
            latency_ms=resp.trace.total_ms,
            trace_id=resp.trace.trace_id,
        )
        results.append(result)
        if progress:
            progress(i, len(items), result)
    return _aggregate(run_id, agent, results)


def _rate(values: list[bool | None]) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else 0.0


def _aggregate(run_id: str, agent: DocQAAgent, results: list[EvalItemResult]) -> EvalReport:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    judged = [r.judge_score for r in results if r.judge_score is not None]
    return EvalReport(
        run_id=run_id,
        chat_model=agent.llm.chat_model,
        embed_model=agent.llm.embed_model,
        num_items=len(results),
        accuracy=_rate([r.correct for r in results]),
        retrieval_hit_rate=_rate([r.retrieval_hit for r in answerable]),
        citation_accuracy=_rate([r.citation_hit for r in answerable]),
        refusal_accuracy=_rate([not r.found for r in unanswerable]),
        mean_keyword_score=round(statistics.fmean(r.keyword_score for r in answerable), 3) if answerable else 0.0,
        mean_judge_score=round(statistics.fmean(judged), 2) if judged else None,
        avg_latency_ms=round(statistics.fmean(r.latency_ms for r in results), 1) if results else 0.0,
        results=results,
    )


# --------------------------------------------------------------------------- reports


def save_report(report: EvalReport, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.strftime("%Y%m%d-%H%M%S")
    json_path = reports_dir / f"eval-{stamp}-{report.run_id}.json"
    md_path = json_path.with_suffix(".md")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(report_markdown(report), encoding="utf-8")
    return json_path, md_path


def latest_report(reports_dir: Path) -> EvalReport | None:
    files = sorted(Path(reports_dir).glob("eval-*.json")) if Path(reports_dir).exists() else []
    return EvalReport.model_validate_json(files[-1].read_text(encoding="utf-8")) if files else None


def summary_metrics(report: EvalReport) -> dict:
    return {
        "accuracy": report.accuracy,
        "retrieval_hit_rate": report.retrieval_hit_rate,
        "citation_accuracy": report.citation_accuracy,
        "refusal_accuracy": report.refusal_accuracy,
        "mean_keyword_score": report.mean_keyword_score,
        "mean_judge_score": report.mean_judge_score,
        "avg_latency_ms": report.avg_latency_ms,
    }


def report_markdown(report: EvalReport) -> str:
    lines = [
        f"# Evaluation report `{report.run_id}`",
        "",
        f"- Created: {report.created_at:%Y-%m-%d %H:%M:%S} UTC",
        f"- Models: chat `{report.chat_model}`, embeddings `{report.embed_model}`",
        f"- Items: {report.num_items}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary_metrics(report).items()]
    lines += ["", "| id | correct | found | retrieval | keywords | judge | ms | answer |", "|---|---|---|---|---|---|---|---|"]
    for r in report.results:
        answer = r.answer.replace("|", "/").replace("\n", " ")[:120]
        lines.append(
            f"| {r.id} | {'yes' if r.correct else 'NO'} | {r.found} | {r.retrieval_hit} | "
            f"{r.keyword_score} | {r.judge_score} | {r.latency_ms:.0f} | {answer} |"
        )
    return "\n".join(lines) + "\n"

