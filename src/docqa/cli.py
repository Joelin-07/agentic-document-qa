"""Debug/test CLI. The primary demo interface is the Streamlit app (app.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from docqa.config import Settings

app = typer.Typer(add_completion=False, help="Local AI Document Assistant (debug CLI)")
console = Console()


def _runtime():
    from docqa.runtime import build_runtime

    return build_runtime(Settings())


def _print_answer(resp) -> None:
    a = resp.answer
    if resp.standalone_question != resp.question:
        console.print(f"[dim]rewritten with memory:[/dim] {resp.standalone_question}")
    console.print(f"[bold green]Answer[/bold green] ({a.confidence}, found={a.found}): {a.answer}")
    for c in a.citations:
        page = f" p.{c.page}" if c.page else ""
        console.print(f"  [cyan]- {c.filename}{page}[/cyan] [{c.chunk_id}] score={c.score}")
    console.print(f"[dim]route: {' -> '.join(resp.trace.route)} | {resp.trace.total_ms:.0f} ms | trace {resp.trace.trace_id}[/dim]")


@app.command()
def doctor() -> None:
    """Check configuration and that Ollama and the models are available."""
    settings = Settings()
    console.print(settings.model_dump(mode="json"))
    if settings.llm_backend == "fake":
        console.print("[yellow]Fake LLM backend: Ollama is not used.[/yellow]")
        return
    from docqa.llm import OllamaLLM

    try:
        health = OllamaLLM(settings).health()
    except Exception as exc:
        console.print(f"[red]Ollama not reachable at {settings.ollama_host}: {exc}[/red]")
        raise typer.Exit(1)
    console.print(health)
    missing = [health[m] for m in ("chat_model", "embed_model") if not health[f"{m}_ok"]]
    if missing:
        console.print(f"[red]Missing models. Run: {'; '.join(f'ollama pull {m}' for m in missing)}[/red]")
        raise typer.Exit(1)
    console.print("[green]All good.[/green]")


@app.command()
def ingest(paths: list[Path] = typer.Argument(None, help="Files or folders (default: demo_data/docs)")) -> None:
    """Ingest documents through the MCP server."""
    rt = _runtime()
    try:
        targets = paths or [rt.settings.demo_docs_dir]
        for p in targets:
            results = rt.ingest_folder(p) if p.is_dir() else [rt.mcp.ingest(str(p))]
            for r in results:
                console.print(f"{r.status:>9}  {r.document.filename}  ({r.document.num_chunks} chunks)")
    finally:
        rt.close()


@app.command()
def docs() -> None:
    """List indexed documents (via MCP list_documents)."""
    rt = _runtime()
    try:
        table = Table("doc_id", "filename", "type", "chunks", "chars")
        for d in rt.mcp.list_documents():
            table.add_row(d.doc_id, d.filename, d.file_type, str(d.num_chunks), str(d.num_chars))
        console.print(table)
    finally:
        rt.close()


@app.command()
def ask(question: str, thread: str = typer.Option("cli", help="Conversation id (memory)")) -> None:
    """Ask one question."""
    rt = _runtime()
    try:
        _print_answer(rt.agent.ask(question, thread_id=thread))
    finally:
        rt.close()


@app.command()
def chat(thread: str = typer.Option("cli", help="Conversation id (memory)")) -> None:
    """Interactive chat with memory. Commands: /history, /clear, /exit."""
    rt = _runtime()
    console.print(f"Thread [bold]{thread}[/bold] - {len(rt.agent.history(thread))} messages in memory. /exit to quit.")
    try:
        while True:
            q = console.input("[bold]you> [/bold]").strip()
            if q in ("/exit", "/quit"):
                break
            if q == "/history":
                for m in rt.agent.history(thread):
                    console.print(f"[dim]{m['role']}:[/dim] {m['content']}")
                continue
            if q == "/clear":
                rt.agent.clear(thread)
                console.print("Memory cleared.")
                continue
            if q:
                _print_answer(rt.agent.ask(q, thread_id=thread))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rt.close()


@app.command("eval")
def evaluate(
    dataset: Optional[Path] = typer.Option(None, help="JSONL dataset (default: demo_data/eval/eval_set.jsonl)"),
    judge: bool = typer.Option(False, help="Also grade answers with the local LLM (slower)"),
    limit: int = typer.Option(0, help="Only run the first N items"),
) -> None:
    """Run the evaluation dataset and write a report to data/eval_reports/."""
    from docqa.evaluation import load_dataset, run_evaluation, save_report, summary_metrics

    rt = _runtime()
    try:
        rt.ingest_folder(rt.settings.demo_docs_dir)
        items = load_dataset(dataset or rt.settings.eval_dataset)
        items = items[:limit] if limit else items

        def progress(i, n, r):
            mark = "[green]ok[/green]" if r.correct else "[red]FAIL[/red]"
            console.print(f"[{i}/{n}] {mark} {r.id}: {r.answer[:90]}")

        report = run_evaluation(rt.agent, items, use_judge=judge, progress=progress)
        json_path, md_path = save_report(report, rt.settings.eval_reports_dir)
        table = Table("metric", "value")
        for k, v in summary_metrics(report).items():
            table.add_row(k, str(v))
        console.print(table)
        console.print(f"Report: {md_path}")
    finally:
        rt.close()


@app.command()
def traces(last: int = typer.Option(10, help="Number of recent traces")) -> None:
    """Show recent traces and aggregate metrics (observability)."""
    from docqa.observability import load_traces, summarize

    settings = Settings()
    items = load_traces(settings.traces_file, last)
    table = Table("trace", "thread", "question", "route", "ms", "tokens", "found")
    for t in items:
        table.add_row(
            t.trace_id, t.thread_id, t.question[:40], " > ".join(t.route), f"{t.total_ms:.0f}",
            str(t.prompt_tokens + t.completion_tokens), str(t.answer_found),
        )
    console.print(table)
    console.print_json(json.dumps(summarize(load_traces(settings.traces_file))))


@app.command()
def graph() -> None:
    """Print the LangGraph workflow as Mermaid."""
    from docqa.graph import DocQAAgent
    from docqa.llm import FakeLLM
    from langgraph.checkpoint.memory import InMemorySaver

    console.print(DocQAAgent(Settings(), FakeLLM(), docs=None, checkpointer=InMemorySaver()).mermaid())


if __name__ == "__main__":
    app()
