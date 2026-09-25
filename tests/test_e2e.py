"""End-to-end: Streamlit UI / CLI -> LangGraph agent -> MCP server subprocess -> vector store.

Uses the fake LLM backend so it runs without Ollama. ``test_live_ollama`` repeats the
core flow against a real Ollama server and is skipped when Ollama is not available.
"""

import pytest
from typer.testing import CliRunner

from docqa.config import PROJECT_ROOT, Settings
from docqa.runtime import build_runtime


@pytest.fixture
def fake_env(settings, monkeypatch):
    """Make every Settings() in this process (CLI, Streamlit) use the temp fake setup."""
    monkeypatch.chdir(settings.data_dir.parent)  # avoid picking up a developer's .env
    monkeypatch.setenv("DOCQA_LLM_BACKEND", "fake")
    monkeypatch.setenv("DOCQA_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("DOCQA_MIN_SCORE", str(settings.min_score))
    return Settings()


def test_full_pipeline_over_mcp(fake_env):
    rt = build_runtime(fake_env)
    try:
        results = rt.ingest_folder(fake_env.demo_docs_dir)
        assert {r.status for r in results} == {"added"}

        resp = rt.agent.ask("When will the legacy REST API v1 be shut down?", thread_id="e2e")
        assert resp.answer.found and "31 December 2026" in resp.answer.answer
        assert resp.answer.citations[0].filename == "release_notes.md"

        follow = rt.agent.ask("How much does the Pro plan cost?", thread_id="e2e")
        assert follow.standalone_question != follow.question  # memory used on the 2nd turn
        assert len(rt.agent.history("e2e")) == 4
    finally:
        rt.close()


def test_cli_commands(fake_env):
    from docqa.cli import app

    runner = CliRunner()
    out = runner.invoke(app, ["ingest"])
    assert out.exit_code == 0, out.output
    assert "added" in out.output and "security_policy.pdf" in out.output

    out = runner.invoke(app, ["ask", "How often must API keys be rotated?", "--thread", "cli-test"])
    assert out.exit_code == 0, out.output
    assert "90 days" in out.output and "security_policy.pdf" in out.output

    out = runner.invoke(app, ["eval", "--limit", "3"])
    assert out.exit_code == 0, out.output
    assert "retrieval_hit_rate" in out.output

    out = runner.invoke(app, ["traces", "--last", "3"])
    assert out.exit_code == 0 and "avg_latency_ms" in out.output

    out = runner.invoke(app, ["graph"])
    assert out.exit_code == 0 and "contextualize" in out.output


def test_streamlit_app_end_to_end(fake_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5

    next(b for b in at.button if b.label == "Load demo docs").click().run()
    assert not at.exception
    assert any("employee_handbook.md" in m.value for m in at.markdown)

    at.chat_input[0].set_value("How long is parental leave at Nimbus Labs?").run()
    assert not at.exception
    assert any("16 weeks" in m.value for m in at.markdown)
    assert any("1 turns" in m.value for m in at.markdown)  # memory indicator
    assert at.session_state["last"].trace.route[-1] == "validate"


def _ollama_ready() -> bool:
    try:
        from docqa.llm import OllamaLLM

        h = OllamaLLM(Settings(_env_file=None)).health()
        return h["chat_model_ok"] and h["embed_model_ok"]
    except Exception:
        return False


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_ready(), reason="Ollama not running or models not pulled")
def test_live_ollama(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    rt = build_runtime(settings)
    try:
        rt.ingest_folder(settings.demo_docs_dir)
        resp = rt.agent.ask("How many days of paid time off do employees get per year?", thread_id="live")
        assert resp.answer.found and "25" in resp.answer.answer
        refused = rt.agent.ask("What is the stock price of Nimbus Labs?", thread_id="live-2")
        assert not refused.answer.found
    finally:
        rt.close()
