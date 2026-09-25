"""LangGraph workflow: routing, validation retry, memory, tracing."""

import json

from langgraph.checkpoint.memory import InMemorySaver

from docqa.graph import DocQAAgent, graph_dot, sqlite_checkpointer
from docqa.llm import FakeLLM
from docqa.observability import Tracer, load_traces
from docqa.prompts import INVALID_ANSWER, NOT_FOUND_ANSWER


def test_answerable_question_takes_generate_path(agent):
    resp = agent.ask("How many days of paid time off do full-time employees get per year?")
    assert resp.trace.route == ["contextualize", "retrieve", "grade", "generate", "validate"]
    assert resp.answer.found
    assert "25 days" in resp.answer.answer
    assert resp.answer.citations[0].filename == "employee_handbook.md"
    assert resp.retrieved and resp.retrieved[0].score >= resp.retrieved[-1].score


def test_irrelevant_question_is_refused(agent):
    resp = agent.ask("What is the CEO's favourite programming language?")
    assert resp.trace.route == ["contextualize", "retrieve", "grade", "refuse"]
    assert not resp.answer.found
    assert resp.answer.answer == NOT_FOUND_ANSWER


def test_invalid_llm_output_is_retried_then_accepted(settings, service):
    valid = json.dumps({"answer": "It is 14 days.", "found": True, "citations": [1], "confidence": "high"})
    llm = FakeLLM(scripted=['{"answer": "oops"', valid])  # first reply is broken JSON
    agent = DocQAAgent(settings, llm, docs=service, checkpointer=InMemorySaver())
    resp = agent.ask("How long is the free trial of the Pro plan?")
    assert resp.trace.route.count("generate") == 2
    assert resp.answer.answer == "It is 14 days."
    first_validate = [s for s in resp.trace.spans if s.node == "validate"][0]
    assert first_validate.detail["action"] == "retry"
    # the validation error is fed back to the model on the retry
    assert "previous reply was invalid" in llm.calls[-1]["messages"][-1]["content"]


def test_gives_up_after_two_invalid_outputs(settings, service):
    llm = FakeLLM(scripted=["nope", '{"found": true}'])
    agent = DocQAAgent(settings, llm, docs=service, checkpointer=InMemorySaver())
    resp = agent.ask("How long is the free trial of the Pro plan?")
    assert resp.answer.answer == INVALID_ANSWER and not resp.answer.found


def test_out_of_range_citations_are_dropped(settings, service):
    llm = FakeLLM(scripted=[json.dumps({"answer": "x", "found": True, "citations": [1, 99], "confidence": "high"})])
    agent = DocQAAgent(settings, llm, docs=service, checkpointer=InMemorySaver())
    resp = agent.ask("How long is the free trial of the Pro plan?")
    assert len(resp.answer.citations) == 1


def test_memory_rewrites_follow_up_questions(agent, fake_llm):
    first = agent.ask("How much does the Pro plan cost?", thread_id="t1")
    assert first.standalone_question == first.question  # no history: no rewrite
    follow = agent.ask("And how much storage does it include per user?", thread_id="t1")
    assert follow.standalone_question != follow.question
    assert "Pro plan" in follow.standalone_question
    assert "100 GB" in follow.answer.answer
    rewrite_call = [c for c in fake_llm.calls if c["schema"] is None][-1]
    assert "user: How much does the Pro plan cost?" in rewrite_call["messages"][-1]["content"]
    history = agent.history("t1")
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]


def test_threads_are_isolated_and_clearable(agent):
    agent.ask("How much does the Pro plan cost?", thread_id="a")
    assert agent.history("b") == []
    agent.clear("a")
    assert agent.history("a") == []


def test_memory_persists_across_agent_instances(settings, fake_llm, service):
    a1 = DocQAAgent(settings, fake_llm, docs=service, checkpointer=sqlite_checkpointer(settings))
    a1.ask("How much does the Pro plan cost?", thread_id="persist")
    a2 = DocQAAgent(settings, fake_llm, docs=service, checkpointer=sqlite_checkpointer(settings))
    assert len(a2.history("persist")) == 2
    assert a2.ask("And how much storage does it include per user?", thread_id="persist").standalone_question != (
        "And how much storage does it include per user?"
    )


def test_traces_are_written(settings, fake_llm, service):
    agent = DocQAAgent(settings, fake_llm, docs=service, checkpointer=InMemorySaver(), tracer=Tracer(settings.traces_file))
    resp = agent.ask("How often must API keys be rotated?")
    [trace] = load_traces(settings.traces_file)
    assert trace.trace_id == resp.trace.trace_id
    assert trace.prompt_tokens > 0 and trace.answer_found is True
    retrieve = next(s for s in trace.spans if s.node == "retrieve")
    assert retrieve.detail["tool"] == "mcp:search_documents"
    assert all(s.duration_ms >= 0 for s in trace.spans)


def test_graph_visualisations(agent):
    assert "contextualize" in agent.mermaid()
    dot = graph_dot(["contextualize", "retrieve", "grade", "refuse"])
    assert '"refuse" [label="refuse", shape=box, fillcolor="#8fd19e"]' in dot
    assert '"generate" [label="generate", shape=box, fillcolor="#f2f2f2"]' in dot
