import pytest
from pydantic import ValidationError

from docqa.config import Settings
from docqa.evaluation import load_dataset
from docqa.models import Answer, EvalItem, LLMAnswer

from .conftest import EVAL_SET


def test_settings_defaults_and_env_override(monkeypatch):
    monkeypatch.setenv("DOCQA_TOP_K", "7")
    s = Settings(_env_file=None)
    assert s.top_k == 7
    assert s.chat_model == "llama3.2:3b"
    assert s.memory_db.name == "memory.sqlite"


def test_settings_rejects_invalid_values():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, top_k=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_backend="openai")


def test_llm_answer_parsing():
    a = LLMAnswer.model_validate_json('{"answer": "25 days", "found": true, "citations": [1], "confidence": "high"}')
    assert a.citations == [1]
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate_json('{"answer": "x", "found": true, "confidence": "certain"}')
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate_json("not json")
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate_json('{"answer": "", "found": true}')


def test_llm_answer_schema_is_json_schema():
    schema = LLMAnswer.model_json_schema()
    assert schema["title"] == "LLMAnswer"
    assert set(schema["required"]) == {"answer", "found"}


def test_answer_defaults():
    assert Answer(answer="x", found=False, confidence="low").citations == []


def test_eval_dataset_is_valid():
    items = load_dataset(EVAL_SET)
    assert 10 <= len(items) <= 20
    assert len({i.id for i in items}) == len(items)
    assert any(not i.answerable for i in items)
    assert any(i.context for i in items)
    for i in items:
        if i.answerable:
            assert i.expected_source and i.expected_keywords


def test_eval_item_validation():
    with pytest.raises(ValidationError):
        EvalItem.model_validate({"id": "x", "question": "q"})  # expected_answer missing
