"""LLM access: a thin Ollama wrapper plus a deterministic fake for tests.

Both implement the same two calls:
  * ``chat(messages, schema=None)`` -> ChatResult  (schema = JSON schema for structured output)
  * ``embed(texts, kind)``          -> list of vectors
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from docqa.config import Settings

EmbedKind = Literal["document", "query"]


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class LLMClient(Protocol):
    chat_model: str
    embed_model: str

    def chat(self, messages: list[dict], schema: dict | None = None) -> ChatResult: ...

    def embed(self, texts: list[str], kind: EmbedKind = "document") -> list[list[float]]: ...


# --------------------------------------------------------------------------- Ollama


class OllamaLLM:
    def __init__(self, settings: Settings):
        import ollama

        self.chat_model = settings.chat_model
        self.embed_model = settings.embed_model
        self.temperature = settings.temperature
        self.client = ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout)

    def chat(self, messages: list[dict], schema: dict | None = None) -> ChatResult:
        resp = self.client.chat(
            model=self.chat_model,
            messages=messages,
            format=schema,
            options={"temperature": self.temperature},
        )
        return ChatResult(
            content=resp.message.content or "",
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=resp.eval_count or 0,
            model=self.chat_model,
        )

    def embed(self, texts: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        if not texts:
            return []
        # nomic-embed-text is trained with task prefixes; they noticeably improve retrieval.
        if "nomic" in self.embed_model:
            prefix = "search_query: " if kind == "query" else "search_document: "
            texts = [prefix + t for t in texts]
        resp = self.client.embed(model=self.embed_model, input=texts)
        return [list(v) for v in resp.embeddings]

    def health(self) -> dict:
        """Return which configured models are available locally (raises if Ollama is unreachable)."""
        names = {m.model for m in self.client.list().models}

        def present(model: str) -> bool:
            return model in names or f"{model}:latest" in names

        return {
            "reachable": True,
            "chat_model": self.chat_model,
            "chat_model_ok": present(self.chat_model),
            "embed_model": self.embed_model,
            "embed_model_ok": present(self.embed_model),
        }


# --------------------------------------------------------------------------- Fake (tests)

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(
    "a an and are as at be by can do does for from how i in is it of on or our the this to "
    "what when where which who why will with you your we us my me that there".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


class FakeLLM:
    """Deterministic, dependency-free stand-in for Ollama.

    * embeddings: hashed bag-of-words (so similar wording => high cosine similarity)
    * chat: picks the context sentence with the largest word overlap with the question.
    ``scripted`` responses (if given) are returned first, in order, for chat calls with a schema.
    """

    chat_model = "fake-chat"
    embed_model = "fake-embed"
    dim = 4096

    def __init__(self, scripted: list[str] | None = None):
        self.scripted = list(scripted or [])
        self.calls: list[dict] = []

    def embed(self, texts: list[str], kind: EmbedKind = "document") -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in tokenize(text):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vec[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def chat(self, messages: list[dict], schema: dict | None = None) -> ChatResult:
        self.calls.append({"messages": messages, "schema": schema})
        prompt = messages[-1]["content"]
        if schema is not None and self.scripted:
            return ChatResult(self.scripted.pop(0), 10, 10, self.chat_model)
        title = (schema or {}).get("title")
        if title == "LLMAnswer":
            return ChatResult(self._answer(prompt), 50, 20, self.chat_model)
        if title == "JudgeVerdict":
            return ChatResult(json.dumps({"score": 4, "reason": "fake judge"}), 30, 10, self.chat_model)
        # Plain-text call: query rewrite. Append the previous user question so follow-ups keep context.
        question = _between(prompt, "Follow-up question:", None).strip()
        previous = re.findall(r"^user: (.+)$", prompt, flags=re.M)
        rewritten = f"{question} {previous[-1]}" if previous else question
        return ChatResult(rewritten, 20, 10, self.chat_model)

    @staticmethod
    def _answer(prompt: str) -> str:
        question = set(tokenize(_between(prompt, "Question:", None)))
        passages = re.findall(r"^\[(\d+)\][^\n]*\n(.+?)(?=^\[\d+\]|^Question:|\Z)", prompt, flags=re.M | re.S)
        best = (0, None, "")
        for num, text in passages:
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
                overlap = len(question & set(tokenize(sentence)))
                if overlap > best[0]:
                    best = (overlap, int(num), sentence.strip())
        if best[1] is None:
            payload = {"answer": "I don't know based on the provided documents.", "found": False, "citations": [], "confidence": "low"}
        else:
            payload = {"answer": best[2], "found": True, "citations": [best[1]], "confidence": "high"}
        return json.dumps(payload)


def _between(text: str, start: str, end: str | None) -> str:
    i = text.rfind(start)
    if i < 0:
        return text
    rest = text[i + len(start):]
    if end and end in rest:
        rest = rest[: rest.index(end)]
    return rest


def get_llm(settings: Settings) -> LLMClient:
    if settings.llm_backend == "fake":
        return FakeLLM()
    return OllamaLLM(settings)
