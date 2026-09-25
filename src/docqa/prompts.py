"""Prompt templates (kept short and explicit for small local models)."""

CONTEXTUALIZE_SYSTEM = (
    "You rewrite follow-up questions. Given a conversation and a follow-up question, rewrite the "
    "follow-up into a single standalone question that can be understood without the conversation. "
    "Replace pronouns like 'it', 'that' or 'they' with what they refer to. Keep names, numbers and "
    "terms exactly. If the question is already standalone, return it unchanged. "
    "Do NOT answer the question. Reply with the rewritten question only."
)

ANSWER_SYSTEM = (
    "You are a careful document assistant. Answer the question using ONLY the numbered context "
    "passages. Rules:\n"
    "- Answer in one or two complete sentences that restate what is being asked about, e.g. "
    "\"The cafeteria is open from 8:00 to 14:00 on weekdays.\" Never reply with just a number.\n"
    "- Include exact figures, dates and names from the context.\n"
    "- In 'citations', list the numbers of the passages you used.\n"
    "- If the passages do not contain the answer, set found=false, citations=[] and answer "
    "\"I don't know based on the provided documents.\"\n"
    "- Never use outside knowledge.\n"
    "Respond with JSON only."
)

NOT_FOUND_ANSWER = "I couldn't find anything relevant to that in the loaded documents."
INVALID_ANSWER = "Sorry, the model did not return a valid answer. Please try rephrasing the question."

JUDGE_SYSTEM = (
    "You grade answers from a document QA system. Compare the system answer with the reference "
    "answer. Score 5 if it contains the same key facts, 3 if partially correct, 1 if wrong or "
    "missing. If the reference says the information is not available and the system also declines, "
    "score 5. Respond with JSON only."
)


def format_history(messages: list, limit: int) -> str:
    lines = []
    for m in messages[-limit:] if limit else []:
        role = "user" if m.type == "human" else "assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def contextualize_prompt(history: str, question: str) -> str:
    return f"Conversation:\n{history}\n\nFollow-up question: {question}"


def answer_prompt(history: str, passages: list[dict], question: str, feedback: str | None = None) -> str:
    parts = []
    if history:
        parts.append(f"Conversation so far (for reference only):\n{history}\n")
    parts.append("Context passages:")
    for i, p in enumerate(passages, start=1):
        page = f", page {p['page']}" if p.get("page") else ""
        parts.append(f"[{i}] (source: {p['filename']}{page})\n{p['text']}\n")
    if feedback:
        parts.append(f"Your previous reply was invalid ({feedback}). Reply with valid JSON matching the schema.\n")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def judge_prompt(question: str, reference: str, answer: str) -> str:
    return f"Question: {question}\nReference answer: {reference}\nSystem answer: {answer}"
