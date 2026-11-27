"""Chatbot-quality metrics (Module 4): faithfulness, relevancy, context precision/recall, "I don't know"."""

from __future__ import annotations

from typing import Any

from app.research.deps import Research
from app.research.kb import is_refusal
from evals import judges
from evals.report_metrics import ratio


def make_questions(report: dict[str, Any], off_topic: str) -> list[dict[str, Any]]:
    """3 answerable questions (with expected answers) + 2 unanswerable ones (off-topic and on-topic)."""
    reply = judges.judge(judges.QUESTIONS, report["markdown"].split("## References")[0])
    questions = [
        {"question": q["question"], "expected": q.get("answer", ""), "answerable": True}
        for q in reply.get("answerable", [])[:3]
        if isinstance(q, dict) and q.get("question")
    ]
    unanswerable = [off_topic, reply.get("unanswerable")]
    questions += [{"question": q, "expected": "", "answerable": False} for q in unanswerable if q]
    return questions


def ask(research: Research, report: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Ask one question in a fresh chat session and grade the answer."""
    report["chat_session_id"] = research.kb.sessions.create("eval").id  # no memory between test questions
    hits = research.kb.retriever_for(report["id"]).retrieve(item["question"])
    answer = research.kb.ask(report, item["question"])["answer"]
    row = {**item, "answer": answer, "refused": is_refusal(answer), "chunks": len(hits)}
    if item["answerable"] and not row["refused"]:
        context = "\n\n".join(f"Chunk {i}: {h.chunk.text}" for i, h in enumerate(hits, start=1))
        user = f"QUESTION: {item['question']}\nEXPECTED: {item['expected']}\nCONTEXT:\n{context}\n\nANSWER: {answer}"
        row["grade"] = judges.judge(judges.CHAT, user)
    return row


def mean(values: list[float]) -> float | None:
    return ratio(sum(values), len(values))


def chat_metrics(research: Research, report: dict[str, Any], off_topic: str) -> dict[str, Any]:
    research.kb.ingest(report)
    rows = [ask(research, report, item) for item in make_questions(report, off_topic)]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    grades = [r["grade"] for r in answerable if r.get("grade")]
    precision = [
        sum(map(bool, g["chunk_relevant"])) / len(g["chunk_relevant"]) for g in grades if g.get("chunk_relevant")
    ]
    return {
        "questions": len(rows),
        "faithfulness": mean([float(g.get("faithful", 0)) for g in grades]),
        "answer_relevancy": mean([float(g.get("relevant", 0)) for g in grades]),
        "answer_correct": ratio(sum(bool(g.get("correct")) for g in grades), len(answerable)),
        "context_precision": mean(precision),
        "context_recall": mean([float(bool(g.get("context_complete"))) for g in grades]),
        "idk_correct": ratio(sum(r["refused"] for r in unanswerable), len(unanswerable)),
        "false_refusals": ratio(sum(r["refused"] for r in answerable), len(answerable)),
        "rows": rows,
    }
