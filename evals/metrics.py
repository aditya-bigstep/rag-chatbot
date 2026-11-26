"""The metrics from the project brief, computed for one finished report."""

from __future__ import annotations

from collections import Counter
from typing import Any

import httpx

from app.research.reader import BROWSER_AGENT
from app.research.render import number_sources, section_text
from evals import judges

OUT_OF_SCOPE = [
    "What is the boiling point of tungsten?",
    "Who won the football world cup in 1998?",
]


def url_alive(url: str) -> bool:
    try:
        response = httpx.get(url, headers={"User-Agent": BROWSER_AGENT}, timeout=10, follow_redirects=True)
        return response.status_code < 400
    except httpx.HTTPError:
        return False


def report_text(report: dict[str, Any]) -> str:
    numbers = number_sources(report)
    parts = []
    for item in report["outline"]:
        section = report["sections"].get(item["id"])
        if section and section["text"]:
            parts.append(f"## {section['title']}\n{section_text(section, numbers)}")
    return "\n\n".join(parts)


def report_metrics(report: dict[str, Any], aspects: list[str], judge: Any) -> dict[str, Any]:
    sources = report["sources"]
    cited = list(number_sources(report))
    alive = [url_alive(sources[sid]["url"]) for sid in cited]

    # Every sentence in the final report, with the evidence passages of the sources it cites.
    pairs, has_citation = [], []
    for section in report["sections"].values():
        for claim in section["claims"]:
            has_citation.append(bool(claim["source_ids"]))
            passages = [p["text"] for p in section["evidence"] if p["source_id"] in claim["source_ids"]]
            pairs.append((claim["text"], " ... ".join(passages[:6])))
    supported = judges.judge_claims(judge, pairs) if pairs else []
    # A sentence with no citation cannot be grounded, whatever the judge thinks.
    grounded = [ok and cited_ok for ok, cited_ok in zip(supported, has_citation, strict=True)]

    text = report_text(report)
    covered = judges.judge_coverage(judge, text, aspects) if text else [False] * len(aspects)
    rubric = judges.judge_readability(judge, text) if text else None
    domains = Counter(sources[sid]["domain"] for sid in cited)

    return {
        "sentences": len(pairs),
        "sources": len(cited),
        "citation_url_valid": share(alive),
        "citation_supports_claim": share([g for g, c in zip(grounded, has_citation, strict=True) if c]),
        "groundedness": share(grounded),
        "hallucination_rate": round(1 - share(grounded), 3) if grounded else None,
        "uncited_sentences": has_citation.count(False),
        "coverage": share(covered),
        "source_diversity": round(len(domains) / len(cited), 3) if cited else 0.0,
        "top_domain_share": round(max(domains.values()) / len(cited), 3) if cited else 0.0,
        "readability": round((rubric.clarity + rubric.organisation + rubric.flow) / 15, 3) if rubric else 0.0,
        "readability_comment": rubric.comment if rubric else "",
        "empty_sections": sum(1 for s in report["sections"].values() if not s["text"]),
    }


def share(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 3) if flags else 0.0


def chat_metrics(report: dict[str, Any], engine: Any, judge: Any, per_report: int = 2) -> dict[str, Any]:
    """Ask the report's own planned questions (answerable) and unrelated ones (must be refused)."""
    questions = []
    for item in report["outline"]:
        section = report["sections"].get(item["id"])
        if section and section["text"] and item.get("questions"):
            questions.append((item["questions"][0], item["id"], section["title"]))
    questions = questions[:per_report]

    rows = []
    for question, _section_id, title in questions:
        result = engine.invoke(message=question, history=[], session_id=f"eval-{report['id']}", scope=report["id"])
        context = "\n\n".join(s.snippet for s in result.sources)
        full = "\n\n".join(_chunk_text(report, s.section) for s in result.sources)
        score = judges.judge_answer(judge, question, result.answer, full or context)
        hits = [s.section == title for s in result.sources]
        rows.append(
            {
                "question": question,
                "answerable": True,
                "faithful": score.faithful,
                "relevant": score.relevant,
                "refused": score.refused,
                "context_recall": float(any(hits)),
                "context_precision": share(hits),
            }
        )
    for question in OUT_OF_SCOPE:
        result = engine.invoke(message=question, history=[], session_id=f"eval-{report['id']}", scope=report["id"])
        text = result.answer.lower()
        refused = any(mark in text for mark in ("couldn't find", "could not find", "can only help", "not in the"))
        rows.append({"question": question, "answerable": False, "refused": refused})

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "questions": len(rows),
        "faithfulness": share([r["faithful"] for r in answerable]),
        "answer_relevancy": share([r["relevant"] for r in answerable]),
        "context_recall": share([bool(r["context_recall"]) for r in answerable]),
        "context_precision": round(sum(r["context_precision"] for r in answerable) / len(answerable), 3)
        if answerable
        else 0.0,
        "answered_rate": share([not r["refused"] for r in answerable]),
        "idk_correct": share([r["refused"] for r in unanswerable]),
        "rows": rows,
    }


def _chunk_text(report: dict[str, Any], title: str) -> str:
    for section in report["sections"].values():
        if section["title"] == title:
            return section["text"]
    return ""
