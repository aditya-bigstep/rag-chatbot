"""Report-quality metrics (Module 4). Every function takes the report record and returns numbers in 0..1."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from app.research.citations import MARKER, split_paragraphs
from app.research.search import USER_AGENT
from evals import judges

Report = dict[str, Any]


def ratio(part: float, whole: float) -> float | None:
    return round(part / whole, 3) if whole else None


def url_alive(url: str) -> bool:
    """Does the cited URL exist right now? (2xx/3xx; 401/403/429 = exists but blocks bots)."""
    try:
        reply = httpx.get(url, timeout=12, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    except Exception:
        return False
    return reply.status_code < 400 or reply.status_code in (401, 403, 429)


def citation_urls(report: Report) -> dict[str, Any]:
    urls = [r["url"] for r in report["references"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        alive = list(pool.map(url_alive, urls))
    return {"url_valid": ratio(sum(alive), len(urls)), "references": len(urls)}


def claims(report: Report) -> dict[str, Any]:
    """Groundedness + citation support, judged sentence by sentence against the cited sources' text."""
    by_number = {r["n"]: r for r in report["references"]}
    total = cited = supported = cited_supported = 0
    for section in report["sections"]:
        sentences = [s for paragraph in split_paragraphs(section["text"]) for s in paragraph]
        if not sentences or not section["refs"]:
            continue
        evidence = "\n\n".join(f"[{n}] {' ... '.join(by_number[n]['passages'])}" for n in section["refs"])
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
        reply = judges.judge(judges.CLAIMS, f"EVIDENCE:\n{evidence}\n\nSENTENCES:\n{listing}")
        for row in reply.get("sentences", []):
            index = row.get("i", 0) - 1
            if not row.get("claim") or not 0 <= index < len(sentences):
                continue
            has_marker = bool(MARKER.search(sentences[index]))
            total += 1
            cited += has_marker
            supported += bool(row.get("supported"))
            cited_supported += bool(row.get("supported")) and has_marker
    return {
        "claims": total,
        "groundedness": ratio(supported, total),  # 1 - hallucination rate
        "citation_coverage": ratio(cited, total),  # "no citation, no claim"
        "citation_support": ratio(cited_supported, cited),  # does the cited source back the claim?
    }


def coverage(report: Report, aspects: list[str]) -> dict[str, Any]:
    listing = "\n".join(f"{i}. {a}" for i, a in enumerate(aspects, start=1))
    reply = judges.judge(judges.COVERAGE, f"ASPECTS:\n{listing}\n\nREPORT:\n{report['markdown']}")
    covered = [bool(c) for c in reply.get("covered", [])][: len(aspects)]
    return {"coverage": ratio(sum(covered), len(aspects))}


def diversity(report: Report) -> dict[str, Any]:
    """Unique sites / sources, and how much of all citations the single most-cited site takes."""
    domain = {r["n"]: r["domain"] for r in report["references"]}
    uses = Counter(domain[int(n)] for s in report["sections"] for n in MARKER.findall(s["text"]) if int(n) in domain)
    top = max(uses.values(), default=0)
    return {
        "source_diversity": ratio(len(set(domain.values())), len(domain)),
        "top_domain_share": ratio(top, sum(uses.values())),
    }


def structure(report: Report) -> dict[str, Any]:
    body = report["markdown"].split("## References")[0]
    reply = judges.judge(judges.STRUCTURE, body)
    scores = [reply.get(key) for key in ("organization", "clarity", "flow")]
    valid = [s for s in scores if isinstance(s, int | float)]
    return {"readability": ratio(sum(valid), 5 * len(valid)), "flow": ratio(reply.get("flow") or 0, 5)}


def report_metrics(report: Report, aspects: list[str]) -> dict[str, Any]:
    words = len(report["markdown"].split("## References")[0].split())
    return {
        "words": words,
        **citation_urls(report),
        **claims(report),
        **coverage(report, aspects),
        **diversity(report),
        **structure(report),
    }
