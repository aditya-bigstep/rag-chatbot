"""Average the per-topic results into one comparison table (Markdown) + summary.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.versions import VERSIONS

# (group in the result file, key, label, format, is a higher value better?)
ROWS: list[tuple[str, str, str, str, bool | None]] = [
    ("report", "url_valid", "Citation validity — cited URLs exist", "%", True),
    ("report", "citation_support", "Citation validity — cited source supports the claim", "%", True),
    ("report", "citation_coverage", "Claims that carry a citation", "%", True),
    ("report", "groundedness", "Groundedness (1 − hallucination rate)", "%", True),
    ("report", "coverage", "Coverage of expected aspects", "%", True),
    ("report", "source_diversity", "Source diversity — unique sites / sources", "%", True),
    ("report", "top_domain_share", "Share of citations from the most-cited site", "%", False),
    ("report", "readability", "Structure & readability (rubric, LLM judge)", "%", True),
    ("report", "flow", "… of which flow", "%", True),
    ("report", "words", "Report length (words)", "n", None),
    ("report", "claims", "Factual claims per report", "n", None),
    ("chat", "faithfulness", "Chatbot faithfulness", "%", True),
    ("chat", "answer_relevancy", "Chatbot answer relevancy", "%", True),
    ("chat", "answer_correct", "Chatbot answers matching the expected answer", "%", True),
    ("chat", "context_precision", "Context precision", "%", True),
    ("chat", "context_recall", "Context recall", "%", True),
    ("chat", "idk_correct", '"I don\'t know" handled correctly', "%", True),
    ("chat", "false_refusals", "Answerable questions wrongly refused", "%", False),
    ("system", "latency_s", "Latency per report (s)", "n", False),
    ("system", "input_tokens", "Input tokens per report", "n", False),
    ("system", "output_tokens", "Output tokens per report", "n", False),
    ("system", "cost_usd", "Estimated cost per report (USD)", "$", False),
    ("system", "llm_calls", "LLM calls per report", "n", None),
    ("system", "search_calls", "Search calls per report", "n", None),
    ("system", "pages_fetched", "Pages read per report", "n", None),
    ("system", "claims_dropped", "Sentences removed by verification", "n", None),
]


def load(results: Path, version: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((results / version).glob("*.json"))]


def average(runs: list[dict[str, Any]], group: str, key: str) -> float | None:
    values = [run[group].get(key) for run in runs]
    numbers = [v for v in values if isinstance(v, int | float)]
    return sum(numbers) / len(numbers) if numbers else None


def show(value: float | None, style: str) -> str:
    if value is None:
        return "—"
    return {"%": f"{value:.1%}", "$": f"${value:.4f}", "n": f"{value:,.1f}"}[style]


def summarize(results: Path, output: Path) -> str:
    versions = [v for v in VERSIONS if (results / v).exists()]
    runs = {v: load(results, v) for v in versions}
    summary: dict[str, Any] = {v: {"topics": len(runs[v])} for v in versions}
    header = "| Metric | " + " | ".join(f"{v} ({len(runs[v])} topics)" for v in versions) + " |"
    lines = [header, "|---|" + "---:|" * len(versions)]
    for group, key, label, style, _ in ROWS:
        values = [average(runs[v], group, key) for v in versions]
        for version, value in zip(versions, values, strict=True):
            summary[version][key] = value
        lines.append(f"| {label} | " + " | ".join(show(v, style) for v in values) + " |")
    table = "\n".join(lines)
    (results / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table + "\n", encoding="utf-8")
    return table
