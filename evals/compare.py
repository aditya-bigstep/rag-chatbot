"""Average the per-topic scores of each run and print a comparison table (Markdown).

uv run python -m evals.compare v1 v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ROWS = [
    ("report", "citation_url_valid", "Citation validity: URL resolves"),
    ("report", "citation_supports_claim", "Citation validity: source supports the claim"),
    ("report", "groundedness", "Groundedness (supported sentences)"),
    ("report", "hallucination_rate", "Hallucination rate (lower is better)"),
    ("report", "uncited_sentences", "Uncited sentences per report (lower is better)"),
    ("report", "coverage", "Coverage of expected aspects"),
    ("report", "source_diversity", "Source diversity (unique domains / sources)"),
    ("report", "readability", "Structure and readability (rubric)"),
    ("report", "sentences", "Sentences per report"),
    ("chat", "faithfulness", "Chat faithfulness"),
    ("chat", "answer_relevancy", "Chat answer relevancy"),
    ("chat", "context_precision", "Context precision"),
    ("chat", "context_recall", "Context recall"),
    ("chat", "idk_correct", "Correct refusals (out-of-scope questions)"),
    ("system", "latency_seconds", "Latency per report (s)"),
    ("system", "llm_calls", "LLM calls per report"),
    ("system", "input_tokens", "Input tokens per report"),
    ("system", "output_tokens", "Output tokens per report"),
    ("system", "cost_usd", "Estimated cost per report (USD)"),
    ("system", "search_calls", "Search calls per report"),
    ("system", "pages_read", "Pages read per report"),
]


def averages(label: str, only: set[str] | None = None) -> tuple[dict[str, Any], dict[tuple[str, str], float], set[str]]:
    data = json.loads((ROOT / "results" / f"{label}.json").read_text())
    rows = [r for r in data["topics"] if "error" not in r and (only is None or r["topic"] in only)]
    result = {}
    for group, key, _ in ROWS:
        values = [r[group][key] for r in rows if r.get(group, {}).get(key) is not None]
        result[(group, key)] = sum(values) / len(values) if values else float("nan")
    return data["config"], result, {r["topic"] for r in rows}


def main() -> None:
    labels = sys.argv[1:] or ["v1", "v2"]
    topic_sets = [averages(label)[2] for label in labels]
    shared = set.intersection(*topic_sets)  # compare only topics every version finished
    runs = [averages(label, shared) for label in labels]
    print(f"Topics scored in every version: {len(shared)}\n")
    print("| Metric | " + " | ".join(labels) + " |")
    print("|---|" + "---:|" * len(labels))
    for group, key, title in ROWS:
        cells = []
        for _, values, _ in runs:
            value = values[(group, key)]
            cells.append(f"{value:.4f}" if key == "cost_usd" else f"{value:.2f}" if value < 10 else f"{value:.0f}")
        print(f"| {title} | " + " | ".join(cells) + " |")
    for label, (config, _, _) in zip(labels, runs, strict=True):
        print(f"\n{label}: {config}")


if __name__ == "__main__":
    main()
