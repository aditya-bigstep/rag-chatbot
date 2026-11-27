"""Evaluation helpers that need no LLM: source diversity and the summary table."""

from __future__ import annotations

import json
from pathlib import Path

from evals.report_metrics import diversity
from evals.summarize import summarize


def test_diversity_counts_sites_and_the_most_cited_one() -> None:
    report = {
        "references": [
            {"n": 1, "domain": "a.example"},
            {"n": 2, "domain": "a.example"},
            {"n": 3, "domain": "b.example"},
        ],
        "sections": [{"text": "One [1]. Two [2]. Three [3]. Four [1]."}],
    }
    assert diversity(report) == {"source_diversity": 0.667, "top_domain_share": 0.75}


def test_summary_averages_topics_per_version(tmp_path: Path) -> None:
    for version, scores in (("v1", [0.5, 0.7]), ("v2", [0.9, 1.0])):
        (tmp_path / version).mkdir()
        for i, score in enumerate(scores):
            run = {"report": {"groundedness": score}, "chat": {}, "system": {"latency_s": 10}}
            (tmp_path / version / f"t{i}.json").write_text(json.dumps(run))
    table = summarize(tmp_path, tmp_path / "table.md")
    assert "| Groundedness (1 − hallucination rate) | 60.0% | 95.0% |" in table
    assert json.loads((tmp_path / "summary.json").read_text())["v2"]["topics"] == 2
