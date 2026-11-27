"""The two system versions the evaluation compares.

  v1  baseline  — the writer's draft is published as is; the chatbot sees every retrieved chunk.
  v2  verified  — a fact-check step removes uncited/unsupported sentences; the chatbot ignores chunks
                  below a similarity threshold (so it can refuse without asking the LLM).

Each version changes ONE thing in the report pipeline and ONE thing in the chatbot, so a change in a
report metric is caused by `verify`, and a change in a chatbot metric mostly by `min_score`.
"""

from __future__ import annotations

from typing import Any

from app.research.types import ResearchConfig

VERSIONS: dict[str, dict[str, Any]] = {
    "v1": {
        "label": "baseline (no verification, no retrieval threshold)",
        "config": ResearchConfig(verify=False),
        "min_score": 0.0,
    },
    "v2": {
        "label": "verified (claim verification + retrieval threshold 0.25)",
        "config": ResearchConfig(verify=True),
        "min_score": 0.25,
    },
}
