"""Analysis agent: sentiment and recommendations for a finished report.

A separate graph from research on purpose: it only reads a finished report, it can be
re-run on its own after an edit, and a failure here never damages the report.

    START -> source_tone (local lexicon) --+
    START -> analyse (one LLM call) -------+-> check -> END
"""

from __future__ import annotations

import hashlib
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.models import utc_now
from app.observability import traced_node
from app.research.deps import Deps
from app.research.llm_json import ask_json
from app.research.render import number_sources, section_text
from app.research.schemas import Analysis

ANALYSIS_PROMPT = (
    "You analyse a research report. Use ONLY what the report says.\n"
    "1. Sentiment: how favourable is the picture for the topic? Give an overall label and one per section "
    "(positive, neutral or negative), a score from -1 to 1 and a one sentence reason.\n"
    "2. Recommendations: 3 key takeaways, 2 or 3 opportunities, 2 or 3 risks, 2 or 3 next steps. "
    "For every item list the section ids it is based on and the [n] source numbers that back it.\n"
    'Shape: {"overall": {"id": "overall", "label": "...", "score": 0.0, "why": "..."}, '
    '"sections": [{"id": "<section id>", "label": "...", "score": 0.0, "why": "..."}], '
    '"takeaways": [{"text": "...", "sections": ["<section id>"], "sources": [1]}], '
    '"opportunities": [], "risks": [], "next_steps": []}'
)
GROUPS = ("takeaways", "opportunities", "risks", "next_steps")
_vader = SentimentIntensityAnalyzer()


class AnalysisState(TypedDict, total=False):
    report: dict[str, Any]
    tone: dict[str, Any]
    raw: dict[str, Any]
    analysis: dict[str, Any]


def label_for(score: float) -> str:
    if score >= 0.15:
        return "positive"
    return "negative" if score <= -0.15 else "neutral"


def report_fingerprint(report: dict[str, Any]) -> str:
    texts = [report["sections"][item["id"]]["text"] for item in report["outline"] if item["id"] in report["sections"]]
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]


class AnalysisGraph:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    def source_tone(self, state: AnalysisState) -> dict[str, Any]:
        """Per-source sentiment with the VADER lexicon: instant, no tokens."""
        report = state["report"]
        texts: dict[str, list[str]] = {}
        for section in report["sections"].values():
            for passage in section.get("evidence", []):
                texts.setdefault(passage["source_id"], []).append(passage["text"])
        sources = {}
        for sid, passages in texts.items():
            score = sum(_vader.polarity_scores(t)["compound"] for t in passages) / len(passages)
            sources[sid] = {"score": round(score, 2), "label": label_for(score)}
        sections = {
            sid: round(_vader.polarity_scores(section["text"])["compound"], 2)
            for sid, section in report["sections"].items()
            if section["text"]
        }
        return {"tone": {"sources": sources, "sections": sections}}

    def analyse(self, state: AnalysisState) -> dict[str, Any]:
        report = state["report"]
        numbers = number_sources(report)
        parts = []
        for item in report["outline"]:
            section = report["sections"].get(item["id"])
            if section and section["text"]:
                parts.append(f"## {item['id']} | {section['title']}\n{section_text(section, numbers)}")
        user = f"Topic: {report['topic']}\n\n" + "\n\n".join(parts)
        return {"raw": ask_json(self.deps.llm, ANALYSIS_PROMPT, user, Analysis).model_dump()}

    def check(self, state: AnalysisState) -> dict[str, Any]:
        """Keep only insights that point at real sections: every recommendation must be traceable."""
        report, raw, tone = state["report"], state["raw"], state["tone"]
        valid_sections = {sid for sid, s in report["sections"].items() if s["text"]}
        valid_numbers = set(number_sources(report).values())

        analysis: dict[str, Any] = {"overall": raw["overall"], "created_at": utc_now()}
        analysis["sections"] = []
        for row in raw["sections"]:
            if row["id"] not in valid_sections:
                continue
            lexicon = tone["sections"].get(row["id"], 0.0)
            row["lexicon_score"] = lexicon
            row["agrees"] = label_for(lexicon) == row["label"] or "neutral" in (label_for(lexicon), row["label"])
            analysis["sections"].append(row)
        for group in GROUPS:
            kept = []
            for insight in raw[group]:
                insight["sections"] = [sid for sid in insight["sections"] if sid in valid_sections]
                insight["sources"] = [n for n in insight["sources"] if n in valid_numbers]
                if insight["sections"]:
                    kept.append(insight)
            analysis[group] = kept
        analysis["sources"] = tone["sources"]
        analysis["fingerprint"] = report_fingerprint(report)
        return {"analysis": analysis}

    def build(self) -> Any:
        graph = StateGraph(AnalysisState)
        graph.add_node("source_tone", traced_node("analysis_source_tone")(self.source_tone))
        graph.add_node("analyse", traced_node("analysis_llm")(self.analyse))
        graph.add_node("check", traced_node("analysis_check")(self.check))
        graph.add_edge(START, "source_tone")
        graph.add_edge(START, "analyse")
        graph.add_edge(["source_tone", "analyse"], "check")
        graph.add_edge("check", END)
        return graph.compile()
