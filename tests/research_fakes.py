"""Stand-ins for the research agent tests: a scripted LLM, a fake web search and a fake page reader.

No network and no API key — the whole research graph (including both human-in-the-loop pauses) can be
tested in milliseconds.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

OUTLINE = {
    "sections": [
        {"title": "Market overview", "guidance": "size; growth", "queries": ["market size", "market growth"]},
        {"title": "Key players", "guidance": "leaders; share", "queries": ["top companies"]},
        {"title": "Outlook", "guidance": "forecast", "queries": ["forecast 2030"]},
    ]
}
DRAFT = "Sales grew strongly [1]. Several firms compete [2]. This sentence has no citation. UNSUPPORTED claim here [1]."
SENTIMENT = {
    "overall": {"label": "positive", "justification": "Growth is strong."},
    "sections": [{"section": 1, "label": "positive", "justification": "Sales grew."}, {"section": 99, "label": "bad"}],
    "sources": [{"n": 1, "label": "neutral", "justification": "Factual."}, {"n": 999, "label": "positive"}],
}
RECOMMENDATIONS = {
    "takeaways": [{"text": "The market is growing.", "sections": [1]}, {"text": "Untraceable.", "sections": []}],
    "opportunities": [{"text": "Enter early.", "sections": [3, 42]}],
    "risks": [{"text": "Competition.", "sections": [2]}],
    "next_steps": [{"text": "Track forecasts.", "sections": [3]}],
}


class ScriptedLLM(BaseChatModel):
    """Answers by looking at the system prompt, and records every call."""

    triage: dict[str, Any] = Field(default_factory=lambda: {"action": "unclear"})
    calls: list[tuple[str, str]] = Field(default_factory=list)  # (system prompt, user prompt)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def count(self, keyword: str) -> int:
        return sum(1 for system, _ in self.calls if keyword in system)

    def _reply(self, system: str, user: str) -> str:
        if "research planner" in system:
            return json.dumps(OUTLINE)
        if "fact-checker" in system:
            numbers = [(int(n), text) for n, text in re.findall(r"^(\d+)\. (.*)$", user, re.MULTILINE)]
            return json.dumps({"verdicts": [{"i": n, "supported": "UNSUPPORTED" not in text} for n, text in numbers]})
        if "asked for a change" in system:
            return json.dumps(self.triage)
        if "sentiment" in system:
            return json.dumps(SENTIMENT)
        if "decision-maker" in system:
            return json.dumps(RECOMMENDATIONS)
        if "ONE section" in system:
            return DRAFT
        return "FAKE-ANSWER"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, user = str(messages[0].content), str(messages[-1].content)
        self.calls.append((system, user))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply(system, user)))])


class FakeSearch:
    """Every query returns pages on different sites (new sites for every query)."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        self.queries.append(query)
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
        sites = [f"{name}{len(self.queries)}.example" for name in ("alpha", "beta", "gamma", "delta")][:max_results]
        return [{"url": f"https://{site}/{slug}", "title": f"{site} on {query}", "snippet": ""} for site in sites]


class FakeReader:
    """Pages on unreadable.example fail; every other page has enough text to count as an article."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def read(self, url: str) -> str:
        self.urls.append(url)
        if "unreadable" in url:
            return ""
        return f"Report from {url}. " + "Sales grew strongly and several firms compete in this market. " * 8
