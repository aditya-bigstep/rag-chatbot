"""Report chat: "summarise it" must read the whole report instead of searching for those words."""

from __future__ import annotations

import pytest

from app.chat.engine import OVERVIEW_INSTRUCTIONS, get_engine
from app.chat.guardrails import check_grounding
from app.chat.router import is_overview
from app.models import Chunk, RetrievedChunk
from tests.research_fakes import ScriptedLLM


class ReportRetriever:
    """A scoped retriever whose similarity search finds nothing, like "summarise it" does in real life."""

    k = 4

    def __init__(self) -> None:
        self.searched: list[str] = []

    def scoped(self, where: dict[str, str]) -> ReportRetriever:
        return self

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self.searched.append(query)
        return []

    def everything(self, max_chars: int = 40_000) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=Chunk(id=f"d:{i}", doc_id="d", text=text, source="Batteries", position=i),
                score=1.0,
            )
            for i, text in enumerate(["Intro: solid-state cells use a solid electrolyte.", "Risks: cost stays high."])
        ]


@pytest.mark.parametrize(
    "message",
    ["summarise it", "sumamrise it", "Summarize the report", "give me a summary", "key takeaways?", "tl;dr",
     "what is this report about?", "give me an overview"],
)
def test_overview_questions_are_recognised(message: str) -> None:
    assert is_overview(message)


@pytest.mark.parametrize("message", ["what does a solid electrolyte cost?", "who makes them?", "it was summer in 2020"])
def test_specific_questions_are_not_overviews(message: str) -> None:
    assert not is_overview(message)


def test_summarise_reads_the_whole_report() -> None:
    llm, retriever = ScriptedLLM(), ReportRetriever()
    engine = get_engine(llm=llm, retriever=retriever, tools=[])
    result = engine.invoke("sumamrise it", [], "s1", scope="r1")  # type: ignore[call-arg]

    system = llm.calls[-1][0]
    assert "solid electrolyte" in system and "cost stays high" in system
    assert OVERVIEW_INSTRUCTIONS in system
    assert retriever.searched == []
    assert len(result.sources) == 2


def test_summarise_outside_a_report_still_searches() -> None:
    llm, retriever = ScriptedLLM(), ReportRetriever()
    get_engine(llm=llm, retriever=retriever, tools=[]).invoke("summarise it", [], "s1")
    assert retriever.searched == ["summarise it"]


def test_list_numbering_is_not_an_unsupported_number() -> None:
    context = ["Solid-state batteries promise higher energy density and improved safety for vehicles."]
    answer = "1. Solid-state batteries promise higher energy density.\n2. They bring improved safety for vehicles."
    assert check_grounding(answer, context).supported
