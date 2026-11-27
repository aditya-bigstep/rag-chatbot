"""Module 1 — research graph: planning, both HITL pauses, citations, verification, section-only edits."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.research.graph import ResearchGraphBuilder, report_of
from app.research.types import ResearchConfig
from tests.fakes import HashEmbedder
from tests.research_fakes import FakeReader, FakeSearch, ScriptedLLM

CONFIG = {"configurable": {"thread_id": "t1"}}


class Rig:
    """A research graph wired to fakes."""

    def __init__(self, verify: bool = True) -> None:
        self.llm, self.search, self.reader = ScriptedLLM(), FakeSearch(), FakeReader()
        builder = ResearchGraphBuilder(
            self.llm, self.search, self.reader, HashEmbedder(), ResearchConfig(verify=verify), InMemorySaver()
        )
        self.graph = builder.build()

    def start(self) -> dict[str, Any]:
        return self.graph.invoke({"topic": "Electric scooters"}, CONFIG)

    def resume(self, value: dict[str, Any], triage: dict[str, Any] | None = None) -> dict[str, Any]:
        if triage:
            self.llm.triage = triage
        return self.graph.invoke(Command(resume=value), CONFIG)

    def pending(self) -> dict[str, Any]:
        return self.graph.get_state(CONFIG).tasks[0].interrupts[0].value


@pytest.fixture()
def rig() -> Rig:
    return Rig()


def test_pauses_for_outline_before_any_research(rig: Rig) -> None:
    rig.start()
    assert rig.pending()["kind"] == "outline"
    assert [s["title"] for s in rig.pending()["outline"]] == ["Market overview", "Key players", "Outlook"]
    assert rig.search.queries == []  # nothing searched or written before the human approves
    assert rig.llm.count("ONE section") == 0


def test_approved_outline_writes_every_section_then_waits_for_feedback(rig: Rig) -> None:
    rig.start()
    state = rig.resume({"sections": None})
    assert rig.pending()["kind"] == "feedback"
    assert [s["revision"] for s in state["sections"].values()] == [1, 1, 1]
    assert state["stats"]["search_calls"] == 4  # 2 + 1 + 1 planner queries


def test_edited_outline_is_what_gets_written(rig: Rig) -> None:
    rig.start()
    edited = [{"title": "Market overview", "guidance": "size; growth"}, {"title": "Battery tech", "guidance": "cost"}]
    state = rig.resume({"sections": edited})
    titles = [state["sections"][key]["title"] for key in state["order"]]
    assert titles == ["Market overview", "Battery tech"]
    assert state["order"][0] == "s1"  # the untouched line kept its id (and its search queries)
    assert rig.llm.count("ONE section") == 2


def test_citations_point_only_at_pages_that_were_read(rig: Rig) -> None:
    rig.start()
    report = report_of(rig.resume({"sections": None}))
    assert report["references"], "report must have a reference list"
    for reference in report["references"]:
        assert reference["url"] in rig.reader.urls  # never a URL the system did not open
    numbers = {r["n"] for r in report["references"]}
    assert numbers == set(range(1, len(numbers) + 1))  # numbered 1..n without gaps
    for section in report["sections"]:
        assert set(section["refs"]) <= numbers
    assert "## References" in report["markdown"]


def test_verify_removes_unsupported_and_uncited_sentences(rig: Rig) -> None:
    rig.start()
    report = report_of(rig.resume({"sections": None}))
    for section in report["sections"]:
        assert "UNSUPPORTED" not in section["text"]
        assert "no citation" not in section["text"]
        assert (section["checked"], section["dropped"]) == (4, 2)


def test_without_verify_the_draft_is_kept_as_written() -> None:
    rig = Rig(verify=False)
    rig.start()
    report = report_of(rig.resume({"sections": None}))
    assert "UNSUPPORTED" in report["sections"][0]["text"]
    assert rig.llm.count("fact-checker") == 0


def test_change_request_rewrites_only_the_named_section(rig: Rig) -> None:
    rig.start()
    before = rig.resume({"sections": None})["sections"]
    writes = rig.llm.count("ONE section")
    after = rig.resume({"text": "Make section 2 more detailed"}, triage={"action": "expand", "sections": [2]})
    assert after["sections"]["s1"] == before["s1"]
    assert after["sections"]["s3"] == before["s3"]
    assert after["sections"]["s2"]["revision"] == 2
    assert len(after["sections"]["s2"]["source_ids"]) > len(before["s2"]["source_ids"])  # expand keeps + adds
    assert rig.llm.count("ONE section") == writes + 1  # exactly one section was rewritten
    assert rig.pending()["kind"] == "feedback"  # and it waits for the next request


def test_replace_sources_uses_different_websites(rig: Rig) -> None:
    rig.start()
    before = rig.resume({"sections": None})
    old = {before["sources"][sid]["domain"] for sid in before["sections"]["s2"]["source_ids"]}
    after = rig.resume({"text": "Replace weak sources in section 2"}, triage={"action": "replace", "sections": [2]})
    new = {after["sources"][sid]["domain"] for sid in after["sections"]["s2"]["source_ids"]}
    assert new and not (new & old)


def test_rewrite_does_no_web_search(rig: Rig) -> None:
    rig.start()
    rig.resume({"sections": None})
    searches = len(rig.search.queries)
    after = rig.resume({"text": "Shorten section 1"}, triage={"action": "rewrite", "sections": [1]})
    assert len(rig.search.queries) == searches
    assert after["sections"]["s1"]["revision"] == 2


def test_add_subtopic_creates_one_new_section(rig: Rig) -> None:
    rig.start()
    before = rig.resume({"sections": None})["sections"]
    triage = {"action": "add", "title": "Government policy", "guidance": "subsidies"}
    after = rig.resume({"text": "Add a subtopic on government policy"}, triage=triage)
    assert after["order"] == ["s1", "s2", "s3", "s4"]
    assert after["sections"]["s4"]["title"] == "Government policy"
    assert {key: after["sections"][key] for key in before} == before


def test_unclear_request_changes_nothing_and_says_so(rig: Rig) -> None:
    rig.start()
    before = rig.resume({"sections": None})["sections"]
    after = rig.resume({"text": "hmm"}, triage={"action": "unclear"})
    assert after["sections"] == before
    assert "could not tell" in rig.pending()["notice"]


def test_finish_ends_the_graph(rig: Rig) -> None:
    rig.start()
    rig.resume({"sections": None})
    rig.resume({"text": "Looks good"}, triage={"action": "finish"})
    assert rig.graph.get_state(CONFIG).next == ()


def test_section_without_readable_sources_makes_no_claims(rig: Rig) -> None:
    rig.reader.read = lambda url: ""  # type: ignore[method-assign]
    rig.start()
    report = report_of(rig.resume({"sections": None}))
    assert report["references"] == []
    assert all("No reliable sources" in s["text"] for s in report["sections"])
    assert rig.llm.count("ONE section") == 0  # the writer is not even called
