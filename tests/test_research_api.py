"""All four modules through the HTTP API, with a fake LLM and a fake web (no network)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.main import create_app
from app.research.deps import build_research, get_research
from app.vectordb.chunking import get_chunker
from tests.fakes import HashEmbedder, ListStore
from tests.research_fakes import FakeReader, FakeSearch, ScriptedLLM


@pytest.fixture()
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture()
def api(settings: Settings, llm: ScriptedLLM) -> Iterator[TestClient]:
    app = create_app(settings)
    stores: dict[str, ListStore] = {}
    research = build_research(
        settings,
        app.state.session_store,
        llm=llm,
        search=FakeSearch(),
        reader=FakeReader(),
        embedder=HashEmbedder(),
        chunker=get_chunker("recursive", 400, 40),
        checkpointer=InMemorySaver(),
        background=False,  # run the graph inline so the test can assert right away
        kb_options={"store_factory": lambda name: stores.setdefault(name, ListStore()), "min_score": 0.2},
    )
    app.dependency_overrides[get_research] = lambda: research
    with TestClient(app) as client:
        yield client


def ready_report(api: TestClient) -> str:
    report_id: str = api.post("/api/research", json={"topic": "Electric scooters"}).json()["id"]
    api.post(f"/api/research/{report_id}/outline", json={})
    return report_id


def test_full_flow_plan_approve_report(api: TestClient) -> None:
    created = api.post("/api/research", json={"topic": "Electric scooters"})
    assert created.status_code == 201
    report_id = created.json()["id"]
    record = api.get(f"/api/research/{report_id}").json()
    assert record["status"] == "awaiting_outline" and len(record["outline"]) == 3
    assert all(s["text"] == "" for s in record["sections"])  # nothing written before approval

    api.post(f"/api/research/{report_id}/outline", json={})
    record = api.get(f"/api/research/{report_id}").json()
    assert record["status"] == "ready" and len(record["sections"]) == 3 and record["references"]
    assert record["metrics"]["search_calls"] == 4 and record["metrics"]["llm_calls"] == 7
    assert "latency_s" in record["metrics"] and "cost_usd" in record["metrics"]
    assert record["log"][-1]["changed"] == ["s1", "s2", "s3"]


def test_feedback_changes_one_section_and_is_logged(api: TestClient, llm: ScriptedLLM) -> None:
    report_id = ready_report(api)
    llm.triage = {"action": "expand", "sections": [3]}
    api.post(f"/api/research/{report_id}/feedback", json={"text": "Make section 3 more detailed"})
    record = api.get(f"/api/research/{report_id}").json()
    assert [s["revision"] for s in record["sections"]] == [1, 1, 2]
    assert record["log"][-1]["changed"] == ["s3"] and record["version"] == 2


def test_requests_in_the_wrong_state_are_rejected(api: TestClient) -> None:
    report_id = api.post("/api/research", json={"topic": "Electric scooters"}).json()["id"]
    assert api.post(f"/api/research/{report_id}/feedback", json={"text": "x"}).status_code == 409
    assert api.post(f"/api/research/{report_id}/analysis").status_code == 409
    assert api.get("/api/research/000000000000").status_code == 404
    assert api.post("/api/research", json={"topic": "x"}).status_code == 422


def test_analysis_keeps_only_traceable_results(api: TestClient) -> None:
    report_id = ready_report(api)
    analysis = api.post(f"/api/research/{report_id}/analysis").json()
    assert analysis["sentiment"]["overall"]["label"] == "positive"
    assert [row["id"] for row in analysis["sentiment"]["sections"]] == ["s1"]  # unknown section 99 dropped
    assert [row["n"] for row in analysis["sentiment"]["sources"]] == [1]  # unknown source 999 dropped
    advice = analysis["recommendations"]
    assert [item["text"] for item in advice["takeaways"]] == ["The market is growing."]  # untraceable one dropped
    assert advice["opportunities"][0]["sections"] == ["s3"]  # unknown section 42 removed from the trace


def test_ingest_then_chat_shows_section_and_source(api: TestClient) -> None:
    report_id = ready_report(api)
    assert api.post(f"/api/research/{report_id}/chat", json={"text": "hi?"}).status_code == 409  # not ingested
    ingested = api.post(f"/api/research/{report_id}/ingest").json()
    assert ingested["chunks"] >= 3
    reply = api.post(f"/api/research/{report_id}/chat", json={"text": "Did sales grow strongly?"}).json()
    assert reply["sources"], "an answer from the knowledge base must show where it came from"
    first = reply["sources"][0]
    assert first["section"].startswith("Section ") and first["references"][0]["url"].startswith("https://")


def test_chat_refuses_when_the_report_does_not_cover_the_question(api: TestClient, llm: ScriptedLLM) -> None:
    report_id = ready_report(api)
    api.post(f"/api/research/{report_id}/ingest")
    calls = len(llm.calls)
    reply = api.post(f"/api/research/{report_id}/chat", json={"text": "Explain quantum chromodynamics"}).json()
    assert reply["answer"].startswith("I couldn't find") and reply["sources"] == []
    assert len(llm.calls) == calls  # refused without asking the LLM at all


def test_reingest_after_an_edit_does_not_duplicate_chunks(api: TestClient) -> None:
    report_id = ready_report(api)
    first = api.post(f"/api/research/{report_id}/ingest").json()["chunks"]
    assert api.post(f"/api/research/{report_id}/ingest").json()["chunks"] == first


def test_pages_and_export(api: TestClient) -> None:
    report_id = ready_report(api)
    assert "Electric scooters" in api.get("/research").text
    page = api.get(f"/research/{report_id}").text
    assert 'href="#ref-1"' in page and "Ask for a change" in page
    export = api.get(f"/api/research/{report_id}/export.md")
    assert export.text.startswith("# Electric scooters") and "## References" in export.text
