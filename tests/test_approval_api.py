"""HTTP-level test of the human-approval flow (feature 5): pause -> GET approval -> POST decision."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.chat.checkpointing import get_checkpointer
from app.chat.engine import get_engine
from app.chat.options import EngineOptions
from app.config import Settings
from app.deps import Components, get_components
from app.tools.admin_tools import make_admin_tools
from tests.fakes import TOOL_TRIGGER, FakeChatModel, StaticRetriever, make_chunk


def test_approval_round_trip(client: TestClient, settings: Settings) -> None:
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    victim = settings.documents_dir / "old.md"
    victim.write_text("old")
    engine = get_engine(
        llm=FakeChatModel(reply="DONE", tool_name="delete_document", tool_args={"filename": "old.md"}),
        retriever=StaticRetriever([make_chunk("x")]),
        tools=make_admin_tools(settings.documents_dir, None),
        options=EngineOptions(routing=True, hitl=True),
        checkpointer=get_checkpointer(settings.checkpoint_db),
    )
    client.app.dependency_overrides[get_components] = lambda: Components(engine=engine)  # type: ignore[attr-defined]

    session = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{session}/approval", json={"approved": True}).status_code == 409

    turn = client.post(f"/api/sessions/{session}/messages", json={"content": f"{TOOL_TRIGGER} delete old.md"}).json()
    assert "approval" in turn["assistant_message"]["content"]
    assert victim.exists()
    assert client.get(f"/api/sessions/{session}/approval").json()["pending"]["kind"] == "tool_approval"

    reply = client.post(f"/api/sessions/{session}/approval", json={"approved": True}).json()
    assert "Deleted old.md" in reply["content"]
    assert not victim.exists()
    assert client.get(f"/api/sessions/{session}/approval").json()["pending"] is None
    assert len(client.get(f"/api/sessions/{session}").json()["messages"]) == 3  # user, "needs approval", final reply
