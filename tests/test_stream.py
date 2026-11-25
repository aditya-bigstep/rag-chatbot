import json

from fastapi.testclient import TestClient


def test_stream_endpoint_sends_tokens_then_done(client: TestClient) -> None:
    session = client.post("/api/sessions").json()
    with client.stream("POST", f"/api/sessions/{session['id']}/messages/stream", json={"content": "hi"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    events = [e for e in body.split("\n\n") if e]
    assert events[0].startswith("event: user")
    assert events[1].startswith("event: token")
    assert events[-1].startswith("event: done")
    done = json.loads(events[-1].split("data: ", 1)[1])
    assert done["role"] == "assistant"
    saved = client.get(f"/api/sessions/{session['id']}").json()
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant"]


def test_stream_unknown_session_is_404(client: TestClient) -> None:
    assert client.post("/api/sessions/nope/messages/stream", json={"content": "hi"}).status_code == 404
