from __future__ import annotations

from fastapi.testclient import TestClient


def test_home_renders_without_sessions(client: TestClient) -> None:
    response = client.get("/chat")
    assert response.status_code == 200
    assert "New chat" in response.text


def test_research_home_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ResearchPilot" in response.text


def test_research_home_sample_topics_change_on_every_visit(client: TestClient) -> None:
    from html import unescape

    from app import starters

    def shown(html: str) -> set[str]:
        return {t for t in starters.ALL if f'data-topic="{t}"' in unescape(html)}

    first = shown(client.get("/").text)
    assert len(first) == 3
    for _ in range(10):
        following = shown(client.get("/").text)  # the client sends back the cookie from the last visit
        assert len(following) == 3 and not following & first
        first = following


def test_ui_flow_create_rename_chat_delete(client: TestClient) -> None:
    created = client.post("/ui/sessions", follow_redirects=False)
    assert created.status_code == 303
    url = created.headers["location"]
    sid = url.rsplit("/", 1)[-1]

    assert client.get(url).status_code == 200
    client.post(f"/ui/sessions/{sid}/rename", data={"title": "My chat"})
    assert "My chat" in client.get(url).text

    partial = client.post(f"/ui/sessions/{sid}/messages", data={"content": "hello there"})
    assert partial.status_code == 200
    assert "hello there" in partial.text

    client.post(f"/ui/sessions/{sid}/delete")
    assert client.get(url).status_code == 404
