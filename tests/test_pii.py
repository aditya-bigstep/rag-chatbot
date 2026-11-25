from pathlib import Path
from typing import Any

import pytest

from app.chat.pii import MASK, redact_pii
from app.models import EngineResult
from app.services import run_chat_turn, run_chat_turn_stream
from app.storage.json_store import SessionStore


@pytest.mark.parametrize(
    ("text", "secret", "kind"),
    [
        ("mail me at john.doe@gmail.com", "john.doe@gmail.com", "email"),
        ("my number is +91 98765 43210", "98765 43210", "phone"),
        ("call 9876543210 today", "9876543210", "phone"),
        ("US office (415) 555-0100", "555-0100", "phone"),
        ("key sk-ant-api03-abcdefghijklmnopqrstuvwx", "sk-ant-api03", "api_key"),
        ("aws AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE", "api_key"),
        ("gh ghp_abcdefghijklmnopqrstuvwxyz0123456789AB", "ghp_", "api_key"),
        ("password: hunter22", "hunter22", "secret"),
        ("my password is hunter22", "hunter22", "secret"),
        ("api_key=abc123xyz", "abc123xyz", "secret"),
        ("Authorization: Bearer abcdefghijklmnop1234567890", "abcdefghijklmnop", "bearer_token"),
        ("card 4111 1111 1111 1111", "4111", "card"),
        ("aadhaar 2345 6789 0123", "6789", "aadhaar"),
        ("PAN ABCDE1234F", "ABCDE1234F", "pan"),
        ("ssn 123-45-6789", "123-45-6789", "ssn"),
    ],
)
def test_redacts_sensitive_values(text: str, secret: str, kind: str) -> None:
    result = redact_pii(text)
    assert secret not in result.text
    assert MASK in result.text
    assert kind in result.kinds


def test_keeps_labels_around_masked_values() -> None:
    assert redact_pii("password: hunter22 thanks").text == f"password: {MASK} thanks"


@pytest.mark.parametrize(
    "text",
    [
        "What is the refund policy?",
        "What is 1234 * 5678?",
        "Orders placed in 2024 ship within 15 days.",
        "Is a password required to log in?",
        "Order id 1234567890123456 is late",  # 16 digits but fails the Luhn check -> not a card
    ],
)
def test_leaves_normal_text_alone(text: str) -> None:
    result = redact_pii(text)
    assert result.text == text
    assert not result.changed


class RecordingEngine:
    """Remembers every message it is given, so we can prove the raw value never reached it."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def invoke(self, message: str, history: list[Any], session_id: str) -> EngineResult:
        self.seen.append(message)
        return EngineResult(answer="ok", sources=[])

    def stream(self, message: str, history: list[Any], session_id: str):  # type: ignore[no-untyped-def]
        self.seen.append(message)
        yield "ok"
        yield EngineResult(answer="ok", sources=[])


SECRET_MSG = "my email is jane@example.com and key sk-proj-abcdefghijklmnopqrstuv"


def _assert_never_leaked(engine: RecordingEngine, store: SessionStore, session_id: str) -> None:
    assert engine.seen and all("jane@example.com" not in m and "sk-proj" not in m for m in engine.seen)
    saved = store.get(session_id)
    assert "jane@example.com" not in saved.title
    for m in saved.messages:
        assert "jane@example.com" not in m.content and "sk-proj" not in m.content


def test_run_chat_turn_masks_before_engine_and_storage(tmp_path: Path) -> None:
    store, engine = SessionStore(tmp_path), RecordingEngine()
    session = store.create()
    user_message, _ = run_chat_turn(store, engine, session.id, SECRET_MSG)  # type: ignore[arg-type]
    assert user_message.content == f"my email is {MASK} and key {MASK}"
    _assert_never_leaked(engine, store, session.id)


def test_stream_turn_masks_before_engine_and_storage(tmp_path: Path) -> None:
    store, engine = SessionStore(tmp_path), RecordingEngine()
    session = store.create()
    items = list(run_chat_turn_stream(store, engine, session.id, SECRET_MSG))  # type: ignore[arg-type]
    assert items[0].content == f"my email is {MASK} and key {MASK}"  # first item: masked user message
    _assert_never_leaked(engine, store, session.id)
