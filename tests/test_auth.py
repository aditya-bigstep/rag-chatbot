"""Accounts: sign-up with an e-mailed code, sign-in, password reset, and the login wall."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth.service import AuthError, AuthService
from app.auth.store import MemoryKV, SqliteKV, SqliteUsers, connect
from app.config import load_settings
from app.main import create_app


class Outbox:
    """Stands in for Brevo: keeps every message so a test can read the code or link from it."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to_email: str, to_name: str, subject: str, html: str) -> bool:
        self.sent.append((to_email, subject, html))
        return True

    def last_code(self) -> str:
        return re.search(r"\b(\d{6})\b", self.sent[-1][1]).group(1)  # type: ignore[union-attr]

    def last_link(self) -> str:
        return re.search(r'href="([^"]+)"', self.sent[-1][2]).group(1)  # type: ignore[union-attr]


@pytest.fixture()
def outbox() -> Outbox:
    return Outbox()


@pytest.fixture()
def auth(tmp_path: Path, outbox: Outbox) -> AuthService:
    users = SqliteUsers(connect(tmp_path / "accounts.sqlite"))
    return AuthService(users, MemoryKV(), outbox, "http://test")


def test_signup_needs_the_emailed_code(auth: AuthService, outbox: Outbox) -> None:
    user = auth.signup("Priya@Example.com", "Priya", "correct horse")
    assert outbox.sent[-1][0] == "priya@example.com"
    with pytest.raises(AuthError):
        auth.verify(user.id, "000000" if outbox.last_code() != "000000" else "111111")
    token = auth.verify(user.id, outbox.last_code())
    assert auth.user_for_token(token).email == "priya@example.com"  # type: ignore[union-attr]


def test_signin_checks_the_password(auth: AuthService, outbox: Outbox) -> None:
    user = auth.signup("a@example.com", "A", "correct horse")
    auth.verify(user.id, outbox.last_code())
    with pytest.raises(AuthError):
        auth.signin("a@example.com", "wrong horse")
    _, token = auth.signin("a@example.com", "correct horse")
    assert token and auth.user_for_token(token) is not None
    auth.end_session(token)
    assert auth.user_for_token(token) is None


def test_unverified_account_cannot_sign_in(auth: AuthService, outbox: Outbox) -> None:
    auth.signup("b@example.com", "B", "correct horse")
    _, token = auth.signin("b@example.com", "correct horse")
    assert token is None, "a fresh code is sent instead of a session"
    assert len(outbox.sent) == 2


def test_short_password_is_refused(auth: AuthService) -> None:
    with pytest.raises(AuthError):
        auth.signup("c@example.com", "C", "short")


def test_reset_link_works_once(auth: AuthService, outbox: Outbox) -> None:
    user = auth.signup("d@example.com", "D", "correct horse")
    auth.verify(user.id, outbox.last_code())
    auth.forgot("d@example.com")
    token = outbox.last_link().split("token=")[1]
    auth.reset(token, "a new password")
    with pytest.raises(AuthError):
        auth.reset(token, "another password")
    assert auth.signin("d@example.com", "a new password")[1]


def test_forgot_does_not_reveal_unknown_addresses(auth: AuthService, outbox: Outbox) -> None:
    auth.forgot("nobody@example.com")
    assert outbox.sent == []


def test_too_many_attempts_are_blocked(auth: AuthService, outbox: Outbox) -> None:
    user = auth.signup("e@example.com", "E", "correct horse")
    auth.verify(user.id, outbox.last_code())
    for _ in range(10):
        with pytest.raises(AuthError):
            auth.signin("e@example.com", "wrong horse")
    with pytest.raises(AuthError, match="Too many attempts"):
        auth.signin("e@example.com", "correct horse")


def test_sessions_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "accounts.sqlite"
    SqliteKV(connect(path)).set("sess:abc", "user-1", 60)
    assert SqliteKV(connect(path)).get("sess:abc") == "user-1"


@pytest.fixture()
def walled(tmp_path: Path, outbox: Outbox) -> TestClient:
    base = load_settings()
    settings = replace(base, data_dir=tmp_path, sessions_dir=tmp_path / "s", auth_enabled=True, brevo_api_key="")
    app = create_app(settings)
    app.state.auth.mailer = outbox
    return TestClient(app)


def test_visitors_see_the_welcome_page_and_nothing_else(walled: TestClient) -> None:
    assert "Research you can check" in walled.get("/").text
    assert walled.get("/api/research").status_code == 401
    assert walled.get("/chat", follow_redirects=False).headers["location"] == "/auth"
    assert walled.post("/api/sessions", follow_redirects=False).status_code == 401


def test_signing_up_in_the_browser_opens_the_app(walled: TestClient, outbox: Outbox) -> None:
    form = {"email": "f@example.com", "name": "F", "password": "correct horse"}
    done = walled.post("/auth/signup", data=form, headers={"Accept": "application/json"}).json()
    user_id = done["next"].split("u=")[1]
    walled.post("/auth/verify", data={"u": user_id, "code": outbox.last_code()})
    home = walled.get("/")
    assert "What should we research?" in home.text and "f@example.com" in home.text


def test_one_user_cannot_open_another_users_chat(walled: TestClient, outbox: Outbox) -> None:
    def sign_up(email: str) -> None:
        walled.cookies.clear()
        form = {"email": email, "name": "", "password": "correct horse"}
        done = walled.post("/auth/signup", data=form, headers={"Accept": "application/json"}).json()
        walled.post("/auth/verify", data={"u": done["next"].split("u=")[1], "code": outbox.last_code()})

    sign_up("g@example.com")
    session_id = walled.post("/api/sessions").json()["id"]
    assert walled.get(f"/api/sessions/{session_id}").status_code == 200
    sign_up("h@example.com")
    assert walled.get(f"/api/sessions/{session_id}").status_code == 404
    assert walled.get("/api/sessions").json() == []


def test_admin_signs_in_with_a_plain_name(outbox: Outbox, tmp_path: Path) -> None:
    service = AuthService(SqliteUsers(connect(tmp_path / "accounts.sqlite")), MemoryKV(), outbox, "http://test")
    with pytest.raises(AuthError):  # no admin configured: "admin" is just an invalid e-mail
        service.signin("admin", "1234")

    service.ensure_admin("admin", "1234")
    service.ensure_admin("admin", "1234")  # every start-up calls it again
    user, token = service.signin(" Admin ", "1234")
    assert token and user.verified and outbox.sent == []
    assert service.user_for_token(token).email == "admin"  # type: ignore[union-attr]
    with pytest.raises(AuthError):
        service.signin("admin", "wrong")

    service.ensure_admin("admin", "new-password")  # changed in .env
    with pytest.raises(AuthError):
        service.signin("admin", "1234")
    assert service.signin("admin", "new-password")[1]
