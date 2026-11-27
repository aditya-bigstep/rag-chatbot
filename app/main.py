"""FastAPI entry point.  Run with:  uv run uvicorn app.main:app --reload"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import web
from app.api import chat, documents, research, sessions
from app.auth.router import COOKIE
from app.auth.router import router as auth_router
from app.config import Settings, get_settings
from app.observability import configure_tracing
from app.storage.json_store import SessionStore

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own Settings that point at temp folders."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    application = FastAPI(title="ResearchPilot")
    application.state.settings = settings
    application.state.session_store = SessionStore(settings.sessions_dir)
    application.state.components = None  # built lazily by app.deps.get_components
    application.state.tracing_enabled = configure_tracing()
    application.state.auth = build_auth(settings) if settings.auth_enabled else None
    if application.state.auth is not None:
        application.middleware("http")(require_login)

    application.include_router(auth_router)

    application.include_router(sessions.router)
    application.include_router(chat.router)
    application.include_router(documents.router)
    application.include_router(research.router)
    application.include_router(web.router)
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "tracing": application.state.tracing_enabled,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "fallback": settings.llm_fallback_provider,
            "search": settings.search_provider,
            "verify": settings.research_verify,
        }

    return application


def build_auth(settings: Settings):  # type: ignore[no-untyped-def]
    from app.auth.email import get_mailer
    from app.auth.service import AuthService
    from app.auth.store import SqliteKV, SqliteUsers, connect

    database = connect(settings.accounts_db)
    service = AuthService(
        users=SqliteUsers(database),
        kv=SqliteKV(database),
        mailer=get_mailer(settings),
        base_url=settings.app_base_url,
        session_ttl_days=settings.session_ttl_days,
    )
    service.ensure_admin(settings.admin_login, settings.admin_password)
    return service


# Anyone may open these without an account. Everything else needs a login cookie.
PUBLIC_PREFIXES = ("/auth", "/static/", "/health", "/welcome")


async def require_login(request: Request, call_next):  # type: ignore[no-untyped-def]
    auth = request.app.state.auth
    request.state.user = auth.user_for_token(request.cookies.get(COOKIE))
    path = request.url.path
    public = path == "/" or any(path == p.rstrip("/") or path.startswith(p) for p in PUBLIC_PREFIXES)
    if request.state.user is None and not public:
        if path.startswith("/api/") or request.headers.get("HX-Request"):
            return JSONResponse({"detail": "Sign in required"}, status_code=401, headers={"HX-Redirect": "/auth"})
        return RedirectResponse("/auth", status_code=303)
    return await call_next(request)


logging.basicConfig(level=logging.INFO)
app = create_app()
