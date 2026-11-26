"""Server-rendered pages (Jinja2 + HTMX). Complete — no TODOs here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.deps import Components, UserSessions, get_components, get_store, is_mine, user_id_of
from app.services import UploadError, run_chat_turn, save_and_index_upload
from app.storage.json_store import SessionNotFoundError

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(include_in_schema=False)


def _static_v(path: str) -> int:
    """A version for a static file's URL (its modified time), so an edited file is fetched again."""
    try:
        return int((TEMPLATES_DIR.parent / "static" / path).stat().st_mtime)
    except OSError:
        return 0


templates.env.globals["static_v"] = _static_v


def _redirect(request: Request, url: str) -> Response:
    if request.headers.get("HX-Request"):
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def _shell(request: Request, store: UserSessions, components: Components) -> dict[str, Any]:
    """What every signed-in page needs for its sidebar: the user, their reports and their chats."""
    reports = components.research.summaries(user_id_of(request)) if components.research else []
    chats = [s for s in store.list() if not s.report_id]  # report chats live inside their report
    return {"user": getattr(request.state, "user", None), "reports": reports, "sessions": chats}


def _chat_page(request: Request, store: UserSessions, components: Components, session_id: str | None) -> HTMLResponse:
    current = None
    if session_id:
        try:
            current = store.get(session_id)
        except SessionNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    context = {**_shell(request, store, components), "current": current, "missing": components.missing}
    return templates.TemplateResponse(request, "index.html", context)


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    store: UserSessions = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    """Visitors get the scroll page that explains the product. Signed-in users get the new-report screen."""
    auth_on = request.app.state.auth is not None
    if auth_on and getattr(request.state, "user", None) is None:
        return welcome(request)
    return templates.TemplateResponse(request, "research_home.html", _shell(request, store, components))


@router.get("/welcome", response_class=HTMLResponse)
def welcome(request: Request) -> HTMLResponse:
    context = {"user": getattr(request.state, "user", None), "auth": request.app.state.auth is not None}
    return templates.TemplateResponse(request, "welcome.html", context)


@router.get("/research/{report_id}", response_class=HTMLResponse)
def research_page(
    report_id: str,
    request: Request,
    store: UserSessions = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    if components.research is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Research service is not available")
    try:
        report = components.research.get(report_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found") from None
    if not is_mine(report.get("user_id"), user_id_of(request)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    context = {**_shell(request, store, components), "report_id": report_id, "topic": report["topic"]}
    return templates.TemplateResponse(request, "research_report.html", context)


@router.get("/chat", response_class=HTMLResponse)
def chat_home(
    request: Request,
    store: UserSessions = Depends(get_store),
    components: Components = Depends(get_components),
) -> Response:
    chats = [s for s in store.list() if not s.report_id]
    if chats:
        return RedirectResponse(f"/sessions/{chats[0].id}", status_code=status.HTTP_303_SEE_OTHER)
    return _chat_page(request, store, components, None)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def session_page(
    session_id: str,
    request: Request,
    store: UserSessions = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    return _chat_page(request, store, components, session_id)


@router.post("/ui/sessions")
def ui_create_session(request: Request, store: UserSessions = Depends(get_store)) -> Response:
    session = store.create()
    return _redirect(request, f"/sessions/{session.id}")


@router.post("/ui/sessions/{session_id}/rename")
def ui_rename_session(
    session_id: str,
    request: Request,
    title: str = Form(...),
    store: UserSessions = Depends(get_store),
) -> Response:
    try:
        store.rename(session_id, title)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return _redirect(request, f"/sessions/{session_id}")


@router.post("/ui/sessions/{session_id}/delete")
def ui_delete_session(session_id: str, request: Request, store: UserSessions = Depends(get_store)) -> Response:
    try:
        store.delete(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return _redirect(request, "/chat")


@router.post("/ui/sessions/{session_id}/messages", response_class=HTMLResponse)
def ui_send_message(
    session_id: str,
    request: Request,
    content: str = Form(..., min_length=1, max_length=8000),
    store: UserSessions = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    try:
        user_message, assistant_message = run_chat_turn(store, components.engine, session_id, content)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return templates.TemplateResponse(
        request, "partials/messages.html", {"messages": [user_message, assistant_message]}
    )


@router.post("/ui/documents", response_class=HTMLResponse)
async def ui_upload_document(
    request: Request,
    file: UploadFile,
    components: Components = Depends(get_components),
) -> HTMLResponse:
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_upload_bytes + 1)
    try:
        result = save_and_index_upload(settings, components, file.filename or "", data)
        message, ok = f"{result.filename}: {result.detail}", True
    except UploadError as exc:
        message, ok = str(exc), False
    return templates.TemplateResponse(request, "partials/upload_status.html", {"message": message, "ok": ok})
