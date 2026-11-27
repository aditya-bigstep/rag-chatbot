"""Server-rendered ResearchPilot pages: /research (list + new topic) and /research/{id} (one report)."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from app.deps import get_store
from app.research.deps import Research, get_research
from app.research.store import ReportNotFoundError
from app.storage.json_store import SessionNotFoundError, SessionStore

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(include_in_schema=False)

_markdown = MarkdownIt("commonmark", {"html": False})  # html off: web text can never inject tags
_CITATION = re.compile(r"\[(\d+)\]")


def render_section(text: str) -> str:
    """Markdown -> HTML, with every [n] turned into a link to reference n."""
    rendered: str = _markdown.render(text)
    return _CITATION.sub(r'<a class="cite" href="#ref-\1">[\1]</a>', rendered)


def chat_history(record: dict[str, Any], sessions: SessionStore) -> list[Any]:
    try:
        return sessions.get(record["chat_session_id"]).messages if record.get("chat_session_id") else []
    except SessionNotFoundError:
        return []


@router.get("/research", response_class=HTMLResponse)
def research_home(request: Request, research: Research = Depends(get_research)) -> HTMLResponse:
    return templates.TemplateResponse(request, "research.html", {"reports": research.service.store.list()})


@router.get("/research/{report_id}", response_class=HTMLResponse)
def report_page(
    report_id: str,
    request: Request,
    research: Research = Depends(get_research),
    sessions: SessionStore = Depends(get_store),
) -> HTMLResponse:
    try:
        record = research.service.get(report_id)
    except ReportNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found") from None
    context = {
        "r": record,
        "bodies": {s["id"]: render_section(s["text"]) for s in record["sections"]},
        "titles": {s["id"]: f"{s['number']}. {html.escape(s['title'])}" for s in record["sections"]},
        "messages": chat_history(record, sessions),
        "busy": record["status"] in ("planning", "writing", "revising"),
    }
    return templates.TemplateResponse(request, "report.html", context)
