"""JSON API for ResearchPilot: create a report, approve the outline, ask for changes, analyse, chat."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.chat.guardrails import check_input
from app.chat.memory import ConversationMemory
from app.chat.router import small_talk
from app.deps import Components, UserSessions, get_components, get_store, is_mine, user_id_of
from app.models import Message
from app.research import events
from app.research.intake import Intake, route_request
from app.research.render import report_markdown
from app.research.service import ResearchService
from app.research.store import ReportNotFoundError
from app.storage.json_store import SessionNotFoundError

router = APIRouter(prefix="/api/research", tags=["research"])
MAX_ACTIVE_RUNS = 2


class NewReport(BaseModel):
    topic: str = Field(min_length=4, max_length=300)


class Ask(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    session_id: str | None = None  # the box's conversation so far (returned by the previous call)


class OutlineItem(BaseModel):
    id: str = ""
    title: str = Field(max_length=120)
    questions: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class OutlineDecision(BaseModel):
    outline: list[OutlineItem] = Field(min_length=1, max_length=8)


class Feedback(BaseModel):
    text: str = Field(default="", max_length=600)
    section_id: str = ""
    action: str = ""


def get_service(request: Request, components: Components = Depends(get_components)) -> ResearchService:
    """The research service. Also checks that the report in the URL belongs to the signed-in user."""
    service = components.research
    if service is None:
        reason = components.missing.get("research", "research service is not available")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, reason)
    report_id = request.path_params.get("report_id")
    if report_id:
        report = _call(service.store.get, report_id)
        if not is_mine(report.get("user_id"), user_id_of(request)):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    return service


def _call(action: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return action(*args, **kwargs)
    except ReportNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found") from None
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None


@router.get("")
def list_reports(request: Request, service: ResearchService = Depends(get_service)) -> list[dict[str, Any]]:
    return service.summaries(user_id_of(request))


@router.post("", status_code=status.HTTP_201_CREATED)
def create_report(body: NewReport, request: Request, service: ResearchService = Depends(get_service)) -> dict[str, Any]:
    if len(service.active) >= MAX_ACTIVE_RUNS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Two reports are already running. Try again shortly.")
    return service.create(body.topic, user_id_of(request))


def _summarizer(service: ResearchService) -> ConversationMemory:
    """Conversation memory for the research box: recent turns verbatim, older ones summarised (cached)."""
    if getattr(service, "home_memory", None) is None:
        settings = service.settings
        service.home_memory = ConversationMemory(  # type: ignore[attr-defined]
            service.deps.fast_llm, settings.memory_max_messages, settings.memory_summarize
        )
    return service.home_memory  # type: ignore[attr-defined,no-any-return]


@router.post("/ask")
def ask(
    body: Ask,
    request: Request,
    service: ResearchService = Depends(get_service),
    sessions: UserSessions = Depends(get_store),
) -> dict[str, Any]:
    """The research box. Routes the message (see app/research/intake.py): a greeting, a sum or a quick
    question is answered right here with the conversation and the user memory, and only a real topic
    starts a report. The conversation is one JSON session file, like every chat."""
    session = sessions.home(body.session_id)
    result = route_request(
        body.text,
        service.deps.fast_llm,
        service.settings.research_routing,
        history=session.messages,
        memory=sessions.memory,
        summarizer=_summarizer(service),
    )
    reply: dict[str, Any] = {
        "route": result.route.value,
        "reason": result.reason,
        "reply": result.reply,
        "report_id": None,
        "session_id": session.id,
    }
    if result.route is Intake.AGENT:
        if len(service.active) >= MAX_ACTIVE_RUNS:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "Two reports are already running. Try again shortly."
            )
        reply["report_id"] = service.create(result.text, user_id_of(request))["id"]
    if result.route is not Intake.BLOCKED:  # a blocked message is never replayed to a model later
        said = result.reply or f"I started a research report on: {result.text}"
        sessions.append_messages(
            session.id, Message(role="user", content=result.text), Message(role="assistant", content=said)
        )
    return reply


def _guard(*texts: str) -> None:
    """Guardrails for text typed into a report (outline titles, change requests): refuse before any work starts."""
    for text in texts:
        if text.strip():
            verdict = check_input(text)
            if not verdict.allowed:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, verdict.reply)


@router.get("/{report_id}")
def get_report(report_id: str, after: int = 0, service: ResearchService = Depends(get_service)) -> dict[str, Any]:
    report = _call(service.get, report_id)
    report["events"] = events.read(report_id, after)
    return report


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(report_id: str, service: ResearchService = Depends(get_service)) -> None:
    _call(service.delete, report_id)


@router.post("/{report_id}/outline", status_code=status.HTTP_202_ACCEPTED)
def approve_outline(
    report_id: str, body: OutlineDecision, service: ResearchService = Depends(get_service)
) -> dict[str, str]:
    """Human step 1: the user approves (or edits) the outline and the research starts."""
    for item in body.outline:
        _guard(item.title, *item.questions, *item.queries)
    _call(service.approve_outline, report_id, [item.model_dump() for item in body.outline])
    return {"status": "researching"}


@router.post("/{report_id}/feedback", status_code=status.HTTP_202_ACCEPTED)
def send_feedback(report_id: str, body: Feedback, service: ResearchService = Depends(get_service)) -> dict[str, str]:
    """Human step 2: a change request. Only the sections it touches are redone."""
    if not body.text.strip() and not body.action:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Say what should change")
    _guard(body.text)
    if not body.action and small_talk(body.text):  # "hi" or "thanks" is not a change request: skip the run
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Say what should change, for example: add a section on government policy."
        )
    _call(service.revise, report_id, body.text, body.section_id, body.action)
    return {"status": "revising"}


@router.post("/{report_id}/resume", status_code=status.HTTP_202_ACCEPTED)
def resume_report(report_id: str, service: ResearchService = Depends(get_service)) -> dict[str, str]:
    _call(service.recover, report_id)
    return {"status": "researching"}


@router.post("/{report_id}/analysis")
def analyse_report(report_id: str, service: ResearchService = Depends(get_service)) -> dict[str, Any]:
    try:
        return _call(service.analyze, report_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Analysis failed: {exc}"[:300]) from None


@router.post("/{report_id}/kb")
def add_to_knowledge_base(
    report_id: str,
    service: ResearchService = Depends(get_service),
    sessions: UserSessions = Depends(get_store),
) -> dict[str, Any]:
    """Index the report and make sure it has a chat session bound to it."""
    report = _call(service.sync_kb, report_id)
    session_id = report.get("session_id")
    try:
        sessions.get(session_id or "")
    except SessionNotFoundError:
        session_id = sessions.create(title=f"Report: {report['topic']}"[:80], report_id=report_id).id
        report = service.store.update(report_id, session_id=session_id)
    return {"session_id": session_id, "kb": report["kb"]}


@router.get("/{report_id}/export.md", response_class=PlainTextResponse)
def export_markdown(report_id: str, service: ResearchService = Depends(get_service)) -> PlainTextResponse:
    report = _call(service.get, report_id)
    headers = {"Content-Disposition": f'attachment; filename="report-{report_id}.md"'}
    return PlainTextResponse(report_markdown(report), media_type="text/markdown", headers=headers)
