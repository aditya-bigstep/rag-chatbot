"""JSON API for chatting: POST /api/sessions/{id}/messages."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.deps import Components, get_components, get_store
from app.models import Message
from app.services import resolve_approval, run_chat_turn, run_chat_turn_stream
from app.storage.json_store import SessionNotFoundError, SessionStore

router = APIRouter(prefix="/api/sessions", tags=["chat"])


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class ChatTurn(BaseModel):
    user_message: Message
    assistant_message: Message


@router.post("/{session_id}/messages", response_model=ChatTurn)
def send_message(
    session_id: str,
    body: SendMessage,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> ChatTurn:
    try:
        user_message, assistant_message = run_chat_turn(store, components.engine, session_id, body.content)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return ChatTurn(user_message=user_message, assistant_message=assistant_message)


class ApprovalDecision(BaseModel):
    approved: bool


@router.get("/{session_id}/approval")
def get_pending_approval(
    session_id: str,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> dict[str, object]:
    """HITL: is this chat paused waiting for a human decision? Returns {"pending": null} when not."""
    try:
        store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return {"pending": components.engine.pending_approval(session_id)}


@router.post("/{session_id}/approval", response_model=Message)
def decide_approval(
    session_id: str,
    body: ApprovalDecision,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> Message:
    """HITL: approve or reject the paused action; the graph resumes and the final reply is saved."""
    try:
        return resolve_approval(store, components.engine, session_id, body.approved)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    except LookupError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/{session_id}/messages/stream")
def stream_message(
    session_id: str,
    body: SendMessage,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> StreamingResponse:
    """Server-sent events: `user` (the saved, PII-masked user message), `token` (text piece) ...
    then `done` (the saved assistant message)."""
    try:
        store.get(session_id)  # 404 now, before the stream starts
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None

    def events() -> Iterator[str]:
        for item in run_chat_turn_stream(store, components.engine, session_id, body.content):
            if isinstance(item, Message):
                yield _sse("user" if item.role == "user" else "done", item.model_dump())
            else:
                yield _sse("token", item)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
