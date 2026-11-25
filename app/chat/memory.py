"""FEATURE 1 — Conversation memory.

Memory = "what the model is shown from earlier in THIS chat session".

  * Source of truth: the per-session JSON files (app/storage/json_store.py). They already hold every
    message, so we build on them instead of storing a second copy.
  * Recent messages (the last `max_messages`) are sent to the LLM word for word.
  * Older messages are squeezed into a short SUMMARY (one LLM call, cached) so long chats keep their
    context without growing the prompt forever.

Memory vs. LangGraph checkpointing (app/chat/checkpointing.py) — they overlap but are NOT the same:

  * Memory       -> decides which PAST MESSAGES the model sees. Works without any checkpointer.
  * Checkpointing-> saves the graph's STATE (where a run is, pending interrupts) so a paused/crashed run
                    can resume. It *could* also hold the chat history, but we deliberately don't rely on
                    it for that: every turn the engine overwrites the checkpoint's messages with the JSON
                    history (see `GraphEngine._inputs`), so there is exactly one source of truth.

Follow-ups such as "Explain the second point again" work because the previous answer is in the messages
the LLM sees (this module) and because the router rewrites the search query using that history
(`rewrite_query` node in engine.py).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.models import Message

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "Summarise the earlier part of this conversation in at most 120 words. Keep names, numbers, the "
    "user's goals and any decisions. Write only the summary."
)
_FALLBACK_CHARS = 600  # if the LLM cannot summarise, keep this many characters of the old text


@dataclass(frozen=True)
class MemoryView:
    """What the engine puts into the prompt for one turn."""

    recent: list[Message]  # sent verbatim
    summary: str  # "" when the chat is still short


class ConversationMemory:
    def __init__(self, llm: BaseChatModel | None, max_messages: int = 10, summarize: bool = False) -> None:
        self.llm = llm
        self.max_messages = max(2, max_messages)
        self.summarize = summarize
        self._cache: dict[str, str] = {}  # hash of the old messages -> summary (avoids re-summarising)

    def build(self, history: list[Message]) -> MemoryView:
        """Split history into (old -> summary, recent -> verbatim)."""
        if len(history) <= self.max_messages:
            return MemoryView(recent=list(history), summary="")
        old, recent = history[: -self.max_messages], history[-self.max_messages :]
        return MemoryView(recent=recent, summary=self._summarise(old))

    def _summarise(self, old: list[Message]) -> str:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in old)
        if not (self.summarize and self.llm is not None):
            return transcript[-_FALLBACK_CHARS:]  # cheap fallback: just the tail of the old text
        key = hashlib.sha256(transcript.encode()).hexdigest()
        if key not in self._cache:
            try:
                messages: list[BaseMessage] = [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=transcript)]
                reply = self.llm.invoke(messages)
                self._cache[key] = reply.content if isinstance(reply.content, str) else transcript[-_FALLBACK_CHARS:]
            except Exception:  # memory must never break the chat
                logger.exception("Could not summarise conversation; using a plain excerpt")
                self._cache[key] = transcript[-_FALLBACK_CHARS:]
        return self._cache[key]
