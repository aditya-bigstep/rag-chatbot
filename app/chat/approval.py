"""FEATURE 5 — Human-in-the-loop (HITL) approval.

WHERE IT IS USED (and why only here)
------------------------------------
Asking a human to approve every ordinary question would make the chatbot unusable. HITL is valuable
where an action is DESTRUCTIVE or IRREVERSIBLE. In this pipeline that is the tool step: the LLM may ask
to call a tool, and for dangerous tools (`SENSITIVE_TOOLS`, e.g. `delete_document`, which removes a file
AND its vectors) a person must say yes first.

HOW IT WORKS
------------
    generate ──(LLM asked for tools)──> approval ──approved──> tools ──> generate
                                            └──rejected──> generate (told "the user declined")

The `approval` node calls LangGraph's `interrupt(payload)`:
  1. the graph STOPS and the checkpointer (SQLite) saves the state,
  2. the engine returns `EngineResult.pending_approval = payload` to the API/UI,
  3. the human clicks Approve/Reject -> POST /api/sessions/{id}/approval,
  4. the engine resumes with `Command(resume={"approved": bool})`; `interrupt()` now RETURNS that value.

Important gotcha: on resume LangGraph re-runs the interrupted node from its first line, so nothing with
side effects may run before `interrupt()` in the node. (Here nothing does.)

Other good places to add HITL later: low-confidence answers on policy/legal topics, uploading a document
that replaces an existing one, and "send this email"-style tools.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.types import interrupt

# Tools that must never run without a human saying yes.
SENSITIVE_TOOLS = frozenset({"delete_document"})

REJECTED_TEXT = "The user rejected this action, so it was NOT performed. Tell the user nothing was changed."


def sensitive_calls(message: AnyMessage) -> list[Any]:
    """The tool calls in an AI message that need approval (empty list for any other message)."""
    if not isinstance(message, AIMessage):
        return []
    return [c for c in message.tool_calls if c["name"] in SENSITIVE_TOOLS]


def needs_approval(message: AnyMessage) -> bool:
    return bool(sensitive_calls(message))


def request_approval(message: AnyMessage) -> tuple[bool, list[ToolMessage]]:
    """Pause the graph and wait for a human decision.

    Returns (approved, rejection_messages). On rejection the messages answer EVERY tool call in the
    AI message (providers require each tool call to get a reply) so the LLM can explain what happened.
    """
    calls = sensitive_calls(message)
    decision = interrupt(  # <-- the graph stops here until someone resumes it
        {
            "kind": "tool_approval",
            "message": "I need your approval before running: "
            + "; ".join(f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})" for c in calls),
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    if approved:
        return True, []
    assert isinstance(message, AIMessage)
    return False, [ToolMessage(content=REJECTED_TEXT, tool_call_id=c["id"], name=c["name"]) for c in message.tool_calls]
