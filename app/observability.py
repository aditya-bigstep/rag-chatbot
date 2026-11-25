"""MILESTONE 2 — Observability with LangSmith.

LangChain/LangGraph send traces to LangSmith automatically once tracing is on.
Your job is to switch it on safely from `.env` and make traces easy to find.

  LANGSMITH_TRACING=true          (the older name LANGCHAIN_TRACING_V2=true also counts)
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=rag-chatbot-<your-name>
  LANGSMITH_ENDPOINT=...          (optional; the evaluator points this at a local fake server)

Rules the evaluator checks (against a local fake LangSmith server, no account needed):
  - `configure_tracing()` returns True only when tracing is switched on AND an API key is set,
    and False otherwise (it must never crash the app when keys are missing).
  - A chat turn produces a trace in the project named by LANGSMITH_PROJECT.
  - The trace carries `session_id` in its metadata (see app/chat/engine.py).
  - The trace contains an LLM run, and your retriever appears as a "retriever" run.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


def configure_tracing() -> bool:
    """Called once at app start-up (see app/main.py).

    OUT: True if LangSmith tracing is active, False if it is off or misconfigured.
    """
    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    if tracing != "true":
        return False

    if not api_key:
        logger.warning("LangSmith tracing is on but no API key is set.")
        return False

    os.environ.setdefault("LANGSMITH_PROJECT", "rag-chatbot")
    return True



# --------------------------------------------------------------------------- FEATURE 6: request tracing
#
# With LANGSMITH_TRACING=true every LangGraph node, LLM call, tool call and retriever call already becomes
# a run in LangSmith, and LLM runs carry token usage automatically. The helpers below add the parts
# LangSmith can't know on its own: WHICH route was chosen, WHAT guardrails decided, how many chunks were
# retrieved (and their scores), and per-step latency. Search/filter in LangSmith by these metadata keys.


def build_run_config(session_id: str, options: object | None = None) -> dict[str, Any]:
    """The `config` passed to every graph call. Keeps `session_id` in metadata (the evaluator checks it)
    and adds `thread_id` so the SQLite checkpointer keeps one timeline per chat session."""
    return {
        "run_name": "chat_turn",  # name of the root trace in LangSmith
        "tags": ["rag-chatbot"],
        "metadata": {"session_id": session_id, "engine_options": repr(options)},
        "configurable": {"thread_id": session_id},
    }


def annotate_run(**metadata: Any) -> None:
    """Attach key/values to the CURRENT LangSmith run (no-op when tracing is off or outside a run)."""
    try:
        run = get_current_run_tree()
        if run is not None:
            run.add_metadata(metadata)
    except Exception:  # tracing must never break a chat turn
        logger.debug("Could not annotate run", exc_info=True)


def traced_node(name: str) -> Callable[[F], F]:
    """Decorator for graph nodes: gives the step its own named run, logs latency and logs errors.

    * latency      -> `latency_ms` metadata on the run (and a debug log line)
    * errors       -> logged with the node name, then re-raised (LangSmith marks the run as failed and
                      keeps the traceback, so you can see exactly which step broke)
    """

    def decorator(fn: F) -> F:
        @traceable(name=name, run_type="chain")
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception("Graph node %r failed", name)
                annotate_run(failed_node=name)
                raise
            finally:
                latency_ms = round((time.perf_counter() - started) * 1000, 1)
                annotate_run(latency_ms=latency_ms)
                logger.debug("node %s took %s ms", name, latency_ms)

        return wrapper  # type: ignore[return-value]

    return decorator
