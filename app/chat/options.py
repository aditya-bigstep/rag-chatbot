"""Feature switches for the chat engine.

WHY THIS FILE EXISTS
--------------------
The trainer's evaluator calls `get_engine(llm=..., retriever=..., tools=...)` with a scripted fake
LLM and expects the ORIGINAL behaviour: retrieve on every turn, exactly one LLM call per answer,
the answer returned untouched. If every new feature (routing, guardrails, HITL ...) were always on,
that evaluation could break.

So every new feature is controlled by `EngineOptions`:

  * `EngineOptions()`               -> "contract-safe" defaults: the engine behaves like before.
  * `EngineOptions.from_settings()` -> the real app (app/deps.py) turns the features on from `.env`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import Settings

GroundingMode = Literal["off", "warn", "enforce"]


@dataclass(frozen=True)
class EngineOptions:
    # Feature 2 — route each message (direct / history / retrieval ...) instead of always retrieving.
    routing: bool = False
    # Even with routing on, force retrieval on every turn (the evaluator-friendly switch).
    always_retrieve: bool = False
    # Feature 3 — block prompt-injection / unsafe / invalid input before it reaches the LLM.
    input_guard: bool = False
    # Feature 3 — "off": no check, "warn": log + tag the trace, "enforce": replace unsupported answers.
    grounding: GroundingMode = "warn"
    # Feature 5 — pause (interrupt) for human approval before dangerous tool calls.
    hitl: bool = False
    # Feature 1 — how many recent messages are sent verbatim; older ones are summarised.
    memory_max_messages: int = 10
    memory_summarize: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> EngineOptions:
        mode = settings.grounding_mode if settings.grounding_mode in ("off", "warn", "enforce") else "enforce"
        return cls(
            routing=settings.smart_routing,
            always_retrieve=settings.always_retrieve,
            input_guard=settings.guardrails,
            grounding=mode,  # type: ignore[arg-type]
            hitl=settings.hitl_enabled,
            memory_max_messages=settings.memory_max_messages,
            memory_summarize=settings.memory_summarize,
        )
