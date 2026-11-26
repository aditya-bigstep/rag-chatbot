"""Ask a chat model for JSON and validate it. Works with any provider (no tool-calling needed)."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part if isinstance(part, str) else str(part.get("text", "")) for part in content)


def extract_json(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start : end + 1])


def ask_json(llm: Any, system: str, user: str, schema: type[T]) -> T:
    messages = [SystemMessage(content=system + "\nReply with one JSON object only."), HumanMessage(content=user)]
    last_error = ""
    for _attempt in range(2):
        reply = message_text(llm.invoke(messages))
        try:
            return schema.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:300]
            logger.warning("Model returned invalid JSON for %s: %s", schema.__name__, last_error)
            messages = [*messages, HumanMessage(content=f"That was not valid ({last_error}). Send the JSON again.")]
    raise ValueError(f"Model did not return valid {schema.__name__}: {last_error}")


def ask_text(llm: Any, system: str, user: str) -> str:
    return message_text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])).strip()
