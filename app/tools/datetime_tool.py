"""Current date/time tool."""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def current_datetime() -> str:
    """Return the current local date and time in ISO 8601 format.

    Use this whenever the user asks about today's date, the current time, or anything
    relative to "now".
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")
