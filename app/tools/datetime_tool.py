"""Current date/time tool."""

from __future__ import annotations

from datetime import datetime, timedelta

from langchain_core.tools import tool


@tool
def current_datetime() -> str:
    """Return the current local date and time in ISO 8601 format.

    Use this whenever the user asks about today's date, the current time, or anything
    relative to "now".
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


@tool
def date_info(offset_days: int = 0) -> str:
    """Return a calendar date WITH its weekday, e.g. "Friday, 27 November 2026".

    offset_days = 0 for today, 1 for tomorrow, -1 for yesterday, 7 for a week from now, and so on.
    Use this for "what day is it", "what is tomorrow's date", "which day was it 3 days ago".
    Never work out a weekday yourself.
    """
    if not -36500 <= offset_days <= 36500:
        return "Error: offset_days is out of range"
    return (datetime.now().astimezone() + timedelta(days=offset_days)).strftime("%A, %d %B %Y")
