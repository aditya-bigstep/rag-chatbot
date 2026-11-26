"""Progress events for running reports. Kept in memory; the UI polls them."""

from __future__ import annotations

import threading
import time
from typing import Any

_events: dict[str, list[dict[str, Any]]] = {}
_lock = threading.Lock()


def emit(report_id: str, stage: str, message: str, **extra: Any) -> None:
    if not report_id:
        return
    event = {"stage": stage, "message": message, "at": round(time.time(), 2), **extra}
    with _lock:
        _events.setdefault(report_id, []).append(event)


def read(report_id: str, after: int = 0) -> list[dict[str, Any]]:
    with _lock:
        return list(_events.get(report_id, [])[after:])


def clear(report_id: str) -> None:
    with _lock:
        _events.pop(report_id, None)
