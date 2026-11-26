"""One JSON file per report. The graph checkpoint is the working state; this is what the UI reads."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[a-f0-9]{12}$")


class ReportNotFoundError(KeyError):
    pass


class ReportStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def _path(self, report_id: str) -> Path:
        if not _ID.match(report_id):
            raise ReportNotFoundError(report_id)
        return self.directory / f"{report_id}.json"

    def get(self, report_id: str) -> dict[str, Any]:
        path = self._path(report_id)
        if not path.exists():
            raise ReportNotFoundError(report_id)
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def save(self, report: dict[str, Any]) -> None:
        path = self._path(report["id"])
        with self.lock:
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=1, ensure_ascii=False)
            os.replace(tmp, path)

    def update(self, report_id: str, **fields: Any) -> dict[str, Any]:
        with self.lock:
            report = self.get(report_id)
            report.update(fields)
            self.save(report)
            return report

    def list(self) -> list[dict[str, Any]]:
        reports = []
        for path in self.directory.glob("*.json"):
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(reports, key=lambda r: r.get("created_at", ""), reverse=True)

    def delete(self, report_id: str) -> None:
        path = self._path(report_id)
        if not path.exists():
            raise ReportNotFoundError(report_id)
        path.unlink()
