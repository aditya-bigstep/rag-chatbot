"""Runs the research and analysis graphs in the background and keeps the report file up to date."""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.types import Command

from app.config import Settings
from app.models import utc_now
from app.research import events
from app.research.analysis import AnalysisGraph, report_fingerprint
from app.research.deps import Deps
from app.research.graph import ResearchGraph
from app.research.kb import remove_report, sync_report
from app.research.render import number_sources
from app.research.store import ReportStore
from app.research.usage import UsageTracker

logger = logging.getLogger(__name__)
BUSY = ("planning", "researching", "revising")


class ResearchService:
    def __init__(self, settings: Settings, deps: Deps, checkpointer: Any, chunker: Any, vector_store: Any) -> None:
        self.settings = settings
        self.deps = deps
        self.chunker = chunker
        self.vector_store = vector_store
        self.store = ReportStore(settings.research_dir / "reports")
        self.graph = ResearchGraph(deps).build(checkpointer)
        self.analysis_graph = AnalysisGraph(deps).build()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.active: set[str] = set()

    # ------------------------------------------------------------------ reading
    def get(self, report_id: str) -> dict[str, Any]:
        report = self.store.get(report_id)
        if report["status"] in BUSY and report_id not in self.active:
            report["status"] = "interrupted"  # the server restarted mid-run; the checkpoint can resume it
        report["numbers"] = number_sources(report)
        analysis = report.get("analysis") or {}
        report["analysis_stale"] = bool(analysis) and analysis.get("fingerprint") != report_fingerprint(report)
        return report

    def summaries(self) -> list[dict[str, Any]]:
        keys = ("id", "topic", "status", "created_at", "updated_at", "trust")
        rows = []
        for report in self.store.list():
            row = {key: report.get(key) for key in keys}
            row["sections"] = len(report.get("outline", []))
            row["sources"] = len(number_sources(report))
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ actions
    def create(self, topic: str) -> dict[str, Any]:
        report_id = uuid.uuid4().hex[:12]
        report: dict[str, Any] = {
            "id": report_id,
            "topic": topic.strip(),
            "status": "planning",
            "error": "",
            "outline": [],
            "sections": {},
            "sources": {},
            "history": [],
            "runs": [],
            "analysis": None,
            "kb": None,
            "session_id": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        self.store.save(report)
        self._submit(report_id, {"report_id": report_id, "topic": report["topic"]}, "plan")
        return report

    def approve_outline(self, report_id: str, outline: list[dict[str, Any]]) -> None:
        self._expect(report_id, "awaiting_outline")
        self.store.update(report_id, status="researching")
        self._submit(report_id, Command(resume={"outline": outline}), "research")

    def revise(self, report_id: str, text: str, section_id: str = "", action: str = "") -> None:
        self._expect(report_id, "ready")
        self.store.update(report_id, status="revising")
        answer = {"text": text, "section_id": section_id, "action": action}
        self._submit(report_id, Command(resume=answer), "revision")

    def recover(self, report_id: str) -> None:
        """Continue a run that died with the server. LangGraph picks up from the last checkpoint."""
        self._expect(report_id, "interrupted")
        self.store.update(report_id, status="researching")
        self._submit(report_id, None, "recovery")

    def delete(self, report_id: str) -> None:
        self.store.delete(report_id)
        events.clear(report_id)
        if self.vector_store is not None:
            remove_report(report_id, self.vector_store)

    def analyze(self, report_id: str) -> dict[str, Any]:
        report = self._expect(report_id, "ready")
        tracker = UsageTracker()
        started = time.time()
        config = {
            "callbacks": [tracker],
            "run_name": "analysis",
            "tags": ["researchpilot", "analysis"],
            "metadata": {"report_id": report_id},
        }
        result = self.analysis_graph.invoke({"report": report}, config=config)
        run = {"kind": "analysis", "at": utc_now(), "seconds": round(time.time() - started, 1), **tracker.summary()}
        self.store.update(report_id, analysis=result["analysis"], runs=[*report["runs"], run])
        return self.get(report_id)

    def sync_kb(self, report_id: str) -> dict[str, Any]:
        report = self._expect(report_id, "ready")
        if self.vector_store is None or self.chunker is None:
            raise RuntimeError("The vector store is not available")
        result = sync_report(report, self.chunker, self.deps.embedder, self.vector_store)
        return self.store.update(report_id, kb={**result, "synced_at": utc_now()})

    # ------------------------------------------------------------------ internals
    def _expect(self, report_id: str, status: str) -> dict[str, Any]:
        report = self.get(report_id)
        if report["status"] != status:
            raise ValueError(f"Report is {report['status']}, expected {status}")
        return report

    def _config(self, report_id: str, kind: str, tracker: UsageTracker) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": f"research-{report_id}"},
            "callbacks": [tracker],
            "max_concurrency": self.settings.research_parallel_sections,
            "recursion_limit": 60,
            "run_name": f"research_{kind}",
            "tags": ["researchpilot", kind],
            "metadata": {"report_id": report_id, "verify": self.settings.research_verify},
        }

    def _submit(self, report_id: str, graph_input: Any, kind: str) -> None:
        self.active.add(report_id)
        self.pool.submit(self.run, report_id, graph_input, kind)

    def run(self, report_id: str, graph_input: Any, kind: str) -> None:
        """One pass of the graph, until it finishes or stops at a human step."""
        tracker = UsageTracker(self.settings.research_max_llm_calls)
        config = self._config(report_id, kind, tracker)
        started = time.time()
        error = ""
        try:
            for _update in self.graph.stream(graph_input, config=config, stream_mode="updates"):
                pass
        except Exception as exc:
            logger.exception("Research run failed for %s", report_id)
            error = f"{type(exc).__name__}: {exc}"[:400]
            events.emit(report_id, "error", error)
        finally:
            self.active.discard(report_id)
        self._save_snapshot(report_id, config, kind, tracker, time.time() - started, error)

    def _save_snapshot(
        self, report_id: str, config: dict[str, Any], kind: str, tracker: UsageTracker, seconds: float, error: str
    ) -> None:
        snapshot = self.graph.get_state(config)
        state = snapshot.values
        waiting = [i.value.get("kind") for task in snapshot.tasks for i in task.interrupts if isinstance(i.value, dict)]

        report = self.store.get(report_id)
        outline = state.get("outline", [])
        ids = [item["id"] for item in outline]
        sections = {sid: s for sid, s in state.get("sections", {}).items() if sid in ids}
        used = {sid for s in sections.values() for sid in s["source_ids"]}
        scores = [s["trust"] for s in sections.values() if s["text"]]

        if error:
            status = "failed" if not sections else "ready"
        elif state.get("blocked"):
            status, error = "blocked", state["blocked"]
        elif "outline" in waiting:
            status = "awaiting_outline"
        elif "feedback" in waiting:
            status = "ready"
        else:
            status = "interrupted"

        # Work done by this run = the stats of the sections named in the newest history entry.
        history = state.get("history", [])
        changed = history[-1]["changed"] if history and kind != "plan" else []
        stats = [sections[c["id"]]["stats"] for c in changed if c["id"] in sections]
        run = {
            "kind": kind,
            "at": utc_now(),
            "seconds": round(seconds, 1),
            "search_calls": sum(s.get("search_calls", 0) for s in stats),
            "pages_read": sum(s.get("pages", 0) for s in stats),
            "sections_run": len(stats),
            **tracker.summary(),
        }
        report.update(
            topic=state.get("topic", report["topic"]),
            status=status,
            error=error,
            outline=outline,
            sections=sections,
            sources={sid: s for sid, s in state.get("sources", {}).items() if sid in used},
            history=state.get("history", []),
            runs=[*report["runs"], run],
            trust=round(sum(scores) / len(scores), 2) if scores else None,
            updated_at=utc_now(),
        )
        self.store.save(report)
        if status == "ready" and report.get("kb") and self.vector_store is not None:
            try:  # the report is already in the knowledge base: keep it in step with the edit
                result = sync_report(report, self.chunker, self.deps.embedder, self.vector_store)
                self.store.update(report_id, kb={**result, "synced_at": utc_now()})
            except Exception:
                logger.exception("Could not refresh the knowledge base for %s", report_id)
