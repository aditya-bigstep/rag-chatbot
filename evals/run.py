"""Run the fixed test topics through the whole system and score them.

    uv run python -m evals.run --label v2                     # full system
    uv run python -m evals.run --label v1 --no-verify --retrieval dense   # baseline

Results go to evals/results/<label>.json. A topic that is already in the file is skipped,
so the run can be stopped and continued (useful with free-tier rate limits).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.chat.engine import get_engine
from app.chat.llm import get_llm
from app.chat.options import EngineOptions
from app.config import load_settings
from app.observability import configure_tracing
from app.research.deps import build_deps
from app.research.graph import ResearchGraph
from app.research.kb import remove_report, sync_report
from app.research.usage import UsageTracker
from app.vectordb.chunking import get_chunker
from app.vectordb.embeddings import get_embedder
from app.vectordb.retriever import get_retriever
from app.vectordb.store import ChromaVectorStore
from evals import judges, metrics

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("evals")


def run_topic(topic: str, graph: Any, label: str, max_calls: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Plan, auto-approve the outline, research. Returns (report, system metrics)."""
    report_id = uuid.uuid4().hex[:12]
    tracker = UsageTracker(max_calls)
    config = {
        "configurable": {"thread_id": f"eval-{report_id}"},
        "callbacks": [tracker],
        "max_concurrency": 2,
        "run_name": f"eval_{label}",
        "tags": ["researchpilot", "eval", label],
        "metadata": {"eval_label": label, "topic": topic},
    }
    started = time.time()
    state = graph.invoke({"report_id": report_id, "topic": topic}, config=config)
    state = graph.invoke(Command(resume={"outline": state["outline"]}), config=config)
    seconds = time.time() - started

    ids = [item["id"] for item in state["outline"]]
    sections = {sid: state["sections"][sid] for sid in ids if sid in state["sections"]}
    report = {
        "id": report_id,
        "topic": topic,
        "outline": state["outline"],
        "sections": sections,
        "sources": state["sources"],
    }
    stats = [s["stats"] for s in sections.values()]
    system = {
        "latency_seconds": round(seconds, 1),
        "search_calls": sum(s.get("search_calls", 0) for s in stats),
        "pages_read": sum(s.get("pages", 0) for s in stats),
        "sentences_removed": sum(s.get("dropped", 0) for s in stats),
        **tracker.summary(),
    }
    return report, system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--topics", type=int, default=12)
    parser.add_argument("--sections", type=int, default=3)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--retrieval", default="hybrid", choices=["dense", "bm25", "hybrid"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    configure_tracing()
    settings = replace(
        load_settings(),
        research_max_sections=args.sections,
        research_verify=not args.no_verify,
        retrieval_mode=args.retrieval,
    )
    embedder = get_embedder()
    deps = build_deps(settings, embedder)
    graph = ResearchGraph(deps).build(MemorySaver())
    store = ChromaVectorStore(ROOT / "results" / "vectors", f"eval-{args.label}")
    retriever = get_retriever(
        store=store, embedder=embedder, k=settings.top_k, min_score=settings.min_score, mode=args.retrieval
    )
    engine = get_engine(llm=get_llm(), retriever=retriever, tools=[], options=EngineOptions(grounding="enforce"))
    judge = judges.get_judge()

    out_file = ROOT / "results" / f"{args.label}.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {"config": {}, "topics": []}
    results["config"] = {
        "verify": not args.no_verify,
        "retrieval": args.retrieval,
        "sections": args.sections,
        "model": settings.llm_model,
        "judge": getattr(judge, "model_name", ""),
    }
    done = {row["topic"] for row in results["topics"] if "error" not in row}
    results["topics"] = [row for row in results["topics"] if "error" not in row]

    for entry in json.loads((ROOT / "topics.json").read_text())[: args.topics]:
        if entry["topic"] in done:
            continue
        logger.info("%s | %s", args.label, entry["topic"])
        row: dict[str, Any] = {"topic": entry["topic"]}
        try:
            report, row["system"] = run_topic(entry["topic"], graph, args.label, settings.research_max_llm_calls)
            row["report"] = metrics.report_metrics(report, entry["aspects"], judge)
            sync_report(report, get_chunker(), embedder, store)
            row["chat"] = metrics.chat_metrics(report, engine, judge)
            remove_report(report["id"], store)
        except Exception as exc:
            logger.exception("Topic failed")
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        results["topics"].append(row)
        out_file.write_text(json.dumps(results, indent=1))
    logger.info("Saved %s", out_file)


if __name__ == "__main__":
    main()
