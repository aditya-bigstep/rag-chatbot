"""FEATURE 4 — LangGraph checkpointing (persistent, SQLite).

A *checkpoint* is a snapshot of the graph's state saved after each step ("super-step"). It is keyed by
`thread_id`; we use the chat `session_id` as the thread id, so every chat session has its own timeline.

What it gives us:
  * Human-in-the-loop: when a node calls `interrupt()` the graph stops, the snapshot is written to disk,
    and the run can be resumed later (even after a server restart) with `Command(resume=...)`.
  * Crash recovery / inspection: `graph.get_state(config)` shows where a thread stopped;
    `graph.get_state_history(config)` shows every step (handy time-travel debugging).

The data lives in `data/checkpoints.sqlite` (Settings.checkpoint_db). Delete the file to reset.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer(db_path: Path) -> SqliteSaver:
    """Open (or create) the SQLite checkpoint database and return a LangGraph checkpointer.

    `check_same_thread=False` is needed because FastAPI runs our sync endpoints in a thread pool;
    SqliteSaver serialises its own writes with an internal lock.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    # Our own classes stored in the graph state must be allow-listed, or newer LangGraph versions refuse to
    # load them back from the checkpoint (only these exact types may be rebuilt).
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("app.models", "Source")])
    saver = SqliteSaver(connection, serde=serde)
    saver.setup()  # creates the tables if they do not exist yet
    return saver
