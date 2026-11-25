"""Admin tools that change data. They need the HITL approval step (app/chat/approval.py).

Unlike `calculator`, these need real dependencies (the documents folder and the vector store), so they
are built by a factory instead of being module-level constants. `app/deps.py` calls `make_admin_tools`
and adds them to the engine's tools only when HITL is enabled — they would be unsafe without approval.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from app.ingest.pipeline import load_document
from app.vectordb.store import VectorStore


def make_admin_tools(documents_dir: Path, store: VectorStore | None) -> list[BaseTool]:
    @tool
    def delete_document(filename: str) -> str:
        """Permanently delete an uploaded document (the file AND its searchable index entries).

        Use only when the user explicitly asks to delete or remove a specific document by file name,
        e.g. "delete leave_policy.md". This cannot be undone; a human is asked to approve first.
        """
        # `Path(...).name` drops any folder part, so "../../etc/passwd" can never escape documents_dir.
        path = documents_dir / Path(filename).name
        if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
            return f"Error: no document named {Path(filename).name!r} was found."
        removed = 0
        if store is not None:
            removed = store.delete_document(load_document(path).id)  # same stable id the ingester uses
        path.unlink()
        return f"Deleted {path.name} and {removed} indexed chunk(s)."

    return [delete_document]
