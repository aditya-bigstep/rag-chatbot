"""Put a report into the vector store so the chatbot can answer questions about it.

Each section is stored as its own document, so after an edit only the changed sections
are embedded again (their content hash is compared with what is already stored).
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.models import Document
from app.research.render import number_sources, section_text


def section_doc_id(report_id: str, section_id: str) -> str:
    return f"report-{report_id}-{section_id}"


def sync_report(report: dict[str, Any], chunker: Any, embedder: Any, store: Any) -> dict[str, int]:
    numbers = number_sources(report)
    wanted: set[str] = set()
    embedded = skipped = total = 0

    for item in report["outline"]:
        section = report["sections"].get(item["id"])
        if not section or not section["text"]:
            continue
        doc_id = section_doc_id(report["id"], section["id"])
        wanted.add(doc_id)
        body = f"{section['title']}\n\n{section_text(section, numbers)}"
        content_hash = hashlib.sha256(body.encode()).hexdigest()
        chunks = chunker.chunk(Document(id=doc_id, text=body, source=report["topic"]))
        total += len(chunks)
        if store.get_content_hash(doc_id) == content_hash:
            skipped += 1
            continue
        urls = " ".join(report["sources"][sid]["url"] for sid in section["source_ids"] if sid in report["sources"])
        for chunk in chunks:
            chunk.metadata = {
                "report_id": report["id"],
                "section_id": section["id"],
                "section": section["title"],
                "urls": urls,
                "content_hash": content_hash,
            }
        store.delete_document(doc_id)
        store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
        embedded += 1

    stale = {c.doc_id for c in store.all_chunks({"report_id": report["id"]})} - wanted
    for doc_id in stale:
        store.delete_document(doc_id)
    return {"chunks": total, "sections_embedded": embedded, "sections_unchanged": skipped, "removed": len(stale)}


def remove_report(report_id: str, store: Any) -> None:
    for doc_id in {c.doc_id for c in store.all_chunks({"report_id": report_id})}:
        store.delete_document(doc_id)
