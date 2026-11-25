"""MILESTONE 6 — Ingestion: file -> Document -> Chunks -> vectors -> store.

Used in two places:
  - `uv run python -m app.ingest` indexes everything in data/documents/.
  - The upload endpoint (app/api/documents.py) indexes one uploaded file.

Rules the evaluator checks (with its own chunker/embedder/store):
  - `load_document` supports .md and .txt; anything else raises ValueError.
  - `Document.id` is stable for the same file name (re-loading gives the same id).
  - `ingest_file` returns the number of chunks stored, and the store grows by that much.
  - Ingesting the same file twice does NOT create duplicates
    (hint: `store.delete_document(doc.id)` before adding).
  - `ingest_directory` ingests every supported file and returns the total chunk count.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.models import Document
from app.vectordb.chunking import Chunker
from app.vectordb.embeddings import Embedder
from app.vectordb.store import VectorStore

SUPPORTED_SUFFIXES = (".md", ".txt")


def load_document(path: Path) -> Document:
    # Takes a file and wraps it up as a Document. We only handle .md and .txt,
    # so anything else gets rejected right away.
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {path.suffix!r}; use one of {SUPPORTED_SUFFIXES}")
    # Grab all the text from the file.
    text = path.read_text(encoding="utf-8")
    # The id comes from the file name, so loading the same file again gives the same id.
    doc_id = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    return Document(id=doc_id, text=text, source=path.name, metadata={"suffix": path.suffix.lower()})

## switched sha1 to sha256 -> both are deterministic( same input -> same output) [useful for reidentifying documents]
## hexdigest() converts hash to human readable



def ingest_file(path: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    # Gets one file into the vector store so the chatbot can search it later.
    # First, read the file into a Document.
    document = load_document(path)
    # Hash the text, so we can tell if the content changed since the last time.
    content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    # Same content as the stored copy? Skip chunking and embedding (saves the embedding cost).
    # Not every store has get_content_hash (it is our own extra, not part of the
    # base contract), so only use the shortcut when the store supports it.
    get_hash = getattr(store, "get_content_hash", None)
    if get_hash is not None and get_hash(document.id) == content_hash:
        return len(chunker.chunk(document))
    # Break it into small chunks.
    chunks = chunker.chunk(document)
    # Remember the hash on every chunk, so the next upload can be compared against it.
    for c in chunks:
        c.metadata["content_hash"] = content_hash
    # Turn each chunk into a vector, sending them all in one go.
    vectors = embedder.embed_documents([c.text for c in chunks])
    # Wipe any older copy of this file first, otherwise re-running would leave duplicates.
    store.delete_document(document.id)
    # Now save the fresh chunks with their vectors.
    if chunks:
        store.add(chunks, vectors)
    # Let the caller know how many chunks went in.
    return len(chunks)

## checks if there is something in chunks -> if yes -> store vector and chunks of that output
## return length of how many saved



def ingest_directory(directory: Path, *, chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    # Ingests every .md/.txt file in a folder.
    # Pick out just the supported files and go through them in name order.
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    # Ingest them one by one and add up the chunk counts.
    return sum(ingest_file(p, chunker=chunker, embedder=embedder, store=store) for p in files)

