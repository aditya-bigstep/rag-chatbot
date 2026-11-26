"""MILESTONE 5 — Vector store.

Store chunk vectors on disk and find the nearest ones to a query vector.

Use any backend: Chroma, FAISS, LanceDB, Qdrant (local mode) ... or your own
NumPy + JSON file. Add the package with `uv add <pkg>`. Data must live under
`persist_dir` (default: settings.vector_dir) — never a hard-coded path.

Rules the evaluator checks:
  - `search(v, k)` returns at most k hits, best first (scores descending).
  - Searching with a stored chunk's own vector returns that chunk first.
  - `add` with an existing chunk id REPLACES it (no duplicates).
  - A new store opened on the same `persist_dir` sees the same data.
  - `delete_document(doc_id)` removes all chunks of that document.
  - Searching an empty store returns [].
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import chromadb

from app.config import get_settings
from app.models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """IN: chunks and their vectors (same length and order). Upsert by chunk.id."""
        ...

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        """IN: a query vector and k.  OUT: up to k RetrievedChunk, highest score first."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk whose doc_id matches. OUT: how many were removed."""
        ...

    def count(self) -> int:
        """How many chunks are stored."""
        ...

    def get_content_hash(self, doc_id: str) -> str | None:
        """The content hash saved with this document's chunks, or None if it isn't stored."""
        ...


class ChromaVectorStore:
    """Keeps chunks and their vectors in Chroma, a small database that saves itself to disk."""

    def __init__(self, persist_dir: Path, collection: str = "chunks") -> None:
        """Open the Chroma database in `persist_dir` (it is created the first time).

        This is what makes the data permanent: when the app restarts, it opens the same
        folder and finds every chunk that was saved before.
        """
        # PersistentClient saves everything inside persist_dir, so data survives a restart.
        client = chromadb.PersistentClient(path=str(persist_dir))
        # "cosine" tells Chroma to compare vectors by direction, same as we did before.
        self.collection = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Save chunks together with their vectors (embeddings).

        Used during ingestion, after documents are split and embedded. Saving a chunk
        whose id already exists replaces the old one, so re-ingesting an updated
        document never leaves duplicates behind.
        """
        if not chunks:
            return
        # upsert = "insert, or replace if the id already exists", so we never get duplicates.
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            # Chroma only accepts simple values here, so we spread the small extras (like
            # content_hash) next to doc_id, source and position instead of nesting them.
            metadatas=[{"doc_id": c.doc_id, "source": c.source, "position": c.position, **c.metadata} for c in chunks],
        )

    def search(
        self, query_embedding: list[float], k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Find the k chunks whose meaning is closest to the question's vector.

        This is the "retrieval" step of RAG: the chatbot passes the returned chunks to
        the LLM so it can answer from your documents. Best match comes first, and the
        score is higher when the chunk is more relevant.
        """
        count = self.collection.count()
        if k <= 0 or count == 0:  # nothing to look through
            return []
        # Chroma hands back the closest chunks first, so no sorting is needed.
        result = self.collection.query(query_embeddings=[query_embedding], n_results=min(k, count), where=where)
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            # Chroma gives a distance (smaller = closer); we want a score (bigger = better).
            hits.append(RetrievedChunk(chunk=_to_chunk(chunk_id, text, meta), score=1 - distance))
        return hits

    def all_chunks(self, where: dict[str, Any] | None = None) -> list[Chunk]:
        """Every stored chunk (optionally filtered). Keyword search needs the raw texts."""
        found = self.collection.get(where=where)
        rows = zip(found["ids"], found["documents"], found["metadatas"], strict=True)
        return [_to_chunk(chunk_id, text, meta) for chunk_id, text, meta in rows]

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk that belongs to one document and return how many were removed.

        Helps keep answers fresh: when a file is deleted or changed, its old chunks must
        go so the chatbot never quotes outdated text.
        """
        # Find this document's chunks first so we can report how many we removed.
        ids = self.collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """Return how many chunks are saved in total (handy for status checks and tests)."""
        return self.collection.count()

    def get_content_hash(self, doc_id: str) -> str | None:
        """Return the saved content hash (a fingerprint of the file's text) for a document.

        Ingestion compares it with the file's current hash: if they match, the file hasn't
        changed and can be skipped, which saves time and embedding costs. Returns None if
        the document isn't stored yet.
        """
        # Every chunk of a document shares the same hash, so looking at one is enough.
        found = self.collection.get(where={"doc_id": doc_id}, limit=1)
        if not found["metadatas"]:
            return None
        return found["metadatas"][0].get("content_hash")


def _to_chunk(chunk_id: str, text: str, meta: Any) -> Chunk:
    extra = dict(meta)
    return Chunk(
        id=chunk_id,
        text=text,
        doc_id=extra.pop("doc_id"),
        source=extra.pop("source"),
        position=extra.pop("position"),
        metadata=extra,
    )


def collection_name(embedding_model: str) -> str:
    """One collection per embedding model, so switching models never mixes vector sizes."""
    slug = re.sub(r"[^a-z0-9]+", "-", embedding_model.lower()).strip("-")
    return f"chunks-{slug}"[:60].rstrip("-") if slug else "chunks"


def get_vector_store(persist_dir: str | Path | None = None) -> VectorStore:
    """Open (or create) the vector store saved in `persist_dir` (default: settings.vector_dir)."""
    settings = get_settings()
    return ChromaVectorStore(
        Path(persist_dir) if persist_dir else settings.vector_dir, collection_name(settings.embedding_model)
    )
