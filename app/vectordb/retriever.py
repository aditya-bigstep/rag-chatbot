"""Retriever: question -> the most relevant chunks.

RETRIEVAL_MODE picks how:
  dense   vector similarity only
  bm25    keyword match only
  hybrid  both lists merged with reciprocal rank fusion
"""

from __future__ import annotations

from typing import Any, Protocol

from langsmith import traceable

from app.models import RetrievedChunk
from app.vectordb.bm25 import bm25_scores
from app.vectordb.embeddings import Embedder, get_embedder
from app.vectordb.store import VectorStore, get_vector_store

RRF_K = 60


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """IN: a user question, optional k.  OUT: the most relevant chunks."""
        ...


def get_retriever(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
    min_score: float | None = None,
    mode: str | None = None,
) -> Retriever:
    from app.config import get_settings

    settings = get_settings()
    return VectorRetriever(
        store if store is not None else get_vector_store(),
        embedder if embedder is not None else get_embedder(),
        k if k is not None else settings.top_k,
        min_score if min_score is not None else settings.min_score,
        mode or settings.retrieval_mode,
    )


class VectorRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        k: int,
        min_score: float = 0.0,
        mode: str = "dense",
        where: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.k = k
        self.min_score = min_score
        self.mode = mode
        self.where = where

    def scoped(self, where: dict[str, Any]) -> VectorRetriever:
        """Same retriever, limited to chunks whose metadata matches `where` (e.g. one report)."""
        return VectorRetriever(self.store, self.embedder, self.k, self.min_score, self.mode, where)

    def everything(self, max_chars: int = 40_000) -> list[RetrievedChunk]:
        """Every chunk in scope, in document order — for "summarise it", where no search query fits."""
        if not hasattr(self.store, "all_chunks"):
            return []
        chunks = sorted(self.store.all_chunks(self.where), key=lambda c: (c.doc_id, c.position))  # type: ignore[attr-defined]
        hits, size = [], 0
        for chunk in chunks:
            size += len(chunk.text)
            if hits and size > max_chars:
                break
            hits.append(RetrievedChunk(chunk=chunk, score=1.0))
        return hits

    def _dense(self, query: str, k: int) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        if self.where:
            hits = self.store.search(vector, k, where=self.where)  # type: ignore[call-arg]
        else:
            hits = self.store.search(vector, k)
        return [hit for hit in hits if hit.score >= self.min_score]

    def _keyword(self, query: str, k: int) -> list[RetrievedChunk]:
        chunks = self.store.all_chunks(self.where)  # type: ignore[attr-defined]
        scores = bm25_scores(query, [chunk.text for chunk in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in ranked[:k] if score > 0]

    @traceable(run_type="retriever", name="retrieve")
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.k
        can_keyword = hasattr(self.store, "all_chunks")
        if self.mode == "bm25" and can_keyword:
            return self._keyword(query, k)
        if self.mode != "hybrid" or not can_keyword:
            return self._dense(query, k)

        dense = self._dense(query, k * 3)
        if not dense:  # nothing is semantically close: let the bot say "not found" instead of keyword-guessing
            return []
        keyword = self._keyword(query, k * 3)
        return fuse(dense, keyword)[:k]


def fuse(dense: list[RetrievedChunk], keyword: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reciprocal rank fusion. The reported score stays the dense similarity when we have one."""
    points: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for hits in (keyword, dense):  # dense last so its similarity score is the one kept
        for rank, hit in enumerate(hits):
            points[hit.chunk.id] = points.get(hit.chunk.id, 0.0) + 1 / (RRF_K + rank + 1)
            best[hit.chunk.id] = hit
    dense_ids = {hit.chunk.id for hit in dense}
    floor = min((hit.score for hit in dense), default=0.0)
    ordered = sorted(points, key=lambda chunk_id: points[chunk_id], reverse=True)
    results = []
    for chunk_id in ordered:
        hit = best[chunk_id]
        score = hit.score if chunk_id in dense_ids else floor
        results.append(RetrievedChunk(chunk=hit.chunk, score=score))
    return results
