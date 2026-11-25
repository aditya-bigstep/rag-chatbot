"""MILESTONE 7 — Retriever.

Glue between a question and the vector store: embed the query, search, and
(optionally) post-process — score threshold, de-duplication, MMR, re-ranking.

Rules the evaluator checks:
  - `retrieve(query)` returns at most `k` RetrievedChunk, best first.
  - `retrieve(query, k=2)` overrides the default k.
  - It uses the embedder and store it was GIVEN (the evaluator passes its own).
  - It shows up in LangSmith as a run with run_type="retriever"
    (hint: `from langsmith import traceable` and `@traceable(run_type="retriever")`).
"""

from __future__ import annotations

from typing import Protocol

from langsmith import traceable

from app.models import RetrievedChunk
from app.vectordb.embeddings import Embedder, get_embedder
from app.vectordb.store import VectorStore, get_vector_store


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """IN: a user question, optional k.  OUT: the most relevant chunks."""
        ...


def get_retriever(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
    min_score: float | None = None,
) -> Retriever:
    """Build a retriever.

    IN:  store / embedder to use (None -> build the defaults with
         get_vector_store() / get_embedder()), default k (None -> settings.top_k),
         minimum score (None -> settings.min_score).
    OUT: an object with `.retrieve(query, k=None)`.

    """
    from app.config import get_settings

    return VectorRetriever(
        store if store is not None else get_vector_store(),
        embedder if embedder is not None else get_embedder(),
        k if k is not None else get_settings().top_k,
        min_score if min_score is not None else get_settings().min_score,
    )


class VectorRetriever:
    """Embeds the question and returns the closest chunks from the store."""

    def __init__(self, store: VectorStore, embedder: Embedder, k: int, min_score: float = 0.0) -> None:
        self.store = store
        self.embedder = embedder
        self.k = k
        self.min_score = min_score

    @traceable(run_type="retriever", name="retrieve")
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        hits = self.store.search(self.embedder.embed_query(query), k or self.k)
        # Similarity threshold: drop weak matches so the LLM never sees chunks that barely
        # relate to the question. If nothing is good enough we return [], and the chatbot
        # says it couldn't find the answer instead of guessing from irrelevant text.
        return [hit for hit in hits if hit.score >= self.min_score]
