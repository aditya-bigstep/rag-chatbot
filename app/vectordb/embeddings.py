"""MILESTONE 4 — Embeddings.

Turn text into vectors so similar meaning -> nearby vectors.

Pick any embedding model: a hosted API (OpenAI, Voyage, Cohere ...) or a local
one (sentence-transformers, fastembed ...). Add its package with `uv add <pkg>`
and read the model name from `EMBEDDING_MODEL` in `.env`.

Rules the evaluator checks:
  - `embed_documents(texts)` returns one vector per text, in the same order.
  - Every vector has length `dimension`.
  - `embed_documents([])` returns `[]`.
  - The same text always gives (almost) the same vector.
  - `embed_query(t)` is close to `embed_documents([t])[0]`.
  - Related sentences are closer (cosine) than unrelated ones.
"""

from __future__ import annotations

from functools import cached_property, lru_cache

from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from app.config import get_settings


def _summarize_vectors(vectors: list[list[float]]) -> dict[str, int]:
    """Log only the size of the result in LangSmith, because the raw numbers are huge and unreadable."""
    return {"vector_count": len(vectors), "dimension": len(vectors[0]) if vectors else 0}


def _summarize_vector(vector: list[float]) -> dict[str, int]:
    """Same as above, for a single vector."""
    return {"dimension": len(vector)}


class Embedder:
    def __init__(self, model: str) -> None:
        # We use OpenAI's embedding model through LangChain so we don't write the API calls ourselves.
        # chunk_size=100 means "send at most 100 texts per API request", which keeps each request
        # small enough to avoid size/rate limits when embedding many chunks.
        self._model = OpenAIEmbeddings(model=model, chunk_size=100)

    @cached_property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""
        # Different models produce different vector lengths, so instead of hard-coding a number
        # we embed a dummy word and measure it. cached_property makes sure we only pay for this
        # API call once; after that the answer is remembered.
        return len(self.embed_query("example"))

    ## "example" is a sample word to embedd to get to know the len of vector that the
    ## model will provide

    # @traceable makes this call show up in LangSmith as an "embedding" run, so you can see
    # how long it took and what was sent. Inside a chat turn it appears under the retriever run.
    @traceable(run_type="embedding", name="embed_documents", process_outputs=_summarize_vectors)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """IN: list of texts.  OUT: list of vectors (same length and order)."""
        # If there is nothing to embed, return [] right away instead of making a pointless API call.
        return self._model.embed_documents(texts) if texts else []

    # Same idea as above, for the single question typed by the user.
    @traceable(run_type="embedding", name="embed_query", process_outputs=_summarize_vector)
    def embed_query(self, text: str) -> list[float]:
        """IN: one search query.  OUT: one vector."""
        # The user's question must be embedded by the same model as the documents,
        # otherwise the vectors live in different "spaces" and can't be compared.
        return self._model.embed_query(text)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the configured embedder (model name from settings.embedding_model)."""
    # lru_cache means every caller shares ONE Embedder. Without it, each call built a new
    # Embedder with an empty cached_property, so `dimension` hit the OpenAI API every time.
    # The model name comes from .env so it can be changed without touching code;
    # if it isn't set we fall back to a cheap, good-enough default.
    return Embedder(get_settings().embedding_model or "text-embedding-3-small")
