"""Embeddings: text -> vectors.

EMBEDDING_MODEL picks the backend:
  fastembed:BAAI/bge-small-en-v1.5   local ONNX model, no API key
  hash                               bag-of-words hashing, no model at all (tests / offline)
  openai:text-embedding-3-small      hosted
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import cached_property, lru_cache
from typing import Protocol

from langsmith import traceable

from app.config import get_settings

DEFAULT_MODEL = "fastembed:BAAI/bge-small-en-v1.5"
_WORD = re.compile(r"[a-z0-9]+")


def _count_only(vectors: list[list[float]]) -> dict[str, int]:
    return {"vector_count": len(vectors), "dimension": len(vectors[0]) if vectors else 0}


def _dimension_only(vector: list[float]) -> dict[str, int]:
    return {"dimension": len(vector)}


class Backend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedBackend:
    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]


class HashBackend:
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]


class OpenAIBackend:
    def __init__(self, model: str) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.model = OpenAIEmbeddings(model=model, chunk_size=100)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)


def _backend(name: str) -> Backend:
    kind, _, model = name.partition(":")
    if kind == "hash":
        return HashBackend()
    if kind == "fastembed":
        return FastEmbedBackend(model or "BAAI/bge-small-en-v1.5")
    if kind == "openai":
        return OpenAIBackend(model or "text-embedding-3-small")
    return OpenAIBackend(name)  # a bare name is treated as an OpenAI model, as before


class Embedder:
    def __init__(self, model: str) -> None:
        self.name = model
        self.backend = _backend(model)

    @cached_property
    def dimension(self) -> int:
        return len(self.backend.embed(["example"])[0])

    @traceable(run_type="embedding", name="embed_documents", process_outputs=_count_only)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.backend.embed(texts) if texts else []

    @traceable(run_type="embedding", name="embed_query", process_outputs=_dimension_only)
    def embed_query(self, text: str) -> list[float]:
        return self.backend.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(get_settings().embedding_model or DEFAULT_MODEL)
