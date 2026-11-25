"""MILESTONE 3 — Chunking.

Split a Document into Chunks small enough to embed and retrieve.

You choose the METHOD (fixed-size with overlap, recursive by paragraph/sentence,
markdown-heading aware, semantic ...). The method is picked by name from
`CHUNKING_STRATEGY` in `.env`, so you can add several and switch between them.

Whatever method you use, every chunker must obey these rules (the evaluator
checks them on documents you have never seen):

  1. An empty / whitespace-only document gives `[]`.
  2. No chunk is empty or whitespace-only.
  3. `len(chunk.text) <= chunk_size` (characters) for every chunk.
  4. No text is lost: every word of the document appears in at least one chunk.
  5. `chunk.doc_id == document.id` and `chunk.source == document.source`.
  6. `position` is 0, 1, 2 ... in document order; `id` is unique (e.g. f"{doc_id}:{position}").
  7. Same input -> same output (no randomness).
  8. A smaller `chunk_size` gives more chunks for the same long document.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from app.config import get_settings
from app.models import Chunk, Document


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        """IN: one Document.  OUT: its Chunks, in order."""
        ...


### protocol meaning -> Any object that acts as a chunker must have a chunk() method that accepts a Document and
### returns a list of Chunk objects.

class LangChainChunker:
    """Wraps any `split_text(text) -> list[str]` function and builds our Chunk objects."""

    def __init__(self, split_text: Callable[[str], list[str]]) -> None:
        # Just remember whichever splitting function we were handed (fixed, recursive, semantic...).
        self.split_text = split_text

    def chunk(self, document: Document) -> list[Chunk]:
        # Cut the document's text into pieces, trim extra spaces, and throw away any blank pieces.
        raw_chunks = self.split_text(document.text)

        texts = []
        for chunk in raw_chunks:
            cleaned = chunk.strip()
            if cleaned:
                texts.append(cleaned)

        # Turn each piece of text into a proper Chunk, giving it a unique id (like "doc1:0", "doc1:1"),
        # a link back to its document and source, and its order number in the document.
        return [
            Chunk(id=f"{document.id}:{i}", doc_id=document.id, text=t, source=document.source, position=i)
            for i, t in enumerate(texts)
        ]


def _semantic_split(size: int, overlap: int) -> Callable[[str], list[str]]:
    """Split where the meaning changes, then re-split anything longer than `size`."""
    # Imported here (not at the top) so these heavy libraries only load if you actually pick "semantic".
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    # Pick the embedding model from settings, or fall back to a sensible default.
    model = get_settings().embedding_model or "text-embedding-3-small"
    # First splitter: reads the text and cuts it where the topic/meaning shifts.
    semantic = SemanticChunker(OpenAIEmbeddings(model=model))
    # Second splitter: a safety net that cuts any piece that is still too long.
    limit = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    def split(text: str) -> list[str]:
        # Split by meaning first, then make sure every resulting piece fits within the size limit.
        return [part for piece in semantic.split_text(text) for part in limit.split_text(piece)]

    return split


def get_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chunker:
    """Return the chunker for `strategy` (default: settings.chunking_strategy).

    IN:  strategy name, max chunk size in characters, overlap in characters.
         Any argument left as None falls back to `app.config.get_settings()`.
    OUT: an object with a `.chunk(document) -> list[Chunk]` method.

    """
    settings = get_settings()
    # Use the strategy passed in; if none, use the one from settings. Clean it up (trim, lowercase) so "Fixed " still works.
    name = (settings.chunking_strategy if strategy is None else strategy).strip().lower()
    # Same idea for size and overlap: use what the caller gave, otherwise the defaults from settings.
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    # A chunk can't have zero or negative length, so stop early with a clear error.
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    # Keep overlap sensible: never negative, and always smaller than the chunk itself.
    overlap = max(0, min(overlap, size - 1))

    # "fixed": cut the text into equal-sized pieces, breaking at spaces.
    if name == "fixed":
        return LangChainChunker(CharacterTextSplitter(separator=" ", chunk_size=size, chunk_overlap=overlap).split_text)
    # "recursive": try to break at paragraphs first, then sentences, then words, so pieces stay natural.
    if name == "recursive":
        return LangChainChunker(RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text)
    # "semantic": cut where the topic changes (uses embeddings, so needs an OpenAI key).
    if name == "semantic":
        return LangChainChunker(_semantic_split(size, overlap))
    # Anything else is a typo or unsupported name, so tell the user what the valid options are.
    raise ValueError(f"Unknown chunking strategy {name!r}; use 'fixed', 'recursive' or 'semantic'")
