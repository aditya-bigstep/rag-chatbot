"""Everything the research graphs need from outside, bundled so tests can pass fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.research.reader import FetchFn, make_fetch
from app.research.search import SearchFn, make_search


@dataclass
class Deps:
    llm: Any
    fast_llm: Any
    search: SearchFn
    fetch: FetchFn
    embedder: Any
    settings: Settings


def build_deps(settings: Settings, embedder: Any) -> Deps:
    from app.chat.llm import get_llm

    cache = settings.research_dir / "cache"
    return Deps(
        llm=get_llm(),
        fast_llm=get_llm(fast=True),
        search=make_search(settings.search_provider, cache / "search"),
        fetch=make_fetch(cache / "pages"),
        embedder=embedder,
        settings=settings,
    )
