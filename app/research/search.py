"""Web search behind one function, so the provider can be swapped in `.env` (SEARCH_PROVIDER)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[dict[str, str]]]
CACHE_SECONDS = 24 * 3600
USER_AGENT = "ResearchPilot/0.1 (training project)"
_one_at_a_time = threading.Lock()


def ddgs_search(query: str, limit: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    for attempt in range(2):
        try:
            rows = DDGS(timeout=10).text(query, max_results=limit, backend="auto")
        except Exception:
            rows = []
        if rows:
            return [{"url": r["href"], "title": r.get("title", ""), "snippet": r.get("body", "")} for r in rows]
        time.sleep(1.5 * (attempt + 1))
    return []


def tavily_search(query: str, limit: int) -> list[dict[str, str]]:
    response = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": "Bearer " + os.environ["TAVILY_API_KEY"]},
        json={"query": query, "max_results": limit, "search_depth": "fast"},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json().get("results", [])
    return [{"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", "")} for r in rows]


def firecrawl_search(query: str, limit: int) -> list[dict[str, str]]:
    response = httpx.post(
        "https://api.firecrawl.dev/v1/search",
        headers={"Authorization": "Bearer " + os.environ["FIRECRAWL_API_KEY"]},
        json={"query": query, "limit": limit},
        timeout=20,
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    return [{"url": r["url"], "title": r.get("title", ""), "snippet": r.get("description", "")} for r in rows]


def wikipedia_search(query: str, limit: int) -> list[dict[str, str]]:
    response = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()["query"]["search"]
    return [
        {
            "url": "https://en.wikipedia.org/wiki/" + r["title"].replace(" ", "_"),
            "title": r["title"],
            "snippet": r.get("snippet", ""),
        }
        for r in rows
    ]


PROVIDERS: dict[str, SearchFn] = {
    "ddgs": ddgs_search,
    "tavily": tavily_search,
    "firecrawl": firecrawl_search,
    "wikipedia": wikipedia_search,
}
KEY_NAMES = {"tavily": "TAVILY_API_KEY", "firecrawl": "FIRECRAWL_API_KEY"}
FREE_ENDPOINTS = ("ddgs", "wikipedia")  # these block bursts, so they are called one at a time


def call_provider(name: str, query: str, limit: int) -> list[dict[str, str]]:
    if name in FREE_ENDPOINTS:
        with _one_at_a_time:
            return PROVIDERS[name](query, limit)
    return PROVIDERS[name](query, limit)


def make_search(provider: str, cache_dir: Path | None = None) -> SearchFn:
    """Search with `provider`, fall back to the other providers if it fails, and cache results for a day."""
    order = [provider] if provider in PROVIDERS else ["ddgs"]
    for backup in ("tavily", "firecrawl", "ddgs", "wikipedia"):
        needs_key = KEY_NAMES.get(backup)
        if backup not in order and (not needs_key or os.getenv(needs_key)):
            order.append(backup)

    def search(query: str, limit: int) -> list[dict[str, str]]:
        cache_file = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(f"{order[0]}|{limit}|{query}".encode()).hexdigest()[:24]
            cache_file = cache_dir / f"{key}.json"
            if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_SECONDS:
                return json.loads(cache_file.read_text())
        for name in order:
            try:
                rows = call_provider(name, query, limit)
            except Exception as exc:
                logger.warning("Search provider %s failed for %r: %s", name, query, str(exc)[:120])
                continue
            if rows:
                if cache_file is not None:
                    cache_file.write_text(json.dumps(rows))
                return rows
        return []

    return search
