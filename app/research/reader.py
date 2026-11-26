"""Fetch a web page and pull out its readable text.

Web pages are untrusted input, so the fetcher refuses anything that is not a public
http(s) address, caps the download size and only accepts HTML.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura

logger = logging.getLogger(__name__)

FetchFn = Callable[[str], dict[str, Any] | None]
MAX_BYTES = 2_000_000
MAX_CHARS = 14_000
MIN_WORDS = 120
CACHE_SECONDS = 7 * 24 * 3600
BROWSER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
SKIP_DOMAINS = ("youtube.com", "facebook.com", "instagram.com", "tiktok.com", "pinterest.com", "x.com", "twitter.com")
SKIP_SUFFIXES = (".pdf", ".zip", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".mp4")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_readable_url(url: str) -> bool:
    """Cheap check with no network: an http(s) page, not a file download or a social media site."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if parsed.path.lower().endswith(SKIP_SUFFIXES):
        return False
    host = domain_of(url)
    return not any(host == d or host.endswith("." + d) for d in SKIP_DOMAINS)


def is_safe_url(url: str) -> bool:
    """True only for public http(s) URLs. Blocks localhost and private networks (SSRF)."""
    if not is_readable_url(url):
        return False
    parsed = urlparse(url)
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except OSError:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def download(url: str) -> dict[str, Any] | None:
    if not is_safe_url(url):
        return None
    try:
        with httpx.stream(
            "GET", url, headers={"User-Agent": BROWSER_AGENT}, timeout=5, follow_redirects=True
        ) as response:
            if response.status_code != 200 or "html" not in response.headers.get("content-type", ""):
                return None
            if not is_safe_url(str(response.url)):  # a redirect must not lead somewhere private
                return None
            body = b""
            for part in response.iter_bytes():
                body += part
                if len(body) > MAX_BYTES:
                    break
            html = body.decode(response.encoding or "utf-8", errors="ignore")
    except (httpx.HTTPError, OSError):
        return None

    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    if len(text.split()) < MIN_WORDS:
        return None
    title_match = _TITLE.search(html)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    return {"url": url, "title": title[:160] or domain_of(url), "text": text[:MAX_CHARS]}


def make_fetch(cache_dir: Path | None = None) -> FetchFn:
    """`download` with a disk cache, so edits and eval reruns do not hit the same site twice."""

    def fetch(url: str) -> dict[str, Any] | None:
        if cache_dir is None:
            return download(url)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_SECONDS:
            return json.loads(cache_file.read_text()) or None
        page = download(url)
        cache_file.write_text(json.dumps(page or {}))
        return page

    return fetch


def split_passages(text: str, size: int = 600) -> list[str]:
    """Group paragraphs into passages of roughly `size` characters."""
    passages: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.split("\n")):
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) > size:
            passages.append(current)
            current = ""
        current = f"{current} {paragraph}".strip()
        while len(current) > size * 1.5:
            cut = current.rfind(". ", 0, size) + 1 or size
            passages.append(current[:cut].strip())
            current = current[cut:].strip()
    if current:
        passages.append(current)
    return [p for p in passages if len(p.split()) >= 12]
