"""How much we trust a source and a section. Shown in the UI as the trust score."""

from __future__ import annotations

from typing import Any


def score_source(domain: str, words: int) -> float:
    """Trust in one source, from 0 (ignore it) to 1 (very reliable)."""
    # TODO(human): score the source from its domain and length.
    return 0.5


def section_trust(claims: list[dict[str, Any]], sources: list[dict[str, Any]]) -> float:
    """Blend of: share of supported claims, average source trust, and how many different sites back it."""
    if not claims or not sources:
        return 0.0
    supported = sum(c["status"] == "supported" for c in claims) / len(claims)
    quality = sum(s["trust"] for s in sources) / len(sources)
    variety = min(1.0, len({s["domain"] for s in sources}) / 3)
    return round(0.6 * supported + 0.25 * quality + 0.15 * variety, 2)
