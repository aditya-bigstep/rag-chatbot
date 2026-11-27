"""Sample topics on the new-report screen.

Each visit shows three topics from three different areas. The ones shown last time are
remembered in a cookie and skipped, so the suggestions change on every visit.
"""

from __future__ import annotations

import random

COOKIE = "pilot_starters"

TOPICS: dict[str, list[str]] = {
    "energy": [
        "State of electric two-wheelers in India",
        "Solid-state batteries: where the technology stands",
        "Green hydrogen: cost and scale in 2026",
        "Rooftop solar adoption in Indian cities",
        "Small modular nuclear reactors: progress and doubts",
    ],
    "technology": [
        "Open-weight AI models versus closed models for businesses",
        "How quantum computing is used today",
        "India's semiconductor manufacturing push",
        "UPI and the growth of real-time payments worldwide",
        "Satellite internet in rural areas: costs and coverage",
    ],
    "health": [
        "GLP-1 weight-loss drugs: benefits, risks and supply",
        "Antimicrobial resistance: how serious is it now",
        "Air pollution and health in North Indian cities",
        "Telemedicine after the pandemic: what stuck",
        "Sleep and its effect on memory: what research shows",
    ],
    "work and society": [
        "Remote work and productivity: what studies show",
        "The four-day work week: results from trials",
        "Gig workers and social security in India",
        "Screen time and teenagers' mental health",
        "Urban heat and how cities are adapting",
    ],
    "economy and business": [
        "Quick commerce in India: growth and profitability",
        "Why central banks are buying gold",
        "Electric vehicle price wars in China",
        "The economics of India's startup IPOs",
        "Global shipping routes after Red Sea disruptions",
    ],
    "science and environment": [
        "Microplastics in food and drinking water",
        "Lab-grown meat: cost, safety and regulation",
        "Groundwater depletion in India",
        "Coral reef bleaching: causes and recovery",
        "Gene editing in agriculture: where it is allowed",
    ],
}

ALL = [topic for topics in TOPICS.values() for topic in topics]


def pick(seen: str | None, taken: set[str] | frozenset[str] = frozenset(), k: int = 3) -> list[str]:
    """Three topics from three different areas, skipping the last set shown and topics already reported on."""
    last = {int(i) for i in (seen or "").split(".") if i.isdigit()}
    skip = {topic.casefold() for topic in taken}
    areas = random.sample(list(TOPICS), k)
    chosen = []
    for area in areas:
        fresh = [t for t in TOPICS[area] if ALL.index(t) not in last and t.casefold() not in skip]
        chosen.append(random.choice(fresh or TOPICS[area]))
    return chosen


def remember(topics: list[str]) -> str:
    """The cookie value for the topics just shown."""
    return ".".join(str(ALL.index(t)) for t in topics)
