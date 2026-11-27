"""The shared state of the research graphs.

Everything is a plain dict (TypedDict) on purpose: LangGraph saves the state to SQLite after every step,
and plain dicts/lists/strings serialise without any extra setup.

KEY IDEA — sections are stored one by one, each with its own sources:

    sections = {"s1": {...}, "s2": {...}}      sources = {"src_ab12": {...}, ...}

so a change request ("make section 2 more detailed") re-runs ONLY section s2 and leaves the rest untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict


class Source(TypedDict):
    """One web page we actually fetched and read. Citations can only point at these."""

    id: str  # "src_" + hash of the url (stable, so the same page is never stored twice)
    url: str
    title: str
    domain: str
    passages: list[str]  # the parts of the page that were shown to the writer


class Section(TypedDict):
    id: str  # "s1", "s2" ... never reused
    title: str
    guidance: str  # subtopics / what this section should cover
    queries: list[str]  # web searches for this section
    content: str  # prose with LOCAL citation markers: [1] = source_ids[0], [2] = source_ids[1] ...
    source_ids: list[str]
    revision: int  # 0 = not written yet; +1 every time the section is (re)written
    checked: int  # sentences examined by the verify step
    dropped: int  # sentences removed by the verify step (unsupported or uncited)


def merge_dicts(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Reducer: parallel section workers each return {section_id: section}; merge them together."""
    return {**old, **new}


def add_counts(old: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """Reducer: sum counters such as search_calls / pages_fetched / llm_calls."""
    total = dict(old)
    for key, value in new.items():
        total[key] = total.get(key, 0) + value
    return total


def merge_sources(old: dict[str, Source], new: dict[str, Source]) -> dict[str, Source]:
    """Reducer for `sources`. Two sections may read the SAME page but keep different passages of it,
    so when a page is already known we add the new passages instead of overwriting the old ones."""
    merged = dict(old)
    for key, source in new.items():
        if key in merged:
            passages = list(dict.fromkeys([*merged[key]["passages"], *source["passages"]]))
            source = Source(**{**source, "passages": passages})  # type: ignore[typeddict-item]
        merged[key] = source
    return merged


class ResearchState(TypedDict, total=False):
    """State of the main graph (app/research/graph.py)."""

    topic: str
    order: list[str]  # section ids in report order (the outline)
    sections: Annotated[dict[str, Section], merge_dicts]
    sources: Annotated[dict[str, Source], merge_sources]
    todo: list[str]  # section ids to (re)write in the current pass
    mode: str  # "new" | "expand" | "replace" | "rewrite" — what the current pass does
    instruction: str  # the human's change request, passed to the section writer
    notice: str  # message for the UI when a change request could not be applied
    done: bool  # the human finished the review
    stats: Annotated[dict[str, int], add_counts]


class SectionState(TypedDict, total=False):
    """State of the section sub-graph (app/research/section_graph.py). One copy per section, run in parallel."""

    topic: str
    section: Section
    mode: str
    instruction: str
    known: list[Source]  # sources this section already had (kept for "expand" / "rewrite")
    hits: list[dict[str, str]]  # search results: {"url", "title", "snippet"}
    material: list[Source]  # sources the writer may cite, in citation order
    # --- results, merged back into the main graph through the reducers above ---
    sections: Annotated[dict[str, Section], merge_dicts]
    sources: Annotated[dict[str, Source], merge_sources]
    stats: Annotated[dict[str, int], add_counts]


class SectionOutput(TypedDict, total=False):
    """Only these keys flow from a section worker back to the main graph."""

    sections: Annotated[dict[str, Section], merge_dicts]
    sources: Annotated[dict[str, Source], merge_sources]
    stats: Annotated[dict[str, int], add_counts]


@dataclass(frozen=True)
class ResearchConfig:
    """Knobs of one system version. The evaluation compares two of these (see evals/versions.py)."""

    max_sections: int = 5
    results_per_query: int = 5
    max_sources: int = 6  # sources given to the writer per section
    passages_per_source: int = 2
    verify: bool = True  # run the claim-verification step


def new_section(section_id: str, title: str, guidance: str = "", queries: list[str] | None = None) -> Section:
    return Section(
        id=section_id,
        title=title.strip(),
        guidance=guidance.strip(),
        queries=queries or [],
        content="",
        source_ids=[],
        revision=0,
        checked=0,
        dropped=0,
    )
