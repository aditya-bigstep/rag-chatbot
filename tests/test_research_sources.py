"""The `sources` reducer: one web page used by two sections keeps the passages of both."""

from __future__ import annotations

from app.research.types import Source, merge_sources


def _source(passages: list[str]) -> Source:
    return Source(id="src_1", url="https://a.example", title="A", domain="a.example", passages=passages)


def test_same_page_from_two_sections_keeps_all_passages() -> None:
    merged = merge_sources({"src_1": _source(["one", "two"])}, {"src_1": _source(["two", "three"])})
    assert merged["src_1"]["passages"] == ["one", "two", "three"]


def test_new_page_is_added() -> None:
    assert list(merge_sources({}, {"src_1": _source(["one"])})) == ["src_1"]
