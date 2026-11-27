"""Citation helpers: marker clean-up, sentence splitting and report compilation."""

from __future__ import annotations

from app.research.citations import compile_report, split_sentences, tidy_markers
from app.research.reader import domain_of, pick_passages, source_id, split_passages
from app.research.types import Source, new_section
from tests.fakes import HashEmbedder


def test_tidy_markers_moves_them_before_the_stop_and_drops_invalid_ones() -> None:
    assert tidy_markers("Sales grew. [1] Costs fell.[2][9]", 2) == "Sales grew [1]. Costs fell [2]."


def test_split_sentences_keeps_abbreviations_and_decimals_together() -> None:
    text = "The U.S. market grew 2.5 times [1]. Prices fell, e.g. batteries [2]. Done [3]."
    assert split_sentences(text) == [
        "The U.S. market grew 2.5 times [1].",
        "Prices fell, e.g. batteries [2].",
        "Done [3].",
    ]


def _source(url: str) -> Source:
    return Source(id=source_id(url), url=url, title=url, domain=domain_of(url), passages=["text"])


def test_compile_gives_one_global_number_per_page_and_lists_only_cited_pages() -> None:
    a, b, c = _source("https://a.example/1"), _source("https://b.example/2"), _source("https://c.example/3")
    first = new_section("s1", "One")
    first.update(content="Fact [2]. Other [1].", source_ids=[a["id"], b["id"], c["id"]], revision=1)
    second = new_section("s2", "Two")
    second.update(content="Again [1]. Bogus [7].", source_ids=[b["id"]], revision=1)
    sources = {s["id"]: s for s in (a, b, c)}
    report = compile_report("Topic", ["s1", "s2"], {"s1": first, "s2": second}, sources)
    assert report["sections"][0]["text"] == "Fact [1]. Other [2]."
    assert report["sections"][1]["text"] == "Again [1]. Bogus ."  # same page -> same number; bad marker removed
    assert [r["url"] for r in report["references"]] == ["https://b.example/2", "https://a.example/1"]


def test_pick_passages_prefers_relevant_text_and_keeps_page_order() -> None:
    passages = ["cooking pasta recipes", "battery prices fell sharply", "football results", "battery supply chain"]
    assert pick_passages("battery prices", passages, HashEmbedder(), 2) == [passages[1], passages[3]]


def test_split_passages_respects_size() -> None:
    passages = split_passages("\n".join(["word " * 30] * 10), size=400)
    assert len(passages) > 1 and all(len(p) <= 800 for p in passages)
