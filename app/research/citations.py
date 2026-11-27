"""Citation bookkeeping — plain string functions, no LLM.

Inside one section the writer cites with LOCAL numbers: [1] is that section's first source.
`compile_report` turns them into GLOBAL numbers for the whole report and builds the reference list,
so the same web page has the same number everywhere and only cited pages are listed.
"""

from __future__ import annotations

import re
from typing import Any

from app.research.types import Section, Source

MARKER = re.compile(r"\[(\d+)\]")
_MARKERS_AFTER_STOP = re.compile(r"([.!?])((?:\s*\[\d+\])+)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_ABBREVIATION = re.compile(r"(?:\b(?:e\.g|i\.e|vs|etc|approx|Rs|Mr|Ms|Dr|No|Inc|Ltd|Co|St)|\b[A-Z])\.$")
_SPACE_BEFORE_STOP = re.compile(r"\s+([.,;:!?])")


def tidy_markers(text: str, source_count: int) -> str:
    """Move markers before the full stop ("x. [1]" -> "x [1].") and delete markers that point nowhere."""
    text = _MARKERS_AFTER_STOP.sub(lambda m: f" {m.group(2).strip()}{m.group(1)}", text)
    text = MARKER.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= source_count else "", text)
    return _SPACE_BEFORE_STOP.sub(r"\1", text)


def cited_numbers(text: str) -> list[int]:
    """Distinct marker numbers in `text`, in order of first appearance."""
    return list(dict.fromkeys(int(n) for n in MARKER.findall(text)))


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, without breaking after abbreviations such as "U.S." or "e.g."."""
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(paragraph.strip()):
        if sentences and _ABBREVIATION.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        elif piece:
            sentences.append(piece)
    return sentences


def split_paragraphs(text: str) -> list[list[str]]:
    """Text -> paragraphs -> sentences."""
    return [split_sentences(p) for p in re.split(r"\n\s*\n", text) if p.strip()]


def join_paragraphs(paragraphs: list[list[str]]) -> str:
    return "\n\n".join(" ".join(sentences) for sentences in paragraphs if sentences)


def compile_report(
    topic: str, order: list[str], sections: dict[str, Section], sources: dict[str, Source]
) -> dict[str, Any]:
    """Assemble the final report. OUT: {"markdown", "sections": [...], "references": [...]}."""
    numbers: dict[str, int] = {}  # source id -> global citation number

    def renumber(section: Section) -> str:
        def swap(match: re.Match[str]) -> str:
            local = int(match.group(1))
            if not 1 <= local <= len(section["source_ids"]):
                return ""
            sid = section["source_ids"][local - 1]
            numbers.setdefault(sid, len(numbers) + 1)
            return f"[{numbers[sid]}]"

        return MARKER.sub(swap, section["content"])

    rendered = []
    for position, section_id in enumerate(order, start=1):
        section = sections[section_id]
        text = renumber(section)
        rendered.append(
            {
                "id": section_id,
                "number": position,
                "title": section["title"],
                "guidance": section["guidance"],
                "text": text,
                "refs": cited_numbers(text),
                "revision": section["revision"],
                "checked": section["checked"],
                "dropped": section["dropped"],
            }
        )
    references = []
    for sid, n in numbers.items():
        source = sources[sid]
        references.append(
            {
                "n": n,
                "url": source["url"],
                "title": source["title"],
                "domain": source["domain"],
                "passages": source["passages"],
            }
        )
    body = "\n\n".join(f"## {s['number']}. {s['title']}\n\n{s['text']}" for s in rendered)
    listing = "\n".join(f"- [{r['n']}] {r['title']} — <{r['url']}>" for r in references)
    markdown = f"# {topic}\n\n{body}\n\n## References\n\n{listing}\n"
    return {"markdown": markdown, "sections": rendered, "references": references}
