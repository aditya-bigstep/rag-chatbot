"""Turn stored sections into readable text with numbered citations."""

from __future__ import annotations

from typing import Any


def number_sources(report: dict[str, Any]) -> dict[str, int]:
    """Source id -> citation number, in the order sources first appear in the report."""
    numbers: dict[str, int] = {}
    for item in report.get("outline", []):
        section = report.get("sections", {}).get(item["id"], {})
        for claim in section.get("claims", []):
            for sid in claim["source_ids"]:
                if sid in report.get("sources", {}):
                    numbers.setdefault(sid, len(numbers) + 1)
    return numbers


def cited_sentence(claim: dict[str, Any], numbers: dict[str, int]) -> str:
    marks = "".join(f"[{numbers[sid]}]" for sid in claim["source_ids"] if sid in numbers)
    text = claim["text"]
    if marks and text[-1:] in ".!?":
        return f"{text[:-1]} {marks}{text[-1]}"
    return f"{text} {marks}".strip()


def section_text(section: dict[str, Any], numbers: dict[str, int]) -> str:
    paragraphs: dict[int, list[str]] = {}
    for claim in section.get("claims", []):
        paragraphs.setdefault(claim["para"], []).append(cited_sentence(claim, numbers))
    return "\n\n".join(" ".join(lines) for lines in paragraphs.values())


def report_markdown(report: dict[str, Any]) -> str:
    numbers = number_sources(report)
    lines = [f"# {report['topic']}", ""]
    for item in report.get("outline", []):
        section = report.get("sections", {}).get(item["id"])
        if not section:
            continue
        body = section_text(section, numbers) or f"_{section.get('note', 'No content.')}_"
        lines += [f"## {section['title']}", "", body, ""]
    lines += ["## References", ""]
    for sid, number in numbers.items():
        source = report["sources"][sid]
        lines.append(f"{number}. [{source['title']}]({source['url']}) ({source['domain']})")
    return "\n".join(lines) + "\n"
