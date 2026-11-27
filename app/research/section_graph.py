"""The section worker: a small sub-graph that researches and writes ONE section.

    START -> search -> read -> write -> verify -> finish -> END
                                          (verify is skipped when ResearchConfig.verify is False)

The main graph starts one copy per section that needs work (in parallel, with `Send`). Because a worker
only knows its own section, re-running it can never disturb the other sections.

`mode` says what kind of pass this is:
  new      first draft                      expand   keep the old sources, search for more, write longer
  replace  drop the old sources, find others rewrite  no web search, reword using the sources we have
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langsmith.utils import ContextThreadPoolExecutor

from app.observability import traced_node
from app.research import prompts
from app.research.citations import MARKER, join_paragraphs, split_paragraphs, tidy_markers
from app.research.llm_json import ask_json, ask_text
from app.research.reader import PageReader, domain_of, pick_passages, source_id, split_passages
from app.research.search import SearchProvider
from app.research.types import ResearchConfig, SectionOutput, SectionState, Source

NO_SOURCES = "No reliable sources could be found for this section, so nothing is claimed here."
MAX_PER_DOMAIN = 2  # source diversity: do not let one website dominate a section


class SectionWorker:
    def __init__(
        self, llm: BaseChatModel, search: SearchProvider, reader: PageReader, embedder: Any, config: ResearchConfig
    ) -> None:
        self.llm = llm
        self.searcher = search
        self.reader = reader
        self.embedder = embedder
        self.config = config

    # ------------------------------------------------------------------ nodes
    def search(self, state: SectionState) -> dict[str, Any]:
        """Search the web for this section. Skips pages we already have (and, for `replace`, their sites)."""
        section, mode = state["section"], state["mode"]
        if mode == "rewrite":
            return {"hits": []}
        focus = f"{state['topic']} {section['title']}"
        queries = section["queries"] or [focus, f"{state['topic']} {section['guidance']}"]
        if mode in ("expand", "replace") and state["instruction"]:
            queries = [f"{focus} {state['instruction']}", *queries[:1]]
        seen = {s["url"] for s in state["known"]}
        banned = {s["domain"] for s in state["known"]} if mode == "replace" else set()
        hits: list[dict[str, str]] = []
        calls = 0
        for query in [*queries, focus]:
            if query == focus and hits:  # the plain "topic + title" search is only a last resort
                break
            calls += 1
            for hit in self.searcher.search(query, self.config.results_per_query):
                if hit["url"] not in seen and domain_of(hit["url"]) not in banned:
                    seen.add(hit["url"])
                    hits.append(hit)
        return {"hits": hits, "stats": {"search_calls": calls}}

    def read(self, state: SectionState) -> dict[str, Any]:
        """Fetch the pages (in parallel), keep the readable ones and their most relevant passages."""
        section, mode = state["section"], state["mode"]
        material: list[Source] = list(state["known"]) if mode in ("expand", "rewrite") else []
        limit = self.config.max_sources + (3 if mode == "expand" else 0)
        hits = state["hits"]
        # Context-aware pool: the page fetches stay nested under this node in the LangSmith trace.
        with ContextThreadPoolExecutor(max_workers=6) as pool:
            texts = list(pool.map(self.reader.read, [h["url"] for h in hits]))
        query = f"{section['title']}. {section['guidance']} {state['instruction']}"
        per_domain: dict[str, int] = {}
        for hit, text in zip(hits, texts, strict=True):
            domain = domain_of(hit["url"])
            if not text or len(material) >= limit or per_domain.get(domain, 0) >= MAX_PER_DOMAIN:
                continue
            per_domain[domain] = per_domain.get(domain, 0) + 1
            passages = pick_passages(query, split_passages(text), self.embedder, self.config.passages_per_source)
            material.append(
                Source(id=source_id(hit["url"]), url=hit["url"], title=hit["title"], domain=domain, passages=passages)
            )
        failed = sum(1 for text in texts if not text)
        return {"material": material, "stats": {"pages_fetched": len(hits) - failed, "pages_failed": failed}}

    def write(self, state: SectionState) -> dict[str, Any]:
        """Draft the section from the numbered sources only. Source ids travel with the text from the start."""
        section, material = state["section"].copy(), state["material"]
        section["revision"] += 1
        section["checked"] = section["dropped"] = 0
        section["source_ids"] = [s["id"] for s in material]
        if not material:
            section["content"] = NO_SOURCES
            return {"section": section}
        blocks = [
            f'<source n="{n}" title="{s["title"]}">\n' + "\n...\n".join(s["passages"]) + "\n</source>"
            for n, s in enumerate(material, start=1)
        ]
        request = f"Reviewer's change request: {state['instruction']}\n" if state["instruction"] else ""
        previous = ""
        if state["mode"] in ("expand", "rewrite") and state["section"]["content"]:
            previous = f"Current version of the section (same source numbers):\n{state['section']['content']}\n\n"
        user = (
            f"Report topic: {state['topic']}\nSection title: {section['title']}\n"
            f"Must cover: {section['guidance']}\n{request}\n{previous}Sources:\n" + "\n\n".join(blocks)
        )
        system = prompts.WRITE.format(paragraphs="4-5" if state["mode"] == "expand" else "2-3")
        section["content"] = tidy_markers(ask_text(self.llm, system, user).strip(), len(material))
        return {"section": section, "stats": {"llm_calls": 1}}

    def verify(self, state: SectionState) -> dict[str, Any]:
        """Check every sentence against the sources it cites; remove what is uncited or unsupported."""
        section, material = state["section"].copy(), state["material"]
        if not material:
            return {}
        paragraphs = split_paragraphs(section["content"])
        sentences = [s for paragraph in paragraphs for s in paragraph]
        cited = {i for i, s in enumerate(sentences, start=1) if MARKER.search(s)}
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1) if i in cited)
        blocks = "\n\n".join(f"[{n}] {' ... '.join(s['passages'])}" for n, s in enumerate(material, start=1))
        reply = ask_json(self.llm, prompts.VERIFY, f"Sources:\n{blocks}\n\nSentences:\n{listing}")
        rejected = {v.get("i") for v in reply.get("verdicts", []) if isinstance(v, dict) and not v.get("supported")}
        keep = cited - rejected
        counter = iter(range(1, len(sentences) + 1))
        kept = [[s for s in paragraph if next(counter) in keep] for paragraph in paragraphs]
        section["content"] = join_paragraphs(kept) or NO_SOURCES
        section["checked"] = len(sentences)
        section["dropped"] = len(sentences) - len(keep)
        return {"section": section, "stats": {"llm_calls": 1, "claims_dropped": len(sentences) - len(keep)}}

    def finish(self, state: SectionState) -> dict[str, Any]:
        """Hand the finished section and its sources back to the main graph."""
        section = state["section"]
        return {"sections": {section["id"]: section}, "sources": {s["id"]: s for s in state["material"]}}

    # ------------------------------------------------------------------ assembly
    def build(self) -> Any:
        graph = StateGraph(SectionState, output_schema=SectionOutput)
        graph.add_node("search", traced_node("section_search")(self.search))
        graph.add_node("read", traced_node("section_read")(self.read))
        graph.add_node("write", traced_node("section_write")(self.write))
        graph.add_node("verify", traced_node("section_verify")(self.verify))
        graph.add_node("finish", self.finish)
        graph.add_edge(START, "search")
        graph.add_edge("search", "read")
        graph.add_edge("read", "write")
        if self.config.verify:
            graph.add_edge("write", "verify")
            graph.add_edge("verify", "finish")
        else:
            graph.add_edge("write", "finish")
        graph.add_edge("finish", END)
        return graph.compile()
