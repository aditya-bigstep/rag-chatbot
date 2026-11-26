"""The research graph.

    guard -> plan -> review_outline (human) -> section x N (parallel) -> assemble -> feedback (human)
                                                    ^                                     |
                                                    +------------ plan_edit <-------------+

`review_outline` and `feedback` call interrupt(): the graph stops, the checkpointer saves the state,
and the run continues when the user answers. After feedback only the sections named in `todo` run again.
"""

from __future__ import annotations

import uuid
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from app.chat.guardrails import check_input
from app.chat.pii import redact_pii
from app.models import utc_now
from app.observability import traced_node
from app.research.deps import Deps
from app.research.events import emit
from app.research.llm_json import ask_json
from app.research.schemas import EditOp, EditPlan, Plan
from app.research.section import SectionWorker

PLAN_PROMPT = (
    "You plan research reports. Break the topic into {n} sections that together cover it well "
    "(background, current state, key players or drivers, challenges, outlook, as fits the topic). "
    "For each section give a short title, 2 questions it should answer and 2 web search queries.\n"
    'Shape: {{"sections": [{{"title": "...", "questions": ["..."], "queries": ["..."]}}]}}'
)
EDIT_PROMPT = (
    "A reader wants changes to a research report. Turn the request into operations on sections.\n"
    "Actions: expand (more detail), rewrite (change focus or tone), replace_sources (find better sources), "
    "add (new section: give title, 2 web search queries that mention the report topic, and `after` = id of the "
    "section it follows), remove.\n"
    "Touch only the sections the request is about. Use the section ids exactly as listed.\n"
    'Shape: {"ops": [{"action": "expand", "section_id": "...", "title": "", "instruction": "...", '
    '"queries": [], "after": ""}], "note": "one line on what you will do"}'
)
QUICK_ACTIONS = ("expand", "rewrite", "replace_sources", "remove")


def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class ResearchState(TypedDict, total=False):
    report_id: str
    topic: str
    blocked: str
    outline: list[dict[str, Any]]
    sections: Annotated[dict[str, dict[str, Any]], merge]
    sources: Annotated[dict[str, dict[str, Any]], merge]
    todo: list[str]
    edits: dict[str, dict[str, str]]
    feedback: dict[str, Any]
    note: str
    history: Annotated[list[dict[str, Any]], add]


def new_id() -> str:
    return "sec" + uuid.uuid4().hex[:6]


def clean_outline(items: list[dict[str, Any]], topic: str, limit: int) -> list[dict[str, Any]]:
    """Whatever the user sent back (edited titles, new rows, removed rows) becomes a valid outline."""
    outline = []
    for item in items[:limit]:
        title = str(item.get("title", "")).strip()[:120]
        if not title:
            continue
        outline.append(
            {
                "id": item.get("id") or new_id(),
                "title": title,
                "questions": [str(q)[:200] for q in item.get("questions", [])][:3],
                "queries": [str(q)[:150] for q in item.get("queries", [])][:3] or [f"{topic} {title}"],
            }
        )
    return outline


class ResearchGraph:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.section_graph = SectionWorker(deps).build()

    def guard(self, state: ResearchState) -> dict[str, Any]:
        topic = redact_pii(state["topic"]).text.strip()
        verdict = check_input(topic)
        if verdict.allowed and len(topic) < 4:
            return {"blocked": "Please give a longer topic."}
        return {"topic": topic, "blocked": "" if verdict.allowed else verdict.reply}

    @staticmethod
    def after_guard(state: ResearchState) -> str:
        return "end" if state.get("blocked") else "plan"

    def plan(self, state: ResearchState) -> dict[str, Any]:
        emit(state["report_id"], "plan", "Planning the outline")
        limit = self.deps.settings.research_max_sections
        plan = ask_json(self.deps.llm, PLAN_PROMPT.format(n=f"{max(3, limit - 1)} to {limit}"), state["topic"], Plan)
        outline = clean_outline([s.model_dump() for s in plan.sections], state["topic"], limit)
        return {"outline": outline}

    def review_outline(self, state: ResearchState) -> dict[str, Any]:
        answer = interrupt({"kind": "outline", "outline": state["outline"]})
        edited = answer.get("outline") if isinstance(answer, dict) else None
        limit = self.deps.settings.research_max_sections + 3
        outline = clean_outline(edited or state["outline"], state["topic"], limit)
        emit(state["report_id"], "plan", f"Outline approved with {len(outline)} sections")
        return {"outline": outline, "todo": [item["id"] for item in outline], "edits": {}, "feedback": {}}

    def dispatch(self, state: ResearchState) -> Any:
        """Fan out: one section worker per id in `todo`. Everything else is left alone."""
        jobs = []
        for item in state["outline"]:
            if item["id"] not in state.get("todo", []):
                continue
            edit = state.get("edits", {}).get(item["id"], {})
            previous = state.get("sections", {}).get(item["id"])
            known = {sid: state["sources"][sid] for sid in (previous or {}).get("source_ids", [])}
            jobs.append(
                Send(
                    "section",
                    {
                        "report_id": state["report_id"],
                        "topic": state["topic"],
                        "item": item,
                        "mode": edit.get("mode", "new"),
                        "instruction": edit.get("instruction", ""),
                        "previous": previous,
                        "sources": known,
                    },
                )
            )
        return jobs or "assemble"

    def section(self, job: dict[str, Any]) -> dict[str, Any]:
        result = self.section_graph.invoke(job)
        section = result["section"]
        return {"sections": {section["id"]: section}, "sources": result["sources"]}

    def assemble(self, state: ResearchState) -> dict[str, Any]:
        changed = state.get("todo", [])
        sections = state.get("sections", {})
        feedback = state.get("feedback") or {}
        entry = {
            "at": utc_now(),
            "kind": "revision" if feedback else "initial",
            "request": feedback.get("text", ""),
            "note": state.get("note", ""),
            "changed": [
                {
                    "id": sid,
                    "title": sections[sid]["title"],
                    "version": sections[sid]["version"],
                    "before": sections[sid].get("previous_text", ""),
                    "after": sections[sid]["text"],
                }
                for sid in changed
                if sid in sections
            ],
        }
        emit(state["report_id"], "assemble", "Report is ready")
        return {"todo": [], "edits": {}, "history": [entry]}

    def feedback(self, state: ResearchState) -> dict[str, Any]:
        answer = interrupt({"kind": "feedback"})
        return {"feedback": answer if isinstance(answer, dict) else {"text": str(answer)}, "note": ""}

    def plan_edit(self, state: ResearchState) -> dict[str, Any]:
        feedback = state["feedback"]
        text = redact_pii(str(feedback.get("text", ""))).text.strip()
        ids = {item["id"] for item in state["outline"]}

        if feedback.get("action") in QUICK_ACTIONS and feedback.get("section_id") in ids:
            # A button on a section already says what to do and where: no LLM call needed.
            op = EditOp(action=feedback["action"], section_id=feedback["section_id"], instruction=text)
            plan = EditPlan(ops=[op])
        else:
            verdict = check_input(text)
            if not verdict.allowed:
                return {"todo": [], "note": verdict.reply}
            emit(state["report_id"], "plan", "Working out which sections to change")
            listing = "\n".join(f"{item['id']} | {n}. {item['title']}" for n, item in enumerate(state["outline"], 1))
            request = f"Report topic: {state['topic']}\nSections:\n{listing}\n\nRequest: {text}"
            plan = ask_json(self.deps.fast_llm, EDIT_PROMPT, request, EditPlan)

        outline, todo, edits = self.apply_ops(plan, state, text)
        changed = bool(todo) or len(outline) != len(state["outline"])
        note = plan.note or ("" if changed else "No matching section was found.")
        return {"outline": outline, "todo": todo, "edits": edits, "note": note}

    def apply_ops(
        self, plan: EditPlan, state: ResearchState, text: str
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, str]]]:
        """Turn the edit plan into: the new outline, the section ids to redo, and how to redo each."""
        outline = list(state["outline"])
        ids = {item["id"] for item in outline}
        limit = self.deps.settings.research_max_sections + 3
        todo: list[str] = []
        edits: dict[str, dict[str, str]] = {}
        for op in plan.ops:
            if op.action == "add":
                draft = {"title": op.title.strip() or text[:80], "queries": op.queries, "questions": [op.instruction]}
                item = clean_outline([draft], state["topic"], 1)
                if not item or len(outline) >= limit:
                    continue
                position = next((i + 1 for i, o in enumerate(outline) if o["id"] == op.after), len(outline))
                outline.insert(position, item[0])
                todo.append(item[0]["id"])
            elif op.section_id in ids and op.action == "remove":
                outline = [item for item in outline if item["id"] != op.section_id]
            elif op.section_id in ids:
                todo.append(op.section_id)
                edits[op.section_id] = {"mode": op.action, "instruction": op.instruction or text}
        return outline, todo, edits

    def build(self, checkpointer: Any = None) -> Any:
        graph = StateGraph(ResearchState)
        graph.add_node("guard", traced_node("research_guard")(self.guard))
        graph.add_node("plan", traced_node("research_plan")(self.plan))
        graph.add_node("review_outline", self.review_outline)
        graph.add_node("section", self.section)  # type: ignore[arg-type]
        graph.add_node("assemble", traced_node("research_assemble")(self.assemble))
        graph.add_node("feedback", self.feedback)
        graph.add_node("plan_edit", traced_node("research_plan_edit")(self.plan_edit))

        graph.add_edge(START, "guard")
        graph.add_conditional_edges("guard", self.after_guard, {"end": END, "plan": "plan"})
        graph.add_edge("plan", "review_outline")
        graph.add_conditional_edges("review_outline", self.dispatch, ["section", "assemble"])
        graph.add_edge("section", "assemble")
        graph.add_edge("assemble", "feedback")
        graph.add_edge("feedback", "plan_edit")
        graph.add_conditional_edges("plan_edit", self.dispatch, ["section", "assemble"])
        return graph.compile(checkpointer=checkpointer)
