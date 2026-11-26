"""LLM-as-judge helpers. The judge is a different model from the one that wrote the report."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from app.chat.llm import build_model
from app.research.llm_json import ask_json


class Flags(BaseModel):
    results: list[bool] = Field(default_factory=list)


class Rubric(BaseModel):
    clarity: int
    organisation: int
    flow: int
    comment: str = ""


class AnswerScore(BaseModel):
    faithful: bool
    relevant: bool
    refused: bool


def get_judge() -> Any:
    provider, _, model = os.getenv("EVAL_JUDGE", "groq:qwen/qwen3.8-27b").partition(":")
    return build_model(provider, model)


def judge_claims(judge: Any, pairs: list[tuple[str, str]]) -> list[bool]:
    """For each (claim, evidence): does the evidence support the claim?"""
    verdicts: list[bool] = []
    for start in range(0, len(pairs), 4):
        batch = pairs[start : start + 4]
        items = "\n\n".join(
            f"{i}. CLAIM: {claim}\nEVIDENCE: {evidence[:3600]}" for i, (claim, evidence) in enumerate(batch)
        )
        system = (
            "You are a strict fact checker. For each numbered item answer true only if the EVIDENCE states "
            "or clearly implies the CLAIM, including every number and name in it.\n"
            'Shape: {"results": [true, false]} with one entry per item, in order.'
        )
        results = ask_json(judge, system, items, Flags).results
        verdicts += (results + [False] * len(batch))[: len(batch)]
    return verdicts


def judge_coverage(judge: Any, report_text: str, aspects: list[str]) -> list[bool]:
    system = (
        "You check whether a report covers given aspects of its topic. An aspect is covered if the report "
        "gives at least one concrete statement about it.\n"
        'Shape: {"results": [true, false]} with one entry per aspect, in order.'
    )
    user = "Aspects:\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(aspects)) + f"\n\nReport:\n{report_text[:6000]}"
    results = ask_json(judge, system, user, Flags).results
    return (results + [False] * len(aspects))[: len(aspects)]


def judge_readability(judge: Any, report_text: str) -> Rubric:
    system = (
        "Grade this research report from 1 (poor) to 5 (excellent) on three things.\n"
        "clarity: sentences are easy to understand and precise.\n"
        "organisation: sections have a clear purpose and sensible order, with no repetition between them.\n"
        "flow: ideas connect; it does not read like a list of unrelated facts.\n"
        'Shape: {"clarity": 3, "organisation": 3, "flow": 3, "comment": "one sentence"}'
    )
    return ask_json(judge, system, report_text[:6000], Rubric)


def judge_answer(judge: Any, question: str, answer: str, context: str) -> AnswerScore:
    system = (
        "Grade a chatbot answer.\n"
        "faithful: every fact in the answer is found in the CONTEXT (an answer that declines is faithful).\n"
        "relevant: the answer addresses the QUESTION directly.\n"
        "refused: the answer says it cannot find or cannot help with this.\n"
        'Shape: {"faithful": true, "relevant": true, "refused": false}'
    )
    user = f"QUESTION: {question}\n\nANSWER: {answer}\n\nCONTEXT:\n{context[:5000] or '(nothing was retrieved)'}"
    return ask_json(judge, system, user, AnswerScore)
