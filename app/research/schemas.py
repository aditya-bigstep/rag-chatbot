"""Shapes the LLM must return (validated with pydantic) and small helpers for report data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlannedSection(BaseModel):
    title: str
    questions: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    sections: list[PlannedSection]


class EditOp(BaseModel):
    action: Literal["expand", "rewrite", "replace_sources", "add", "remove"]
    section_id: str = ""
    title: str = ""
    instruction: str = ""
    queries: list[str] = Field(default_factory=list)
    after: str = ""


class EditPlan(BaseModel):
    ops: list[EditOp] = Field(default_factory=list)
    note: str = ""


class Verdict(BaseModel):
    id: int
    supported: bool


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


class SectionSentiment(BaseModel):
    id: str
    label: Literal["positive", "neutral", "negative"]
    score: float = 0.0
    why: str = ""


class Insight(BaseModel):
    text: str
    sections: list[str] = Field(default_factory=list)
    sources: list[int] = Field(default_factory=list)


class Analysis(BaseModel):
    overall: SectionSentiment
    sections: list[SectionSentiment] = Field(default_factory=list)
    takeaways: list[Insight] = Field(default_factory=list)
    opportunities: list[Insight] = Field(default_factory=list)
    risks: list[Insight] = Field(default_factory=list)
    next_steps: list[Insight] = Field(default_factory=list)
