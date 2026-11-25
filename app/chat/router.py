"""FEATURE 2 — Intelligent routing.

`classify(message, has_history)` looks at the user's message and picks a Route. The engine
(app/chat/engine.py) then uses LangGraph *conditional edges* to send the message down the matching path:

    Route      Example                               Path in the graph
    ---------  ------------------------------------  ----------------------------------------
    DIRECT     "Hi", "thanks", "who are you?"        generate (plain LLM answer, no retrieval)
    HISTORY    "What did I ask earlier?"             generate (answers from the conversation)
    TOOL       "what is 12*7", "delete leave.md"     generate (+ tools), no retrieval
    FOLLOWUP   "Explain the second point again"      rewrite_query -> retrieve -> generate
    COMPARE    "Compare leave and expense policy"    rewrite_query -> retrieve (bigger k) -> generate
    DOCUMENT   anything else (the safe default)      retrieve -> generate

The classifier is plain rules (regex), NOT an LLM call: it is free, instant, deterministic, easy to
unit-test, and it never confuses a scripted fake LLM. Because the default is DOCUMENT, an unclear
message still goes through retrieval — mistakes are cheap.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Route(StrEnum):
    DIRECT = "direct"
    HISTORY = "history"
    TOOL = "tool"
    FOLLOWUP = "followup"
    COMPARE = "compare"
    DOCUMENT = "document"


# Routes that need the vector store. Everything else skips retrieval.
RETRIEVAL_ROUTES = frozenset({Route.DOCUMENT, Route.FOLLOWUP, Route.COMPARE})
# Routes whose question must be rewritten into a standalone query before searching.
REWRITE_ROUTES = frozenset({Route.FOLLOWUP, Route.COMPARE})

_GREETING = re.compile(
    r"^\s*(hi|hii+|hello|hey|hiya|yo|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|thx|ok(ay)?|"
    r"cool|great|bye|goodbye|who\s+are\s+you|what\s+can\s+you\s+do|help)\b[\s!.?,]*(there|bot|assistant)?[\s!.?]*$",
    re.IGNORECASE,
)
# Small talk about the user themselves: "my name is Aditya", "hi, I'm Aditya", "call me Adi".
# Anchored + short (max 3 words for the name) so "I am looking for the leave policy" still goes to DOCUMENT.
_INTRODUCTION = re.compile(
    r"^\s*((hi|hii+|hello|hey|hiya|yo|good\s+(morning|afternoon|evening))\b[\s!.,]*(there)?[\s!.,]*)?"
    r"(my\s+name\s+is|my\s+name'?s|i\s+am|i'?m|im|call\s+me|this\s+is|it'?s)\s+"
    r"[a-z][\w.'\-]*(\s+[a-z][\w.'\-]*){0,2}[\s!.]*$",
    re.IGNORECASE,
)
# Questions about what the user told us about themselves: "what is my name?", "do you remember my name?"
_PERSONAL_RECALL = re.compile(
    r"(what('?s|\s+is|\s+was)\s+my\s+name|do\s+you\s+(know|remember)\s+(my\s+name|me|who\s+i\s+am)|"
    r"who\s+am\s+i\b|what\s+(did\s+i\s+say|should\s+you\s+call\s+me)|(tell|say)\s+me\s+my\s+name|"
    r"remember\s+my\s+name)",
    re.IGNORECASE,
)
_HISTORY = re.compile(
    r"(what\s+(did|have)\s+i\s+(ask|say|tell|mention)|what\s+was\s+my\s+(first|last|previous|earlier)|"
    r"(sum+ari[sz]e|recap|repeat)\s+(our|this|the)\s+(conversation|chat|discussion)|"
    r"what\s+(have\s+)?we\s+(talk|discuss)|earlier\s+(i|you)\s+(said|asked|mentioned)|"
    r"my\s+(previous|earlier|last)\s+question)",
    re.IGNORECASE,
)
_COMPARE = re.compile(
    r"\b(compare|comparison|difference\s+between|differences?\s+between|versus|vs\.?)\b", re.IGNORECASE
)
_FOLLOWUP = re.compile(
    r"(\b(the\s+)?(first|second|third|fourth|fifth|last|previous|above|earlier)\s+(point|one|item|step|option|part)\b|"
    r"\b(explain|elaborate|clarify|expand|say|repeat)\b.*\b(again|more|further|that|it|this)\b|"
    r"\b(tell\s+me\s+more|go\s+on|why\s+is\s+that|what\s+about\s+(that|it|this)|and\s+(that|it)\b)|"
    r"^\s*(and|but|so|then|why|how\s+about)\b.{0,40}$)",
    re.IGNORECASE,
)
# Short questions whose subject is a pronoun pointing at the previous turn: "what is it?",
# "how many days is that?", "who approves them?". Searched raw, "what is it" matches nothing,
# so these must be rewritten ("what is the leave policy?") before retrieval.
_PRONOUN_FOLLOWUP = re.compile(
    r"^\s*(?=(?:\S+\s+){0,7}\S+[\s?.!]*$)"  # at most ~8 words
    r".*\b(it|its|that|this|they|them|their|those|these|he|she|him|her)\b.*$",
    re.IGNORECASE,
)
_TOOL = re.compile(
    r"(^[\s\d.+\-*/()%^]+$|\bwhat\s+is\s+[\d\s.+\-*/()%^]+\??$|\bcalculate\b|\bcompute\b|"
    r"\b(today'?s?\s+date|current\s+(date|time)|what\s+time\s+is\s+it|what\s+day\s+is\s+(it|today))\b|"
    r"\b(delete|remove|erase)\b.{0,40}\b(document|file|\.md|\.txt)\b)",
    re.IGNORECASE,
)


def classify(message: str, has_history: bool) -> Route:
    """Pick the path for `message`. `has_history` = there are earlier messages in this session."""
    text = message.strip()
    if _GREETING.match(text) or _INTRODUCTION.match(text):
        return Route.DIRECT
    # "What is my name?" is answered from the conversation, never from the documents.
    if _PERSONAL_RECALL.search(text):
        return Route.HISTORY if has_history else Route.DIRECT
    if has_history and _HISTORY.search(text):
        return Route.HISTORY
    if _TOOL.search(text):
        return Route.TOOL
    if _COMPARE.search(text):
        return Route.COMPARE
    # "Explain the second point again" only makes sense when something came before it.
    if has_history and (_FOLLOWUP.search(text) or _PRONOUN_FOLLOWUP.match(text)):
        return Route.FOLLOWUP
    return Route.DOCUMENT
