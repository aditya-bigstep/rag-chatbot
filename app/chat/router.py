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
    OVERVIEW = "overview"  # report chat only: "summarise it" -> read the whole report, not the top-k chunks


# Routes that need the vector store. Everything else skips retrieval.
RETRIEVAL_ROUTES = frozenset({Route.DOCUMENT, Route.FOLLOWUP, Route.COMPARE, Route.OVERVIEW})
# Routes whose question must be rewritten into a standalone query before searching.
REWRITE_ROUTES = frozenset({Route.FOLLOWUP, Route.COMPARE})

_GREETING = re.compile(
    r"^\s*(hi|hii+|hello|hey|hiya|yo|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|thx|ok(ay)?|"
    r"cool|great|bye|goodbye|who\s+are\s+you|what\s+can\s+you\s+do|help)\b[\s!.?,]*(there|bot|assistant)?[\s!.?]*$",
    re.IGNORECASE,
)
# Small talk by kind, so a caller can pick a fitting canned reply (see app/research/intake.py).
# Every pattern must match the WHOLE message: "hi" is small talk, "hi, what is the leave policy?" is not.
_HELLO = r"(hi+|hello+|hey+|heya|hiya|yo|sup|hola|namaste|namaskar|greetings|good\s+(morning|afternoon|evening|day))"
_TAIL = r"[\s!.?,]*(there|bot|assistant|buddy|pilot|researchpilot|everyone|all)?[\s!.?,]*"
_SMALL_TALK: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, re.compile(rf"^\s*({_HELLO}{_TAIL})?({body}){_TAIL}$", re.IGNORECASE))
    for kind, body in (
        (
            "wellbeing",
            r"how\s+(are|r)\s+(you|u)(\s+doing)?(\s+today)?|how('?s|\s+is)\s+it\s+going|how\s+do\s+you\s+do|"
            r"what'?s\s+up|wass+up",
        ),
        (
            "thanks",
            r"thanks?(\s+a\s+lot|\s+so\s+much)?|thank\s+(you|u)(\s+(so|very)\s+much)?|thx|ty|cheers|"
            r"(ok(ay)?|cool|great|nice|awesome|perfect|got\s+it)([\s,]+(thanks?|thank\s+you))?",
        ),
        ("bye", r"bye+|goodbye|good\s+night|see\s+(you|ya)(\s+later)?|cya|take\s+care"),
        (
            "identity",
            r"who\s+are\s+you|what\s+are\s+you|what\s+(can|do)\s+you\s+do|help(\s+me)?|how\s+does\s+(this|it)\s+work|"
            r"what\s+is\s+this|how\s+do\s+(i|you)\s+use\s+(this|it)|are\s+you\s+(an?\s+)?(bot|ai|human|robot|real)|"
            r"what('?s|\s+is)\s+your\s+name|introduce\s+yourself",
        ),
        ("greeting", _HELLO),
    )
)


def small_talk(message: str) -> str | None:
    """The kind of small talk this message is ("greeting", "thanks", "bye", "identity", "wellbeing",
    "introduction"), or None when it is a real question."""
    text = message.strip()
    for kind, pattern in _SMALL_TALK:
        if pattern.match(text):
            return kind
    return "introduction" if _INTRODUCTION.match(text) else None


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
    r"remember\s+my\s+name|what\s+do\s+you\s+(know|remember)\s+about\s+me|where\s+do\s+i\s+(live|work)|"
    r"where\s+am\s+i\s+from|what\s+did\s+i\s+(ask|tell)\s+you\s+to\s+remember)",
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
    r"(^[\s\d.+\-*/()%^x×÷]+[=?\s]*$|\bwhat\s+is\s+[\d\s.+\-*/()%^x×÷]+\??$|\bcalculate\b|\bcompute\b|"
    r"\d\s*%\s*of\s+\d|\b(square\s+root|sqrt)\s+of\s+\d|"
    r"\b(today'?s?\s+date|current\s+(date|time)|what\s+time\s+is\s+it|what\s+day\s+is\s+(it|today|tomorrow)|"
    r"what('?s|\s+is)\s+the\s+(date|time)|(date|time|day)\s+(today|now|right\s+now)|which\s+day\s+is\s+(it|today))\b|"
    r"\b(delete|remove|erase)\b.{0,40}\b(document|file|\.md|\.txt)\b)",
    re.IGNORECASE,
)

# Whole-report asks: "summarise it", "give me an overview", "key takeaways", "tl;dr". A similarity search
# for "summarise it" matches nothing useful, so these read every chunk of the report instead.
# The first branch tolerates typos ("sumamrise", "sumarize", "summerise").
_OVERVIEW = re.compile(
    r"(\bsum+[ae]?m*[ae]?r+[iy]|\boverview\b|\btl;?dr\b|\bgist\b|\brecap\b|\bin\s+(brief|short|a\s+nutshell)\b|"
    r"\b(key|main|major|important|top)\s+(points?|takeaways?|findings?|ideas?|insights?|conclusions?|themes?)\b|"
    r"\bwhat\s+(is|are|does)\s+(this|the|it)\s*(report|research|document|study)?\s*(all\s+)?(about|say|cover|conclude)\b|"
    r"\b(explain|describe|outline)\s+(this|the)\s+(whole\s+|entire\s+)?(report|research|document)\b)",
    re.IGNORECASE,
)


# Telling the bot something to keep, or to drop: "remember that I prefer short answers", "forget my name",
# "I live in Pune". The fact itself is stored by app/chat/profile.py; here it only means "no search needed".
_MEMORY_COMMAND = re.compile(
    r"^\s*(please\s+)?(remember|note|keep\s+in\s+mind|don'?t\s+forget)\s+(that|this|:)|"
    r"\bforget\s+(my\s+(name|location|city|workplace|company|notes?)|me\b|who\s+i\s+am|(everything|all)\b.*\babout\s+me)|"
    r"\b(clear|wipe|erase|reset)\s+(your|my)\s+memory\b|"
    r"^\s*i\s+(live|stay|work)\s+(in|at|for)\s+[\w .'\-]{2,40}[\s.!]*$|"
    r"^\s*i('?m|\s+am)\s+from\s+[\w .'\-]{2,40}[\s.!]*$",
    re.IGNORECASE,
)


def is_memory_command(message: str) -> bool:
    return bool(_MEMORY_COMMAND.search(message))


def is_personal_recall(message: str) -> bool:
    return bool(_PERSONAL_RECALL.search(message))


def is_overview(message: str) -> bool:
    """True when the message asks about the report as a whole rather than one fact in it."""
    return bool(_OVERVIEW.search(message))


def classify(message: str, has_history: bool) -> Route:
    """Pick the path for `message`. `has_history` = there are earlier messages in this session."""
    text = message.strip()
    if _GREETING.match(text) or small_talk(text) or is_memory_command(text):
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
