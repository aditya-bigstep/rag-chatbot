"""PII redaction — mask personal / secret data in a user message BEFORE it reaches the LLM.

`redact_pii(text)` replaces API keys, tokens, passwords, emails, phone numbers, card numbers,
Aadhaar / PAN / SSN numbers and IBANs with `MASK` ("******").

It runs in `app/services.py` on every incoming message, before the engine is called AND before the
message is saved. So the raw value never reaches:
  - the LLM (current turn, query rewrite, or later turns via replayed history),
  - LangSmith traces,
  - the session JSON on disk (or the auto-generated session title).

Rule-based, like `guardrails.py`: fast and predictable, but not exhaustive. For production, consider
adding an NER-based detector (e.g. Microsoft Presidio) behind the same function signature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MASK = "******"


@dataclass(frozen=True)
class Redaction:
    text: str
    kinds: list[str] = field(default_factory=list)  # e.g. ["email", "phone"], for logs / traces (never values)

    @property
    def changed(self) -> bool:
        return bool(self.kinds)


# Patterns with a `keep` group keep that part (the label) and mask only the rest.
# Order matters: the most specific patterns run first, so e.g. a key containing digits is not
# half-matched as a phone number.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "password: hunter2", "api_key=abc123", "token - xyz"  -> mask only the value, keep the label
    (
        "secret",
        re.compile(
            r"(?i)(?P<keep>\b(?:password|passwd|pwd|pass|passcode|pin|otp|secret|api[\s_-]?key|access[\s_-]?key|"
            r"secret[\s_-]?key|auth[\s_-]?token|access[\s_-]?token|token|client[\s_-]?secret)"
            r"(?:\s+is)?\s*[:=\-]\s*)['\"]?[^\s'\",;]{3,}['\"]?"
        ),
    ),
    # "my password is hunter2", "otp is 482913" -> the value must contain a digit, so
    # ordinary sentences like "the password is required" are left alone.
    (
        "secret",
        re.compile(
            r"(?i)(?P<keep>\b(?:password|passwd|passcode|pin|otp|api[\s_-]?key|secret[\s_-]?key)\s+is\s+)"
            r"['\"]?(?=[^\s'\",;]*\d)[^\s'\",;]{3,}['\"]?"
        ),
    ),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer_token", re.compile(r"(?i)(?P<keep>\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}")),
    (
        "api_key",
        re.compile(
            r"\b(?:"
            r"sk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_-]{16,}"  # Anthropic / OpenAI style
            r"|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"  # Stripe
            r"|AKIA[0-9A-Z]{16}"  # AWS access key id
            r"|AIza[0-9A-Za-z_-]{35}"  # Google API key
            r"|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}"  # GitHub
            r"|xox[abprs]-[A-Za-z0-9-]{10,}"  # Slack
            r"|hf_[A-Za-z0-9]{30,}"  # Hugging Face
            r"|glpat-[A-Za-z0-9_-]{20,}"  # GitLab
            r")"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),  # Indian PAN card
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),  # US SSN
    # Card numbers are checked with Luhn below, so random 16-digit IDs are left alone.
    ("card", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")),
    (
        "phone",
        re.compile(
            r"(?<![\w+])(?:"
            r"\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,4}\d{2,4}"  # +91 98765 43210, +1 (415) 555-0100
            r"|0?[6-9]\d{4}[\s-]?\d{5}"  # Indian mobile: 9876543210, 98765-43210, 09876543210
            r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"  # (415) 555-0100, 415-555-0100
            r")(?!\w)"
        ),
    ),
]


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def redact_pii(text: str) -> Redaction:
    """Return `text` with every detected secret / personal value replaced by `MASK`."""
    kinds: list[str] = []

    for kind, pattern in _PATTERNS:

        def _sub(match: re.Match[str], kind: str = kind) -> str:
            value = match.group(0)
            if kind == "card" and not _luhn_ok(re.sub(r"\D", "", value)):
                return value
            kinds.append(kind)
            keep = match.groupdict().get("keep") or ""  # a label like "password: " stays visible
            return keep + MASK

        text = pattern.sub(_sub, text)

    return Redaction(text=text, kinds=sorted(set(kinds)))
