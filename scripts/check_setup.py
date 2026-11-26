"""Checks that this machine can run ResearchPilot: settings present, keys working, models reachable.

    uv run python scripts/check_setup.py

Exits with 1 when something the app cannot run without is broken (no working LLM, no embeddings).
Everything else is reported as a warning.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.chat.llm import DEFAULT_MODELS, KEY_NAMES, build_model  # noqa: E402
from app.config import load_settings  # noqa: E402

GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
problems: list[str] = []


def ok(name: str, detail: str = "") -> None:
    print(f"  {GREEN}ok{RESET}    {name:<22} {detail}")


def warn(name: str, detail: str) -> None:
    print(f"  {YELLOW}warn{RESET}  {name:<22} {detail}")


def fail(name: str, detail: str) -> None:
    print(f"  {RED}fail{RESET}  {name:<22} {detail}")
    problems.append(name)


def check_llm(provider: str, model: str) -> bool:
    """One tiny real call. True when the model answered."""
    provider = provider.strip().lower()
    label = f"LLM {provider}"
    if provider not in DEFAULT_MODELS:
        warn(label, "unknown provider name")
        return False
    if not os.getenv(KEY_NAMES[provider], "").strip():
        warn(label, f"{KEY_NAMES[provider]} is empty")
        return False
    started = time.time()
    try:
        build_model(provider, model).invoke("Reply with the single word: ok")
    except Exception as exc:
        warn(label, f"{model or DEFAULT_MODELS[provider]} did not answer: {str(exc)[:90]}")
        return False
    ok(label, f"{model or DEFAULT_MODELS[provider]} answered in {time.time() - started:.1f}s")
    return True


def check_search(provider: str) -> None:
    from app.research.search import KEY_NAMES as SEARCH_KEYS
    from app.research.search import PROVIDERS

    key_name = SEARCH_KEYS.get(provider)
    if key_name and not os.getenv(key_name, "").strip():
        warn(f"search {provider}", f"{key_name} is empty, the free search will be used instead")
        return
    try:
        rows = PROVIDERS[provider]("what is retrieval augmented generation", 3)
    except Exception as exc:
        warn(f"search {provider}", f"failed: {str(exc)[:90]}. The app falls back to the free search")
        return
    if rows:
        ok(f"search {provider}", f"{len(rows)} results")
    else:
        warn(f"search {provider}", "returned nothing")


def check_langsmith() -> None:
    key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true" or not key:
        warn("LangSmith", "tracing is off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")
        return
    try:
        url = "https://api.smith.langchain.com/api/v1/sessions?limit=1"
        response = httpx.get(url, headers={"x-api-key": key}, timeout=15)
    except httpx.HTTPError as exc:
        warn("LangSmith", f"unreachable: {type(exc).__name__}")
        return
    if response.status_code == 200:
        ok("LangSmith", f"tracing to project {os.getenv('LANGSMITH_PROJECT', 'default')}")
    else:
        warn("LangSmith", f"key rejected ({response.status_code})")


def check_accounts(settings) -> None:  # type: ignore[no-untyped-def]
    if not settings.auth_enabled:
        warn("accounts", "AUTH_ENABLED is false: anyone who can open the page can use it")
        return
    if not settings.brevo_api_key:
        warn("accounts", "no BREVO_API_KEY: sign-up codes are printed in this terminal instead of e-mailed")
        return
    from app.auth.email import BrevoMailer

    status = BrevoMailer.from_settings(settings).status()
    if status == "ok":
        ok("accounts (Brevo)", "key works and a sender address is verified")
    else:
        warn("accounts (Brevo)", f"{status}. Sign-up e-mails will not arrive")


def check_embeddings(settings) -> None:  # type: ignore[no-untyped-def]
    from app.vectordb.embeddings import get_embedder

    started = time.time()
    try:
        size = get_embedder().dimension
    except Exception as exc:
        fail("embeddings", f"{settings.embedding_model} could not load: {str(exc)[:90]}")
        return
    name = settings.embedding_model or "default model"
    ok("embeddings", f"{name}, {size} numbers per vector, {time.time() - started:.1f}s")


def main() -> int:
    settings = load_settings()
    print("\nChecking the setup\n")

    primary = check_llm(settings.llm_provider, settings.llm_model)
    backup = False
    if settings.llm_fallback_provider:
        backup = check_llm(settings.llm_fallback_provider, settings.llm_fallback_model)
    if not primary and not backup:
        fail("LLM", "no model answered. Fill in a key in .env (GOOGLE_API_KEY or GROQ_API_KEY)")
    elif not primary:
        warn("LLM", "the main model is down, so the fallback will handle every call")

    check_embeddings(settings)
    check_search(settings.search_provider)
    check_langsmith()
    check_accounts(settings)

    if problems:
        print(f"\n{RED}Cannot start:{RESET} fix {', '.join(problems)} and run ./start.sh again.\n")
        return 1
    print(f"\n{GREEN}Ready.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
