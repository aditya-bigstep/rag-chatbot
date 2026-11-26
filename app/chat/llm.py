"""Chat model factory.

Pick the provider and model in `.env` (LLM_PROVIDER / LLM_MODEL). If the main
provider errors out, LLM_FALLBACK_PROVIDER takes over for that call.
"""

from __future__ import annotations

import logging
import os

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "google": "gemini-3.8-flash",
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
}
KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_model(provider: str, model: str = "") -> BaseChatModel:
    provider = provider.strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM provider {provider!r}; use one of {sorted(DEFAULT_MODELS)}")
    name = model or DEFAULT_MODELS[provider]

    if provider == "groq":
        from langchain_groq import ChatGroq

        effort = "low" if "gpt-oss" in name else None  # reasoning tokens count against the free-tier limit
        return ChatGroq(model=name, temperature=0, max_retries=5, reasoning_effort=effort)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=name, temperature=0, max_retries=1, timeout=20)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model_name=name, temperature=0, max_retries=2, timeout=60, stop=None)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=name, temperature=0, max_retries=2)


def has_key(provider: str) -> bool:
    return bool(os.getenv(KEY_NAMES.get(provider.strip().lower(), ""), "").strip())


def get_llm(fast: bool = False) -> BaseChatModel:
    """The configured chat model, wrapped with the fallback provider when one is set.

    `fast=True` asks for the cheaper model (LLM_FAST_MODEL) used for small jobs.
    """
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    model = (settings.llm_fast_model if fast else "") or settings.llm_model
    primary = build_model(provider, model)

    fallback_provider = settings.llm_fallback_provider.strip().lower()
    fallback_model = settings.llm_fallback_model or DEFAULT_MODELS.get(fallback_provider, "")
    same = fallback_provider == provider and fallback_model == (model or DEFAULT_MODELS[provider])
    if not fallback_provider or same or not has_key(fallback_provider):
        return primary
    try:
        fallback = build_model(fallback_provider, fallback_model)
    except Exception:
        logger.warning("Fallback provider %s could not be built", fallback_provider, exc_info=True)
        return primary
    return primary.with_fallbacks([fallback])  # type: ignore[return-value]
