"""MILESTONE 1 — The LLM.

Return a LangChain chat model for the provider/model set in `.env`.

  LLM_PROVIDER=anthropic  -> `uv add langchain-anthropic`, use ChatAnthropic
  LLM_PROVIDER=openai     -> `uv add langchain-openai`,    use ChatOpenAI

The API key is read by the provider package from the environment
(ANTHROPIC_API_KEY / OPENAI_API_KEY). Never hard-code keys.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_llm() -> BaseChatModel:
    """OUT: a ready-to-use chat model (supports `.invoke(messages)` and `.bind_tools(tools)`)."""
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model or DEFAULT_OPENAI_MODEL, temperature=0)

    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}; expected 'openai'")
