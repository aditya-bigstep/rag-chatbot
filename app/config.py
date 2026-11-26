"""Application settings, read from environment variables (and `.env`).

Every path the app writes to comes from here, so tests and the trainer's
evaluator can point the app at temporary folders without touching your data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Values already set in the real environment win over `.env`.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None or value == "" else value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    sessions_dir: Path
    documents_dir: Path
    vector_dir: Path
    chunking_strategy: str = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    min_score: float = 0.0  # chunks scoring below this are thrown away before reaching the LLM
    llm_provider: str = "anthropic"
    llm_model: str = ""
    embedding_model: str = ""
    max_upload_bytes: int = 5 * 1024 * 1024
    allowed_upload_suffixes: tuple[str, ...] = field(default=(".md", ".txt"))

    # --- Advanced chat-engine features (see app/chat/options.py for what each one does) ---
    smart_routing: bool = True  # classify each message and skip retrieval when it is not needed
    always_retrieve: bool = False  # True = ignore routing and retrieve on every single turn
    guardrails: bool = True  # input validation + prompt-injection / unsafe-request blocking
    grounding_mode: str = "enforce"  # "off" | "warn" | "enforce" — is the answer backed by the documents?
    hitl_enabled: bool = True  # pause for human approval before dangerous tool calls
    memory_max_messages: int = 10  # recent messages sent verbatim; older ones get summarised
    memory_summarize: bool = True  # summarise older messages (costs one extra LLM call, cached)

    llm_fallback_provider: str = "groq"
    llm_fallback_model: str = ""
    llm_fast_model: str = ""  # cheaper model for small jobs (judging, edit planning); empty = main model
    retrieval_mode: str = "dense"  # dense | bm25 | hybrid

    search_provider: str = "ddgs"
    research_max_sections: int = 5
    research_sources_per_section: int = 4
    research_parallel_sections: int = 2
    research_verify: bool = True
    research_max_llm_calls: int = 40

    auth_enabled: bool = False  # true = an account is needed for everything except the welcome page
    brevo_api_key: str = ""
    brevo_sender_email: str = ""
    brevo_sender_name: str = "ResearchPilot"
    app_base_url: str = "http://127.0.0.1:8000"
    session_ttl_days: int = 14

    @property
    def checkpoint_db(self) -> Path:
        """SQLite file where LangGraph saves its checkpoints (see app/chat/checkpointing.py)."""
        return self.data_dir / "checkpoints.sqlite"

    @property
    def accounts_db(self) -> Path:
        return self.data_dir / "accounts.sqlite"

    @property
    def research_dir(self) -> Path:
        return self.data_dir / "research"

    def ensure_dirs(self) -> None:
        for path in (self.sessions_dir, self.documents_dir, self.vector_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build a fresh Settings object from the current environment."""
    data_dir = _env_path("DATA_DIR", PROJECT_ROOT / "data")
    return Settings(
        data_dir=data_dir,
        sessions_dir=_env_path("SESSIONS_DIR", data_dir / "sessions"),
        documents_dir=_env_path("DOCUMENTS_DIR", data_dir / "documents"),
        vector_dir=_env_path("VECTOR_DIR", data_dir / "vectorstore"),
        chunking_strategy=os.getenv("CHUNKING_STRATEGY", "fixed"),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
        top_k=_env_int("TOP_K", 4),
        min_score=_env_float("MIN_SCORE", 0.0),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        smart_routing=_env_bool("SMART_ROUTING", True),
        always_retrieve=_env_bool("ALWAYS_RETRIEVE", False),
        guardrails=_env_bool("GUARDRAILS", True),
        grounding_mode=os.getenv("GROUNDING_MODE", "enforce").strip().lower(),
        hitl_enabled=_env_bool("HITL_ENABLED", True),
        memory_max_messages=_env_int("MEMORY_MAX_MESSAGES", 10),
        memory_summarize=_env_bool("MEMORY_SUMMARIZE", True),
        llm_fallback_provider=os.getenv("LLM_FALLBACK_PROVIDER", "groq"),
        llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL", ""),
        llm_fast_model=os.getenv("LLM_FAST_MODEL", ""),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "dense").strip().lower(),
        search_provider=os.getenv("SEARCH_PROVIDER", "ddgs").strip().lower(),
        research_max_sections=_env_int("RESEARCH_MAX_SECTIONS", 5),
        research_sources_per_section=_env_int("RESEARCH_SOURCES_PER_SECTION", 4),
        research_parallel_sections=_env_int("RESEARCH_PARALLEL_SECTIONS", 2),
        research_verify=_env_bool("RESEARCH_VERIFY", True),
        research_max_llm_calls=_env_int("RESEARCH_MAX_LLM_CALLS", 40),
        auth_enabled=_env_bool("AUTH_ENABLED", False),
        brevo_api_key=os.getenv("BREVO_API_KEY", ""),
        brevo_sender_email=os.getenv("BREVO_SENDER_EMAIL", ""),
        brevo_sender_name=os.getenv("BREVO_SENDER_NAME", "ResearchPilot"),
        app_base_url=os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        session_ttl_days=_env_int("SESSION_TTL_DAYS", 14),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for the running app. Call `get_settings.cache_clear()` in tests."""
    return load_settings()
