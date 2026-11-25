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

    @property
    def checkpoint_db(self) -> Path:
        """SQLite file where LangGraph saves its checkpoints (see app/chat/checkpointing.py)."""
        return self.data_dir / "checkpoints.sqlite"

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
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for the running app. Call `get_settings.cache_clear()` in tests."""
    return load_settings()
