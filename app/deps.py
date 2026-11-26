"""Wires your modules into the web app. You should not need to change this file.

`build_components` calls each `get_*` factory. A factory that still raises
NotImplementedError is skipped, so the app keeps running while you work
through the milestones. Tests (and the trainer's evaluator) replace the whole
bundle with `app.dependency_overrides[get_components] = ...`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastapi import Request

from app.chat.checkpointing import get_checkpointer
from app.chat.engine import ChatEngine, PlaceholderEngine, get_engine
from app.chat.options import EngineOptions
from app.config import Settings, get_settings
from app.research.deps import build_deps
from app.research.service import ResearchService
from app.storage.json_store import SessionStore
from app.tools import get_tools
from app.tools.admin_tools import make_admin_tools
from app.vectordb.chunking import Chunker, get_chunker
from app.vectordb.embeddings import Embedder, get_embedder
from app.vectordb.retriever import Retriever, get_retriever
from app.vectordb.store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class Components:
    engine: ChatEngine
    chunker: Chunker | None = None
    embedder: Embedder | None = None
    store: VectorStore | None = None
    retriever: Retriever | None = None
    research: ResearchService | None = None
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def can_index(self) -> bool:
        return None not in (self.chunker, self.embedder, self.store)


def _try(name: str, factory: Callable[..., T], missing: dict[str, str], **kwargs: Any) -> T | None:
    try:
        return factory(**kwargs)
    except NotImplementedError as exc:
        missing[name] = str(exc)
    except Exception as exc:  # a broken module must not take the whole app down
        logger.exception("Could not build %s", name)
        missing[name] = f"{type(exc).__name__}: {exc}"
    return None


def build_components(settings: Settings | None = None) -> Components:
    settings = settings or get_settings()
    missing: dict[str, str] = {}
    chunker = _try("chunker", get_chunker, missing)
    embedder = _try("embedder", get_embedder, missing)
    store = _try("vector_store", get_vector_store, missing)
    retriever = None
    if store is not None and embedder is not None:
        retriever = _try("retriever", get_retriever, missing, store=store, embedder=embedder)
    # Switch on the advanced engine features from .env (memory, routing, guardrails, checkpointing, HITL).
    options = EngineOptions.from_settings(settings)
    checkpointer = _try("checkpointer", get_checkpointer, missing, db_path=settings.checkpoint_db)
    tools = get_tools()
    if options.hitl and checkpointer is not None:  # dangerous tools exist ONLY when approval can pause them
        tools = [*tools, *make_admin_tools(settings.documents_dir, store)]
    engine = _try(
        "engine", get_engine, missing, retriever=retriever, tools=tools, options=options, checkpointer=checkpointer
    )
    research = None
    if embedder is not None and checkpointer is not None:
        research = _try(
            "research",
            _build_research,
            missing,
            settings=settings,
            embedder=embedder,
            checkpointer=checkpointer,
            chunker=chunker,
            store=store,
        )
    return Components(
        research=research,
        engine=engine or PlaceholderEngine(),
        chunker=chunker,
        embedder=embedder,
        store=store,
        retriever=retriever,
        missing=missing,
    )


def _build_research(
    settings: Settings, embedder: Any, checkpointer: Any, chunker: Any, store: Any
) -> ResearchService:
    return ResearchService(settings, build_deps(settings, embedder), checkpointer, chunker, store)


def get_store(request: Request) -> SessionStore:
    store: SessionStore = request.app.state.session_store
    return store


def get_components(request: Request) -> Components:
    state = request.app.state
    if getattr(state, "components", None) is None:
        state.components = build_components(state.settings)
        for name, reason in state.components.missing.items():
            logger.info("Not available yet: %s (%s)", name, reason)
    components: Components = state.components
    return components
