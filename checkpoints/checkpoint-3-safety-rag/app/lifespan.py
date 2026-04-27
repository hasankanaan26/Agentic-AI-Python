"""FastAPI lifespan: build singletons on startup, dispose on shutdown.

CP3 introduces RAG (Chroma + embeddings) and the LangGraph agent.
The orchestrator and trace store arrive in CP4.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from app.agents.langgraph import LangGraphAgentRunner
from app.logging_config import configure_logging, get_logger
from app.rag.ingest import ensure_indexed
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import VectorStore
from app.settings import Settings, get_settings
from app.tools.registry import ToolRegistry


@dataclass
class AppState:
    """Bag of process-wide singletons attached to ``app.state``.

    Constructed once on startup and read by every request via the
    ``Depends(get_app_state)`` chain in ``app.deps``. Holding these as a
    single dataclass keeps DI wiring trivial — every dep just pulls the
    right field off of one object.
    """

    settings: Settings
    llm: LLMService
    embeddings: EmbeddingService
    vector_store: VectorStore
    registry: ToolRegistry
    langgraph_runner: LangGraphAgentRunner


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build singletons before the app accepts traffic; dispose on shutdown.

    Yields control back to FastAPI once everything is ready. The teardown in
    the ``finally`` block runs on a clean shutdown AND on most crash paths,
    so HTTP clients are always closed.
    """
    settings = get_settings()
    # Configure logging FIRST so every subsequent ``log.info(...)`` is captured
    # in the configured JSON / console renderer.
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger(__name__)
    log.info("startup", provider=settings.llm_provider, model=settings.model_name())

    # Construction order matters: the registry needs embeddings + vector_store
    # so the knowledge_search tool can be wired, and the LangGraph runner
    # needs the fully-built registry to expose its tools to the LLM.
    llm = LLMService(settings)
    embeddings = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_path)
    registry = ToolRegistry.build(settings, embeddings, vector_store)
    langgraph_runner = LangGraphAgentRunner(settings, registry)

    # Stash everything on app.state so request-scoped Depends can fetch it.
    app.state.app_state = AppState(
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        registry=registry,
        langgraph_runner=langgraph_runner,
    )

    # Best-effort RAG warm-up. ``ensure_indexed`` is idempotent and swallows
    # transient errors so a flaky embedding endpoint doesn't break boot —
    # the operator can always re-run via POST /rag/ingest.
    await ensure_indexed(
        embeddings=embeddings,
        store=vector_store,
        knowledge_path=settings.knowledge_data_path,
    )

    try:
        yield
    finally:
        # Close async HTTP clients to flush in-flight connections cleanly.
        log.info("shutdown")
        await llm.aclose()
        await embeddings.aclose()
