"""FastAPI lifespan: build singletons on startup, dispose on shutdown.

Engineering standard: things that should exist exactly once per process
(LLM client, embedding client, vector store, tool registry, agent runner,
trace store) live on `app.state` and are exposed to handlers via the
DI providers in `deps.py`. This is the only place we instantiate them.
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
from app.services.orchestrator import OrchestratorService
from app.services.tracer import TraceStore, langsmith_enabled
from app.services.vector_store import VectorStore
from app.settings import Settings, get_settings
from app.tools.registry import ToolRegistry


@dataclass
class AppState:
    """Container for every singleton built at startup.

    Held on ``app.state.app_state`` and surfaced to handlers via the
    dependency providers in :mod:`app.deps`.
    """

    settings: Settings
    llm: LLMService
    embeddings: EmbeddingService
    vector_store: VectorStore
    registry: ToolRegistry
    langgraph_runner: LangGraphAgentRunner
    orchestrator: OrchestratorService
    traces: TraceStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI startup/shutdown context.

    Builds singletons in dependency order on entry, runs the best-effort
    RAG warmup, then yields control to the server. On exit, closes the
    LLM and embedding HTTP clients so we don't leak sockets.
    """
    settings = get_settings()
    # Configure logging FIRST so every subsequent line is structured.
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger(__name__)
    log.info(
        "startup",
        provider=settings.llm_provider,
        model=settings.model_name(),
        langsmith=langsmith_enabled(),
    )

    # Build singletons in dependency order. Each subsequent line may rely
    # on the objects above it, so the ordering is intentional.
    llm = LLMService(settings)
    embeddings = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_path)
    registry = ToolRegistry.build(settings, embeddings, vector_store)
    langgraph_runner = LangGraphAgentRunner(settings, registry)
    traces = TraceStore(max_size=settings.trace_store_max)
    orchestrator = OrchestratorService(
        # Reuse the LangChain chat client the runner already constructed --
        # one HTTP client per process.
        chat_model=langgraph_runner.chat_model,
        runner=langgraph_runner,
        traces=traces,
        settings=settings,
    )

    app.state.app_state = AppState(
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        registry=registry,
        langgraph_runner=langgraph_runner,
        orchestrator=orchestrator,
        traces=traces,
    )

    # Best-effort RAG warmup. Won't crash boot if a provider is offline --
    # ensure_indexed swallows exceptions and logs them.
    await ensure_indexed(
        embeddings=embeddings,
        store=vector_store,
        knowledge_path=settings.knowledge_data_path,
    )

    try:
        yield
    finally:
        # Always close clients on shutdown, even if startup raised partway.
        log.info("shutdown")
        await llm.aclose()
        await embeddings.aclose()
