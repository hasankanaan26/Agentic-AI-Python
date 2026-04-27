"""FastAPI lifespan: build singletons on startup, dispose on shutdown.

This is the dependency-injection root: every long-lived object the app
needs (settings, LLM client, tool registry) is constructed exactly once
here and stashed on ``app.state`` so the routes can read it via Depends.

CP2 owns: settings, LLM client, tool registry. RAG, the LangGraph agent,
the orchestrator and trace store arrive in later checkpoints.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from app.logging_config import configure_logging, get_logger
from app.services.llm import LLMService
from app.settings import Settings, get_settings
from app.tools.registry import ToolRegistry


@dataclass
class AppState:
    """Container for the process-wide singletons the routes depend on.

    Held on ``app.state.app_state`` and read back through ``deps.get_app_state``.
    Using a dataclass (rather than separate attributes) keeps the wiring tidy
    and makes it trivial to add more singletons in later checkpoints.
    """

    settings: Settings
    llm: LLMService
    registry: ToolRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup before yielding and shutdown after.

    On startup we configure logging, build the LLM client and tool registry,
    and attach them to ``app.state``. On shutdown we close the underlying
    HTTP client(s) so we don't leak sockets.
    """
    settings = get_settings()
    # Configure logging FIRST so subsequent startup messages obey the format.
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger(__name__)
    log.info("startup", provider=settings.llm_provider, model=settings.model_name())

    # Build the singletons. These constructors do no I/O — actual network
    # calls only happen when a route invokes them.
    llm = LLMService(settings)
    registry = ToolRegistry.build(settings)

    app.state.app_state = AppState(settings=settings, llm=llm, registry=registry)
    try:
        yield
    finally:
        # Shutdown phase: always run, even if startup or requests raised.
        log.info("shutdown")
        await llm.aclose()
