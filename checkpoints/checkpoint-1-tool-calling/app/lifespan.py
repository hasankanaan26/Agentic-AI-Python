"""FastAPI lifespan: build singletons on startup, dispose on shutdown.

This is the single place where long-lived dependencies are constructed —
no global state, no module-level side effects. The lifespan runs once
when the server boots and once when it shuts down. Everything routes
need is attached to `app.state.app_state` and reached via `app.deps`.

CP1 owns: settings, LLM client, tool registry. RAG, the LangGraph agent,
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
    """Bundle of process-wide singletons built by the lifespan.

    A single instance lives on `app.state.app_state` for the lifetime of
    the server. Routes pull individual fields out via the dependency
    helpers in `app.deps`, which keeps Depends signatures small and makes
    each route's actual dependencies explicit.
    """

    settings: Settings
    llm: LLMService
    registry: ToolRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown context for the FastAPI application.

    Yields control back to FastAPI once initialisation is done; the code
    after the `yield` runs when the server is shutting down (e.g. SIGTERM
    in production, Ctrl-C locally).

    Args:
        app: The FastAPI application; we attach state to `app.state`.

    Yields:
        Nothing — the body of the contextmanager is the running app.
    """
    # Settings is the *first* thing we touch — validation errors here
    # should crash the process before any heavier resources are built.
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger(__name__)
    log.info("startup", provider=settings.llm_provider, model=settings.model_name())

    # Build collaborators. Any exception here propagates and prevents the
    # server from accepting traffic — better than booting a half-broken app.
    llm = LLMService(settings)
    registry = ToolRegistry.build(settings)

    # Stash on `app.state` so DI helpers can pull it out per-request.
    app.state.app_state = AppState(settings=settings, llm=llm, registry=registry)
    try:
        yield
    finally:
        # `finally` ensures we always close the HTTP pool, even if the
        # server is shutting down due to an unhandled error elsewhere.
        log.info("shutdown")
        await llm.aclose()
