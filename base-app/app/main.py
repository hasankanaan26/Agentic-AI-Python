"""Base app — minimal scaffold for live coding sessions.

Each checkpoint builds out from this starting point. The skeleton already
has the engineering foundation (settings, lifespan, logging, DI) so the
session focuses on the agentic primitives, not on plumbing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from app.logging_config import configure_logging, get_logger
from app.settings import Settings, get_settings


@dataclass
class AppState:
    """Container for objects that live for the lifetime of the FastAPI app.

    Stored on ``app.state.app_state`` during startup and retrieved per
    request via the ``get_app_state`` dependency. Checkpoints will grow
    this dataclass with extra fields (LLM client, tool registry, vector
    store, etc.) — keeping it as a single typed object avoids leaking
    untyped ``app.state.*`` lookups across the codebase.
    """

    settings: Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and build shared state on startup; tear down on shutdown.

    FastAPI calls the function once when the ASGI server starts (before
    serving the first request) and resumes after ``yield`` on shutdown.
    Anything that needs deterministic init/cleanup (LLM clients, DB
    connections, etc.) belongs here rather than at module import time.
    """
    # Settings are validated here so a misconfigured deploy fails fast,
    # before any traffic is accepted.
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger(__name__)
    log.info("base_app_startup", provider=settings.llm_provider)
    # Stash shared state on the app object so request handlers can reach
    # it through dependency injection (see ``get_app_state``).
    app.state.app_state = AppState(settings=settings)
    yield
    log.info("base_app_shutdown")


app = FastAPI(
    title="Project 4 — Base App",
    version="0.0.0",
    description="Empty scaffold for live coding. Each checkpoint adds layers on top.",
    lifespan=lifespan,
)


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency that returns the per-app ``AppState`` singleton.

    Using ``Depends(get_app_state)`` (instead of touching ``request.app.state``
    directly inside endpoints) keeps the type signature explicit and makes
    overrides trivial in tests via ``app.dependency_overrides``.
    """
    return request.app.state.app_state


@app.get("/health")
async def health(
    state: Annotated[AppState, Depends(get_app_state)],
) -> dict:
    """Liveness probe for orchestrators (Docker, k8s, etc.).

    Returns the configured LLM provider so an operator can sanity-check
    which credentials the running process actually picked up.
    """
    return {"status": "healthy", "checkpoint": "base-app", "provider": state.settings.llm_provider}
