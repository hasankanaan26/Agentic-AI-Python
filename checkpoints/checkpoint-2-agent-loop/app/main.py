"""FastAPI entry point for checkpoint-2 (agent loop).

This module just wires the app together — it should stay tiny. All
singletons are built in :mod:`app.lifespan`, all routes are defined in
:mod:`app.routes`, and the dependency providers live in :mod:`app.deps`.
Run with ``uvicorn app.main:app --reload``.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes import agent, health, tools

# The FastAPI instance. ``lifespan`` is the async context manager that
# constructs/disposes process-wide singletons (settings, LLM, registry).
app = FastAPI(
    title="Project 4 — Agentic AI Service (Checkpoint 2: Agent Loop)",
    version="0.2.0",
    description=(
        "Async FastAPI agent service. Adds the from-scratch agent loop, "
        "tenacity retries, and structured tool errors on top of the CP1 "
        "tool-calling primitives."
    ),
    lifespan=lifespan,
)

# Mount routers. Order is purely cosmetic (it affects the OpenAPI doc layout).
app.include_router(health.router)
app.include_router(tools.router)
app.include_router(agent.router)
