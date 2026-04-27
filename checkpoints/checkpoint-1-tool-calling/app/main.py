"""FastAPI entry point for checkpoint-1 (async tool calling).

This module constructs the singleton `app` object that uvicorn (or any
ASGI server) loads via `app.main:app`. The lifespan context manager from
`app.lifespan` is responsible for building and tearing down the runtime
state (settings, LLM client, tool registry) that routes depend on.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes import health, tools

# A single FastAPI instance per process. The `lifespan` callable wires up
# long-lived dependencies (LLM client, tool registry) on startup and
# disposes them on shutdown — keeping construction out of import-time.
app = FastAPI(
    title="Project 4 — Agentic AI Service (Checkpoint 1: Tool Calling)",
    version="0.1.0",
    description=(
        "Async FastAPI service that lets an LLM choose and call tools. "
        "Demonstrates the engineering foundation: pydantic-settings, "
        "lifespan-managed singletons, dependency injection via Depends, "
        "structured logging, async LLM client, and the tool-call primitive."
    ),
    lifespan=lifespan,
)

# Routers are mounted in dependency order: health first (cheap, no LLM
# required), then the tool-calling endpoints. Adding a new router here is
# the only change needed to expose a new domain over HTTP.
app.include_router(health.router)
app.include_router(tools.router)
