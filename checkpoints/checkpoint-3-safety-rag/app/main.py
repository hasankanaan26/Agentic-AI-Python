"""FastAPI entry point for checkpoint-3 (safety + RAG + LangGraph).

This module constructs the ``FastAPI`` application instance that ``uvicorn``
imports as ``app.main:app``. It wires up:

- the lifespan context manager (singleton construction + shutdown),
- every feature router (health, tools, agent, safety, rag).

No request-handling logic lives here on purpose — keeping the entry point
small makes the wiring obvious during code review.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes import agent, health, rag, safety, tools

# The single ASGI app object. ``lifespan`` builds shared singletons
# (LLM client, vector store, tool registry, etc.) on startup and disposes
# them on shutdown — see ``app.lifespan`` for the wiring details.
app = FastAPI(
    title="Project 4 — Agentic AI Service (Checkpoint 3: Safety + RAG)",
    version="0.3.0",
    description=(
        "Async FastAPI agentic service with prompt-injection detection, tool "
        "permissions, the LangGraph ReAct agent, and a RAG-backed "
        "knowledge_search tool."
    ),
    lifespan=lifespan,
)

# Order is cosmetic only (it controls the OpenAPI doc grouping); each router
# carries its own URL prefix and tags so there's no path collision risk.
app.include_router(health.router)
app.include_router(tools.router)
app.include_router(agent.router)
app.include_router(safety.router)
app.include_router(rag.router)
