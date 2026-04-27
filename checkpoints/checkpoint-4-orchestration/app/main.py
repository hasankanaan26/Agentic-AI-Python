"""FastAPI entry point for checkpoint-4.

All routes are async, all dependencies are wired through Depends().
The lifespan handler is the ONLY place we instantiate clients.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes import agent, health, orchestrate, rag, safety, tools, trace

app = FastAPI(
    title="Project 4 — Agentic AI Service (Checkpoint 4: Orchestration)",
    version="0.4.0",
    description=(
        "Async FastAPI agentic service. Tool calling, agent loops, RAG-backed "
        "knowledge_search, multi-agent orchestration with structured planning, "
        "and end-to-end tracing."
    ),
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(tools.router)
app.include_router(agent.router)
app.include_router(safety.router)
app.include_router(orchestrate.router)
app.include_router(trace.router)
app.include_router(rag.router)
