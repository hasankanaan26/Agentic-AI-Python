"""FastAPI dependency providers wiring singletons to request handlers.

Every ``get_*`` function is a thin accessor that pulls one field off
``app.state.app_state`` (the dataclass built in ``app.lifespan``). Routes
declare the dependencies they need with ``Annotated[..., Depends(get_x)]``
and FastAPI resolves them per-request — no global state, no module-level
clients, easy to swap with ``app.dependency_overrides`` in tests.

CP3 — no orchestrator/traces yet (those land in CP4).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agents.langgraph import LangGraphAgentRunner
from app.lifespan import AppState
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import VectorStore
from app.settings import Settings
from app.tools.registry import ToolRegistry


def get_app_state(request: Request) -> AppState:
    """Return the process-wide ``AppState`` attached during lifespan startup."""
    return request.app.state.app_state


def get_settings_dep(state: Annotated[AppState, Depends(get_app_state)]) -> Settings:
    """Return the cached ``Settings`` instance."""
    return state.settings


def get_llm(state: Annotated[AppState, Depends(get_app_state)]) -> LLMService:
    """Return the singleton async LLM client."""
    return state.llm


def get_embeddings(state: Annotated[AppState, Depends(get_app_state)]) -> EmbeddingService:
    """Return the singleton async embedding client."""
    return state.embeddings


def get_vector_store(state: Annotated[AppState, Depends(get_app_state)]) -> VectorStore:
    """Return the persistent Chroma-backed vector store."""
    return state.vector_store


def get_registry(state: Annotated[AppState, Depends(get_app_state)]) -> ToolRegistry:
    """Return the fully-built tool registry shared by both agent runners."""
    return state.registry


def get_langgraph_runner(
    state: Annotated[AppState, Depends(get_app_state)],
) -> LangGraphAgentRunner:
    """Return the LangGraph ReAct agent runner singleton."""
    return state.langgraph_runner
