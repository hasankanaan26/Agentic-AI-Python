"""Dependency providers.

Engineering standard: handlers declare what they need with `Depends(...)`.
This is the FastAPI-native way to inject testable, lifecycle-managed
objects. In tests, override with `app.dependency_overrides[...] = ...`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agents.langgraph import LangGraphAgentRunner
from app.lifespan import AppState
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.orchestrator import OrchestratorService
from app.services.tracer import TraceStore
from app.services.vector_store import VectorStore
from app.settings import Settings
from app.tools.registry import ToolRegistry


def get_app_state(request: Request) -> AppState:
    """Pull the lifespan-built :class:`AppState` off the FastAPI app.

    Every other ``get_*`` dependency below chains off this one so they
    share the same singletons inside a request.
    """
    return request.app.state.app_state


def get_settings_dep(state: Annotated[AppState, Depends(get_app_state)]) -> Settings:
    """Return the application :class:`Settings` instance."""
    return state.settings


def get_llm(state: Annotated[AppState, Depends(get_app_state)]) -> LLMService:
    """Return the singleton :class:`LLMService`."""
    return state.llm


def get_embeddings(state: Annotated[AppState, Depends(get_app_state)]) -> EmbeddingService:
    """Return the singleton :class:`EmbeddingService`."""
    return state.embeddings


def get_vector_store(state: Annotated[AppState, Depends(get_app_state)]) -> VectorStore:
    """Return the singleton :class:`VectorStore` (Chroma wrapper)."""
    return state.vector_store


def get_registry(state: Annotated[AppState, Depends(get_app_state)]) -> ToolRegistry:
    """Return the singleton :class:`ToolRegistry`."""
    return state.registry


def get_langgraph_runner(
    state: Annotated[AppState, Depends(get_app_state)],
) -> LangGraphAgentRunner:
    """Return the singleton :class:`LangGraphAgentRunner`."""
    return state.langgraph_runner


def get_orchestrator(
    state: Annotated[AppState, Depends(get_app_state)],
) -> OrchestratorService:
    """Return the singleton :class:`OrchestratorService`."""
    return state.orchestrator


def get_traces(state: Annotated[AppState, Depends(get_app_state)]) -> TraceStore:
    """Return the singleton in-memory :class:`TraceStore`."""
    return state.traces
