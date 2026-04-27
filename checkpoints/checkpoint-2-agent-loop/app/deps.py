"""Dependency providers (CP2 — LLM + registry only).

Each function below is a FastAPI dependency that reaches into ``app.state``
and pulls out one of the singletons set up in ``lifespan``. Routes use
``Annotated[..., Depends(get_xxx)]`` rather than touching ``request.app.state``
directly so individual deps can be overridden in tests via
``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.lifespan import AppState
from app.services.llm import LLMService
from app.settings import Settings
from app.tools.registry import ToolRegistry


def get_app_state(request: Request) -> AppState:
    """Return the ``AppState`` bundle attached to the running FastAPI app."""
    return request.app.state.app_state


def get_settings_dep(state: Annotated[AppState, Depends(get_app_state)]) -> Settings:
    """Return the loaded :class:`Settings` singleton."""
    return state.settings


def get_llm(state: Annotated[AppState, Depends(get_app_state)]) -> LLMService:
    """Return the shared :class:`LLMService` instance."""
    return state.llm


def get_registry(state: Annotated[AppState, Depends(get_app_state)]) -> ToolRegistry:
    """Return the shared :class:`ToolRegistry` instance."""
    return state.registry
