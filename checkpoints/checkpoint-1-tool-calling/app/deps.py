"""Dependency providers (CP1 — LLM + registry only).

These small helpers are the glue between FastAPI's `Depends` machinery
and the `AppState` constructed in `app.lifespan`. Routes write
`Depends(get_llm)` (etc.) instead of reaching into `app.state` directly —
which keeps routes ignorant of how the singletons were built and lets
tests swap in fakes via `app.dependency_overrides`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.lifespan import AppState
from app.services.llm import LLMService
from app.settings import Settings
from app.tools.registry import ToolRegistry


def get_app_state(request: Request) -> AppState:
    """Return the `AppState` attached to the FastAPI app during lifespan."""
    # `request.app` is the same FastAPI instance we configured in `main.py`;
    # the lifespan stored our singletons under `app.state.app_state`.
    return request.app.state.app_state


def get_settings_dep(state: Annotated[AppState, Depends(get_app_state)]) -> Settings:
    """DI provider for the validated `Settings` instance."""
    return state.settings


def get_llm(state: Annotated[AppState, Depends(get_app_state)]) -> LLMService:
    """DI provider for the shared async `LLMService`."""
    return state.llm


def get_registry(state: Annotated[AppState, Depends(get_app_state)]) -> ToolRegistry:
    """DI provider for the process-wide `ToolRegistry`."""
    return state.registry
