"""Liveness check.

Exposes a single `GET /health` endpoint that confirms the app booted and
reports which LLM provider/model is configured. Useful for smoke tests
and container orchestrators (Kubernetes, ECS) that need a cheap probe.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_settings_dep
from app.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(settings: Annotated[Settings, Depends(get_settings_dep)]) -> dict:
    """Return a small JSON payload confirming the service is alive.

    Args:
        settings: Injected validated `Settings`. The endpoint touches it
            so a misconfigured app surfaces here rather than on the first
            real request.

    Returns:
        A dict with status, checkpoint name, and LLM provider/model.
    """
    return {
        "status": "healthy",
        "checkpoint": "1-tool-calling",
        "provider": settings.llm_provider,
        "model": settings.model_name(),
    }
