"""Liveness check.

Returns a small JSON payload that also confirms which provider/model
the running process is configured against — handy when staging multiple
deployments side by side.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_settings_dep
from app.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(settings: Annotated[Settings, Depends(get_settings_dep)]) -> dict:
    """Return a static "healthy" payload plus the active provider/model."""
    return {
        "status": "healthy",
        "checkpoint": "2-agent-loop",
        "provider": settings.llm_provider,
        "model": settings.model_name(),
    }
