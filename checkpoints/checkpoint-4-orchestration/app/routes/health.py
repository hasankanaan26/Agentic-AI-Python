"""Liveness check."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_settings_dep
from app.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(settings: Annotated[Settings, Depends(get_settings_dep)]) -> dict:
    """Lightweight liveness probe used by load balancers / smoke tests."""
    return {
        "status": "healthy",
        "checkpoint": "4-orchestration",
        "provider": settings.llm_provider,
        "model": settings.model_name(),
    }
