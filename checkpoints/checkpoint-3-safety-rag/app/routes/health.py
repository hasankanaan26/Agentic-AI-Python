"""Liveness check."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_settings_dep
from app.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(settings: Annotated[Settings, Depends(get_settings_dep)]) -> dict:
    """Cheap liveness probe — also surfaces the active provider/model."""
    return {
        "status": "healthy",
        "checkpoint": "3-safety-rag",
        "provider": settings.llm_provider,
        "model": settings.model_name(),
    }
