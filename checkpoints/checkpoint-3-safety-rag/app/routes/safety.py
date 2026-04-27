"""Safety inspection endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_registry
from app.models import PromptCheckRequest, SafetyCheckResult
from app.services.safety import check_prompt_injection
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/safety", tags=["safety"])


@router.post("/check-prompt", response_model=SafetyCheckResult)
async def check_prompt(request: PromptCheckRequest) -> SafetyCheckResult:
    """Run the heuristic prompt-injection scanner against ``request.text``."""
    return SafetyCheckResult(**check_prompt_injection(request.text))


@router.get("/permissions")
async def list_permissions(
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> dict:
    """Return the read/write classification for every registered tool."""
    return {
        "permissions": registry.permissions(),
        "description": {
            "read": "Tool only reads data — safe to execute without approval",
            "write": "Tool modifies state — may require human approval",
        },
    }
