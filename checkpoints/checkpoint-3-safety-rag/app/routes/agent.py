"""Agent endpoints: LangGraph (production) and the raw loop (comparison)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.agents.langgraph import LangGraphAgentRunner
from app.agents.raw_loop import run_agent
from app.deps import get_langgraph_runner, get_llm, get_registry, get_settings_dep
from app.models import (
    AgentRequest,
    AgentResponse,
    ApprovalAction,
    LangGraphAgentResponse,
    SafeAgentRequest,
)
from app.services.llm import LLMService
from app.services.safety import check_prompt_injection
from app.settings import Settings
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=LangGraphAgentResponse)
async def agent_run(
    request: SafeAgentRequest,
    runner: Annotated[LangGraphAgentRunner, Depends(get_langgraph_runner)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> LangGraphAgentResponse:
    """Run the LangGraph ReAct agent with optional tool gating + approval.

    When ``settings.enable_injection_detection`` is on, the request is
    rejected with HTTP 400 if the heuristic detector flags the goal — this
    runs BEFORE we spend any tokens on the agent.
    """
    if settings.enable_injection_detection:
        check = check_prompt_injection(request.goal)
        if check["flagged"]:
            # Surface the structured findings so the client can show the user
            # which heuristics tripped, not just an opaque error.
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Prompt injection detected",
                    "risk_level": check["risk_level"],
                    "findings": [f["description"] for f in check["findings"]],
                },
            )

    result = await runner.run(
        goal=request.goal,
        allowed_tools=request.allowed_tools,
        require_approval=request.require_approval,
        max_steps=request.max_steps,
    )
    return LangGraphAgentResponse(**result)


@router.post("/run-raw", response_model=AgentResponse)
async def agent_run_raw(
    request: AgentRequest,
    llm: Annotated[LLMService, Depends(get_llm)],
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> AgentResponse:
    """Run the from-scratch async agent loop (kept for comparison/teaching)."""
    result = await run_agent(
        goal=request.goal, llm=llm, registry=registry, max_steps=request.max_steps
    )
    return AgentResponse(**result)


@router.post("/approve")
async def agent_approve(action: ApprovalAction) -> dict:
    """Placeholder for human-in-the-loop approval flows.

    Real wiring would resume the LangGraph thread via the checkpointer
    when ``approved`` is true; for now this returns a stub status.
    """
    if action.approved:
        return {
            "thread_id": action.thread_id,
            "status": "approved",
            "message": f"Approval granted for thread {action.thread_id}.",
        }
    return {
        "thread_id": action.thread_id,
        "status": "rejected",
        "message": f"Approval denied for thread {action.thread_id}.",
    }
