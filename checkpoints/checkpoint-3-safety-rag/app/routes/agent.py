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
    """Run the LangGraph ReAct agent with optional tool gating + approval."""
    if settings.enable_injection_detection:
        check = check_prompt_injection(request.goal)
        if check["flagged"]:
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


@router.post("/approve", response_model=LangGraphAgentResponse)
async def agent_approve(
    action: ApprovalAction,
    runner: Annotated[LangGraphAgentRunner, Depends(get_langgraph_runner)],
) -> LangGraphAgentResponse:
    """Resume (approve) or abort (reject) a paused LangGraph thread.

    On approve, the runner rebuilds the graph without ``interrupt_before``
    and calls ``ainvoke(None, ...)`` — LangGraph picks up from the
    checkpoint and runs the previously proposed tool call to completion.
    On reject, the pending record is dropped and a ``status="rejected"``
    response is returned without invoking any tool.
    """
    try:
        result = await runner.resume(
            thread_id=action.thread_id, approved=action.approved
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Thread not found",
                "thread_id": action.thread_id,
                "hint": "The thread is not in a paused state, has already been resumed, or never existed.",
            },
        )
    return LangGraphAgentResponse(**result)


@router.get("/pending")
async def agent_pending(
    runner: Annotated[LangGraphAgentRunner, Depends(get_langgraph_runner)],
) -> dict:
    """List threads currently awaiting human approval."""
    return {"pending": runner.pending_threads()}


@router.get("/thread/{thread_id}")
async def agent_thread(
    thread_id: str,
    runner: Annotated[LangGraphAgentRunner, Depends(get_langgraph_runner)],
) -> dict:
    """Inspect the full LangGraph state and checkpoint history for a thread.

    Returns the current snapshot plus every prior checkpoint, with messages,
    writes, and ``next`` decoded for the UI. Useful for explaining how the
    agent's state evolves across each tool call.
    """
    try:
        return await runner.get_thread(thread_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"error": "Thread not found", "thread_id": thread_id},
        )
