"""``POST /agent/run`` — the from-scratch async agent loop.

This route is a thin shell: it validates input via :class:`AgentRequest`,
delegates to :func:`run_agent`, and re-validates the result against
:class:`AgentResponse`. All real work happens in :mod:`app.agents.raw_loop`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.agents.raw_loop import run_agent
from app.deps import get_llm, get_registry
from app.models import AgentRequest, AgentResponse
from app.services.llm import LLMService
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=AgentResponse)
async def agent_run(
    request: AgentRequest,
    llm: Annotated[LLMService, Depends(get_llm)],
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> AgentResponse:
    """Execute the agent loop for a single user goal and return the trace.

    Args:
        request: Validated payload containing the goal and step budget.
        llm: Shared LLM client (DI).
        registry: Shared tool registry (DI).

    Returns:
        :class:`AgentResponse` with the full step-by-step trace and final answer.
    """
    result = await run_agent(
        goal=request.goal, llm=llm, registry=registry, max_steps=request.max_steps
    )
    # ``run_agent`` returns a plain dict; revalidating it through Pydantic
    # both type-checks the shape and renders nicely in OpenAPI.
    return AgentResponse(**result)
