"""Single-step tool calling — useful for inspecting LLM tool decisions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_llm, get_registry
from app.models import ToolCallInfo, ToolCallRequest, ToolCallResponse
from app.prompts import TOOL_AGENT_PROMPT
from app.services.llm import LLMService
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/list")
async def list_tools(
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> dict:
    """Return every tool the agent has access to, with its JSON schema."""
    defs = registry.definitions()
    return {"tools": defs, "count": len(defs)}


@router.post("/call", response_model=ToolCallResponse)
async def call_with_tools(
    request: ToolCallRequest,
    llm: Annotated[LLMService, Depends(get_llm)],
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> ToolCallResponse:
    """Single-turn tool call: ask the LLM, run the chosen tool, return the result.

    Useful for inspecting the LLM's tool-selection behaviour in isolation
    without the multi-step agent loop machinery.
    """
    response = await llm.call_with_tools(
        messages=[{"role": "user", "content": request.message}],
        tools=registry.definitions(),
        system_prompt=TOOL_AGENT_PROMPT,
    )

    # If the LLM picked a tool, execute the FIRST one (single-shot endpoint).
    if response["tool_calls"]:
        first = response["tool_calls"][0]
        result = await registry.execute(first["name"], first["arguments"])
        return ToolCallResponse(
            message=request.message,
            tool_called=True,
            tool_call=ToolCallInfo(name=first["name"], arguments=first["arguments"]),
            tool_result=result.output,
            tool_status=result.status,
            llm_response=None,
            model=llm.model_name,
        )

    # No tool needed -- return the LLM's plain answer.
    return ToolCallResponse(
        message=request.message,
        tool_called=False,
        tool_call=None,
        tool_result=None,
        tool_status=None,
        llm_response=response["response_text"],
        model=llm.model_name,
    )
