"""Single-step tool calling — useful for inspecting LLM tool decisions.

This module exposes two endpoints:
  * ``GET  /tools/list`` — return the registered tool definitions.
  * ``POST /tools/call`` — feed a user message to the LLM and, if the
    model chooses a tool, run that single tool and return both the
    decision and the tool's output.

The endpoint deliberately stops after one tool call: it's the simplest
slice of the agent loop and makes it easy to observe what the LLM
decides to do given a particular message + tool catalogue. Multi-step
agents arrive in later checkpoints.
"""

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
    """List the JSON-schema tool definitions currently registered.

    Args:
        registry: Injected `ToolRegistry` (one per process).

    Returns:
        Dict with the tool definitions and a count for convenience.
    """
    defs = registry.definitions()
    return {"tools": defs, "count": len(defs)}


@router.post("/call", response_model=ToolCallResponse)
async def call_with_tools(
    request: ToolCallRequest,
    llm: Annotated[LLMService, Depends(get_llm)],
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> ToolCallResponse:
    """Run one round of LLM-driven tool calling.

    Hands the user message to the LLM along with the available tools.
    If the LLM chooses a tool, that tool is executed and its output is
    returned alongside the chosen call. Otherwise the LLM's direct reply
    is returned.

    Args:
        request: Validated `ToolCallRequest` body.
        llm: Injected shared `LLMService`.
        registry: Injected shared `ToolRegistry`.

    Returns:
        A `ToolCallResponse` describing either the tool decision +
        result, or the LLM's direct answer.
    """
    # Step 1: ask the LLM what to do. The provider returns either a list
    # of `tool_calls` or a free-form `response_text` — never both.
    response = await llm.call_with_tools(
        messages=[{"role": "user", "content": request.message}],
        tools=registry.definitions(),
        system_prompt=TOOL_AGENT_PROMPT,
    )

    if response["tool_calls"]:
        # Single-step: only the first tool call is honored. Parallel /
        # multi-step tool calling is intentionally deferred to later CPs.
        first = response["tool_calls"][0]
        # `registry.execute` is guaranteed to return a ToolResult, even
        # if the tool blew up — so we can assemble the response without
        # extra try/except plumbing here.
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

    # No tool was chosen — return the model's plain-text answer.
    return ToolCallResponse(
        message=request.message,
        tool_called=False,
        tool_call=None,
        tool_result=None,
        tool_status=None,
        llm_response=response["response_text"],
        model=llm.model_name,
    )
