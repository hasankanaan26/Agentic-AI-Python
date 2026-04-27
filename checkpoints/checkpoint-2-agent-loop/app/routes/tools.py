"""Single-step tool calling — useful for inspecting LLM tool decisions.

Unlike ``/agent/run``, these endpoints do NOT loop. ``/tools/call`` makes
exactly one LLM round-trip and, if a tool was selected, executes only
the first call. This makes the route ideal for debugging the LLM's
tool-routing behaviour in isolation.
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
    """Return all registered tool definitions plus a count."""
    defs = registry.definitions()
    return {"tools": defs, "count": len(defs)}


@router.post("/call", response_model=ToolCallResponse)
async def call_with_tools(
    request: ToolCallRequest,
    llm: Annotated[LLMService, Depends(get_llm)],
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> ToolCallResponse:
    """Make one LLM call with the full tool catalogue and return the outcome.

    If the LLM chose to call a tool, only the FIRST proposed call is
    executed. If it chose to answer directly, that text is returned in
    ``llm_response``.
    """
    response = await llm.call_with_tools(
        messages=[{"role": "user", "content": request.message}],
        tools=registry.definitions(),
        system_prompt=TOOL_AGENT_PROMPT,
    )

    # Tool-call branch: execute only the first proposed call. This route is
    # intentionally single-shot; multi-call orchestration lives in /agent/run.
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

    # Direct-answer branch: model didn't need a tool, hand the text back.
    return ToolCallResponse(
        message=request.message,
        tool_called=False,
        tool_call=None,
        tool_result=None,
        tool_status=None,
        llm_response=response["response_text"],
        model=llm.model_name,
    )
