"""Pydantic models for the agent endpoints (raw loop + LangGraph).

These shapes are shared between ``POST /agent/run`` (LangGraph),
``POST /agent/run-raw`` (the from-scratch loop), and
``POST /agent/approve`` (placeholder approval action).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """One think/act/observe step recorded by the agent."""

    step: int = Field(..., description="1-based step index in execution order.")
    tool_name: str = Field(..., description="Name of the tool invoked at this step.")
    tool_input: dict[str, Any] = Field(..., description="Arguments the LLM produced for the tool.")
    tool_output: str = Field(..., description="Stringified output the tool returned.")
    tool_status: str = Field(default="ok", description="``ok`` or ``error`` from the tool result.")


class AgentRequest(BaseModel):
    """Body for ``POST /agent/run-raw`` (and base for the safer LangGraph variant)."""

    goal: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language goal the agent should accomplish.",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Hard cap on think/act iterations before the agent gives up.",
    )


class AgentResponse(BaseModel):
    """Response shape shared by both agent endpoints."""

    goal: str = Field(..., description="Echo of the user's goal.")
    steps: list[AgentStep] = Field(..., description="Ordered tool-call trace.")
    final_answer: str | None = Field(..., description="Final natural-language answer, if produced.")
    steps_completed: int = Field(..., description="Number of tool steps actually executed.")
    model: str = Field(..., description="Identifier of the model used for this run.")


class SafeAgentRequest(AgentRequest):
    """Extended request used by the LangGraph endpoint.

    Adds optional tool whitelisting and a human-in-the-loop approval gate.
    """

    allowed_tools: list[str] | None = Field(
        default=None,
        description="If set, restrict the agent to this subset of tool names.",
    )
    require_approval: bool = Field(
        default=False,
        description="When true, pause before any write tool runs (interrupt_before='tools').",
    )


class LangGraphAgentResponse(AgentResponse):
    """Response shape from the LangGraph endpoint.

    Adds ``thread_id`` (so callers can resume) and a ``status`` that may be
    ``completed`` or ``awaiting_approval`` (in which case ``pending_tool``
    names the tool blocked on approval).
    """

    thread_id: str = Field(..., description="LangGraph checkpoint thread id; reuse to resume.")
    engine: str = Field(default="langgraph", description="Discriminator for the engine that ran.")
    status: str = Field(
        default="completed",
        description="``completed``, ``awaiting_approval``, ``rejected`` or ``error``.",
    )
    pending_tool: str | None = Field(
        default=None,
        description="Tool name the agent is waiting on approval for, if any.",
    )


class ApprovalAction(BaseModel):
    """Body for ``POST /agent/approve`` -- approve or reject a paused thread."""

    thread_id: str = Field(..., description="Thread id returned from a prior agent run.")
    approved: bool = Field(..., description="True to allow the pending tool, false to reject it.")
