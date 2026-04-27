"""Pydantic models for the agent endpoints (raw loop + LangGraph).

Request/response shapes are kept narrow so the OpenAPI surface clearly
documents what each endpoint accepts and returns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """One think-act-observe iteration recorded in the response trace."""

    step: int = Field(description="1-indexed iteration counter.")
    tool_name: str = Field(description="Name of the tool the LLM invoked.")
    tool_input: dict[str, Any] = Field(description="Arguments passed to the tool.")
    tool_output: str = Field(description="String output the tool returned to the LLM.")
    tool_status: str = Field(default="ok", description="'ok' on success, 'error' on failure.")


class AgentRequest(BaseModel):
    """Body for the raw agent loop endpoint (``POST /agent/run-raw``)."""

    goal: str = Field(
        ..., min_length=1, max_length=4000,
        description="Natural-language goal for the agent to accomplish.",
    )
    max_steps: int = Field(
        default=10, ge=1, le=25,
        description="Hard cap on think/act iterations before the agent gives up.",
    )


class AgentResponse(BaseModel):
    """Result of an agent run — both raw loop and LangGraph variants extend this."""

    goal: str = Field(description="Echo of the original goal for trace correlation.")
    steps: list[AgentStep] = Field(description="Ordered list of tool invocations made.")
    final_answer: str | None = Field(description="Final natural-language answer, if the agent finished.")
    steps_completed: int = Field(description="Number of tool calls completed (== len(steps)).")
    model: str = Field(description="Model identifier that drove the run.")


class SafeAgentRequest(AgentRequest):
    """Body for ``POST /agent/run`` — adds tool gating and approval flags."""

    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional allow-list of tool names; ``None`` exposes all registered tools.",
    )
    require_approval: bool = Field(
        default=False,
        description="When true, the LangGraph agent pauses before each tool for human approval.",
    )


class LangGraphAgentResponse(AgentResponse):
    """Response from the LangGraph endpoint; includes thread/engine metadata."""

    thread_id: str = Field(description="Thread identifier for resuming this run via the checkpointer.")
    engine: str = Field(default="langgraph", description="Which agent engine produced this response.")


class ApprovalAction(BaseModel):
    """Body for ``POST /agent/approve`` — placeholder for the resume API."""

    thread_id: str = Field(description="Thread to resume or reject.")
    approved: bool = Field(description="True to continue execution, False to abort.")
