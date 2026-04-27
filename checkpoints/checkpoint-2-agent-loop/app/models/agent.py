"""Pydantic models for the from-scratch agent loop endpoint.

These define the wire contract for ``POST /agent/run``: what the client
must send (:class:`AgentRequest`), what each iteration looks like in the
response (:class:`AgentStep`), and the overall response envelope
(:class:`AgentResponse`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """One think -> act -> observe iteration in an agent run.

    Captured for every tool invocation so the caller (and the trace UI)
    can replay exactly what the agent did.
    """

    step: int = Field(..., description="1-indexed iteration number within the agent loop.")
    tool_name: str = Field(..., description="Name of the tool the LLM chose to call.")
    tool_input: dict[str, Any] = Field(..., description="Arguments passed to the tool.")
    tool_output: str = Field(..., description="Stringified tool output fed back to the LLM.")
    tool_status: str = Field(
        default="ok",
        description="Either 'ok' or 'error'; mirrors ToolResult.status.",
    )


class AgentRequest(BaseModel):
    """Inbound payload for ``POST /agent/run``."""

    goal: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language objective the agent should accomplish.",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Hard cap on think/act iterations; bounds runtime cost.",
    )


class AgentResponse(BaseModel):
    """Outbound payload returned by ``POST /agent/run``."""

    goal: str = Field(..., description="Echo of the original goal the agent was given.")
    steps: list[AgentStep] = Field(..., description="Every tool invocation, in order.")
    final_answer: str | None = Field(
        ...,
        description="The LLM's final response, or a fallback if max_steps was hit.",
    )
    steps_completed: int = Field(..., description="Number of tool steps actually executed.")
    model: str = Field(..., description="Identifier of the LLM that produced the answer.")
