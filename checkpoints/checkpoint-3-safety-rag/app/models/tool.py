"""Tool I/O contracts.

The `ToolResult` is the return type EVERY tool produces. Failed tool
runs return `ToolResult(status="error", error=...)` — they do not raise.
That's the engineering standard: the agent loop should never crash because
a single tool's external API hiccupped; the LLM should see a structured
error message and decide what to do next.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """JSON-schema definition the LLM sees as a callable function."""

    name: str = Field(description="Stable tool identifier used in tool_calls.")
    description: str = Field(description="Human-readable summary shown to the LLM.")
    parameters: dict[str, Any] = Field(description="JSON-schema describing accepted arguments.")


class ToolResult(BaseModel):
    """Outcome of executing one tool call.

    Tools should always return one of these — never raise on expected errors —
    so the agent loop can feed the structured failure back to the LLM.
    """

    status: Literal["ok", "error"] = Field(
        default="ok", description="'ok' on success, 'error' on a recoverable failure."
    )
    output: str = Field(default="", description="String returned to the LLM as the observation.")
    error: str | None = Field(
        default=None, description="Original error message when status == 'error'."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form bag (cache hit flags, hit counts, etc.) for traces.",
    )

    @classmethod
    def ok(cls, output: str, **metadata: Any) -> ToolResult:
        """Build a successful result with the given output and optional metadata."""
        return cls(status="ok", output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        """Build a failure result; ``output`` is auto-prefixed for LLM clarity."""
        return cls(status="error", output=f"Tool error: {error}", error=error, metadata=metadata)


class ToolCallInfo(BaseModel):
    """A single tool call the LLM decided to make in ``/tools/call``."""

    name: str = Field(description="Tool identifier the LLM invoked.")
    arguments: dict[str, Any] = Field(description="Parsed argument dict for the tool.")


class ToolCallRequest(BaseModel):
    """Body for ``POST /tools/call`` — single-step tool invocation."""

    message: str = Field(..., min_length=1, description="User message the LLM should react to.")


class ToolCallResponse(BaseModel):
    """Response from ``POST /tools/call``; either tool output or a direct answer."""

    message: str = Field(description="Echo of the user message for trace correlation.")
    tool_called: bool = Field(description="True when the LLM chose to invoke a tool.")
    tool_call: ToolCallInfo | None = Field(
        default=None, description="Details of the chosen tool call (only when tool_called is True)."
    )
    tool_result: str | None = Field(
        default=None, description="String output the tool produced (only when tool_called is True)."
    )
    tool_status: str | None = Field(
        default=None, description="Status of the tool execution: 'ok' or 'error'."
    )
    llm_response: str | None = Field(
        default=None, description="Direct LLM answer when no tool was called."
    )
    model: str = Field(description="Model identifier that produced the response.")
