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
    """JSON-schema definition the LLM sees.

    The shape mirrors the OpenAI/Gemini function-calling specs so a
    `ToolDefinition` can be serialised and handed straight to either
    provider with minimal massaging.
    """

    name: str = Field(..., description="Stable identifier the LLM uses to invoke this tool.")
    description: str = Field(
        ..., description="Natural-language explanation of what the tool does and when to use it."
    )
    parameters: dict[str, Any] = Field(
        ..., description="JSON Schema (object) describing the tool's input arguments."
    )


class ToolResult(BaseModel):
    """Outcome of executing one tool call.

    Tools never raise on expected failures — they return
    `ToolResult.fail(...)`. This keeps the agent loop crash-free and lets
    the LLM reason about errors as just another piece of evidence.
    """

    status: Literal["ok", "error"] = Field(
        default="ok", description="Whether the tool call succeeded or produced a structured error."
    )
    output: str = Field(
        default="",
        description="Human-readable result string fed back to the LLM as the tool's reply.",
    )
    error: str | None = Field(
        default=None,
        description="Error message when status is 'error'; None on success.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary tool-specific metadata (timings, source ids, etc.).",
    )

    @classmethod
    def ok(cls, output: str, **metadata: Any) -> ToolResult:
        """Construct a successful result with optional metadata kwargs."""
        return cls(status="ok", output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        """Construct a structured failure result.

        The `output` field is set to a short user-readable string prefixed
        with ``Tool error:`` so that even if a caller forgets to inspect
        `status`, the error is visible in the conversation.
        """
        return cls(status="error", output=f"Tool error: {error}", error=error, metadata=metadata)


class ToolCallInfo(BaseModel):
    """Description of a single tool invocation chosen by the LLM."""

    name: str = Field(..., description="Name of the tool the LLM asked to call.")
    arguments: dict[str, Any] = Field(
        ..., description="Arguments the LLM supplied for the call (already JSON-decoded)."
    )


class ToolCallRequest(BaseModel):
    """Inbound request body for the `/tools/call` endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        description="The user's natural-language message that the LLM will reason over.",
    )


class ToolCallResponse(BaseModel):
    """Outbound response body for the `/tools/call` endpoint.

    Either `tool_call`/`tool_result` are populated (the LLM chose a tool)
    OR `llm_response` is populated (the LLM answered directly), but not
    both. The `tool_called` boolean disambiguates without inspecting nulls.
    """

    message: str = Field(..., description="Echo of the original user message.")
    tool_called: bool = Field(
        ..., description="True if the LLM decided to call a tool; False if it answered directly."
    )
    tool_call: ToolCallInfo | None = Field(
        default=None, description="Details of the chosen tool invocation, when one occurred."
    )
    tool_result: str | None = Field(
        default=None, description="The tool's textual output, when a tool was called."
    )
    tool_status: str | None = Field(
        default=None, description="Status of the tool execution ('ok' or 'error')."
    )
    llm_response: str | None = Field(
        default=None, description="Direct LLM answer when no tool was called."
    )
    model: str = Field(..., description="Identifier of the LLM model that produced the response.")
