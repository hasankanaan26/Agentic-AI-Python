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

    Tools advertise themselves through this struct. The ``parameters``
    field follows the OpenAI/Gemini function-calling JSON-schema dialect.
    """

    name: str = Field(..., description="Unique identifier the LLM uses to invoke the tool.")
    description: str = Field(..., description="Human-readable description shown to the LLM.")
    parameters: dict[str, Any] = Field(..., description="JSON-schema for the tool's arguments.")


class ToolResult(BaseModel):
    """Outcome of executing one tool call.

    Tools NEVER raise on expected failure — they construct
    ``ToolResult.fail(...)`` so the agent loop can feed the structured
    error back to the LLM and let it recover.
    """

    status: Literal["ok", "error"] = Field(
        default="ok",
        description="Discriminator; 'error' means the call failed semantically.",
    )
    output: str = Field(default="", description="Stringified result fed back to the LLM.")
    error: str | None = Field(default=None, description="Error message when status='error'.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata (cache hits, counts, ...).",
    )

    @classmethod
    def ok(cls, output: str, **metadata: Any) -> ToolResult:
        """Construct a successful result.

        Args:
            output: User-facing string returned from the tool.
            **metadata: Arbitrary keyword args attached as structured metadata.
        """
        return cls(status="ok", output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        """Construct a failure result with a uniform output prefix.

        Args:
            error: Short human-readable error message.
            **metadata: Arbitrary keyword args attached as structured metadata.
        """
        return cls(status="error", output=f"Tool error: {error}", error=error, metadata=metadata)


class ToolCallInfo(BaseModel):
    """Echo of which tool the LLM chose plus the arguments it supplied."""

    name: str = Field(..., description="Name of the tool the LLM selected.")
    arguments: dict[str, Any] = Field(..., description="Arguments passed to the tool.")


class ToolCallRequest(BaseModel):
    """Inbound payload for the single-shot ``POST /tools/call`` endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        description="The user's message; the LLM may decide to invoke a tool to answer it.",
    )


class ToolCallResponse(BaseModel):
    """Outbound payload from ``POST /tools/call``.

    Two shapes share this struct:

    * Tool was called: ``tool_called=True`` and the ``tool_*`` fields are populated.
    * No tool was needed: ``tool_called=False`` and ``llm_response`` carries the answer.
    """

    message: str = Field(..., description="Echo of the original user message.")
    tool_called: bool = Field(..., description="True if the LLM invoked a tool.")
    tool_call: ToolCallInfo | None = Field(
        default=None,
        description="Tool name + arguments, when one was called.",
    )
    tool_result: str | None = Field(default=None, description="Tool output, when one was called.")
    tool_status: str | None = Field(
        default=None,
        description="Mirror of ToolResult.status when a tool was called.",
    )
    llm_response: str | None = Field(
        default=None,
        description="Direct LLM answer when no tool was needed.",
    )
    model: str = Field(..., description="Identifier of the LLM that produced the response.")
