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

    Mirrors the OpenAI / Gemini ``function_declarations`` shape so the same
    dict can be passed to either provider without translation.
    """

    name: str = Field(..., description="Stable tool identifier the LLM emits when calling.")
    description: str = Field(..., description="Natural-language description shown to the LLM.")
    parameters: dict[str, Any] = Field(
        ..., description="JSON Schema describing the tool's accepted arguments."
    )


class ToolResult(BaseModel):
    """Outcome of executing one tool call.

    Tools never raise on expected failures -- they return a ``ToolResult``
    with ``status="error"`` so the LLM can read the error and decide what to
    do next. Use the :meth:`ok` / :meth:`fail` constructors instead of
    instantiating directly to keep that contract uniform.
    """

    status: Literal["ok", "error"] = Field(
        default="ok", description="``ok`` for success, ``error`` for an expected failure."
    )
    output: str = Field(default="", description="Human-readable output sent back to the LLM.")
    error: str | None = Field(
        default=None, description="Short error message; only set when ``status == 'error'``."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (e.g. cache hits, match counts) for observability.",
    )

    @classmethod
    def ok(cls, output: str, **metadata: Any) -> ToolResult:
        """Construct a successful ``ToolResult`` carrying ``output`` and metadata."""
        return cls(status="ok", output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        """Construct a failure result with a structured error suitable for the LLM."""
        # Note: we still populate ``output`` so simple LLM prompts that read
        # only ``output`` still see the failure reason.
        return cls(status="error", output=f"Tool error: {error}", error=error, metadata=metadata)


class ToolCallInfo(BaseModel):
    """Lightweight record of a single tool invocation."""

    name: str = Field(..., description="Name of the tool the LLM chose to call.")
    arguments: dict[str, Any] = Field(
        ..., description="Arguments the LLM produced (validated by the tool itself)."
    )


class ToolCallRequest(BaseModel):
    """Body for ``POST /tools/call`` -- a single user message."""

    message: str = Field(..., min_length=1, description="User message routed to the LLM.")


class ToolCallResponse(BaseModel):
    """Response from ``POST /tools/call``.

    Either ``tool_called`` is true (then ``tool_call`` / ``tool_result`` /
    ``tool_status`` are populated) or it is false (then ``llm_response`` is
    populated). They are never both populated.
    """

    message: str = Field(..., description="Echo of the user's input message.")
    tool_called: bool = Field(..., description="``True`` if the LLM chose to invoke a tool.")
    tool_call: ToolCallInfo | None = Field(
        default=None, description="The tool name + arguments, when ``tool_called`` is true."
    )
    tool_result: str | None = Field(
        default=None, description="String output from the tool, when one was called."
    )
    tool_status: str | None = Field(
        default=None, description="``ok`` or ``error`` from the executed tool."
    )
    llm_response: str | None = Field(
        default=None, description="LLM's plain-text answer when no tool was invoked."
    )
    model: str = Field(..., description="Identifier of the model that produced the response.")
