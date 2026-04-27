"""Tool base class.

Every tool is:
  1. a JSON-schema definition the LLM sees,
  2. an async `run()` that returns a `ToolResult`.

`ToolResult` is the contract: tools never raise on expected failures —
they return `ToolResult.fail(...)` so the agent loop can feed the
structured error back to the LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from app.models.tool import ToolResult


class BaseTool(ABC):
    """Abstract base class for all tools the agent can invoke.

    Subclasses MUST set the class-level ``name`` and ``definition``
    attributes and implement the async ``run`` method. ``permission``
    defaults to ``"read"``; tools that mutate external state should set it
    to ``"write"`` so future safety layers can gate them.
    """

    # Stable identifier used by the LLM to call this tool.
    name: ClassVar[str]
    # "read" tools can run without confirmation; "write" tools may be
    # gated behind approval in later checkpoints.
    permission: ClassVar[str] = "read"  # "read" | "write"
    # JSON-schema-shaped description handed to the LLM.
    definition: ClassVar[dict[str, Any]]

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Run the tool. Return a ToolResult — never raise on expected failure.

        Args:
            **kwargs: Tool-specific arguments matching the JSON schema in
                ``definition.parameters``.

        Returns:
            A `ToolResult`. Use `ToolResult.fail(...)` for expected
            failures (bad inputs, upstream API errors) so the agent loop
            can surface the error to the LLM as context.
        """
