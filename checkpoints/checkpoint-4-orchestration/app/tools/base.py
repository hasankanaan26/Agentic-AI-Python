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
    """Abstract base class every concrete tool inherits from.

    Subclasses must define three class-level attributes (``name``,
    ``permission``, ``definition``) and implement the async ``run`` method.
    The contract: ``run`` MUST return a :class:`ToolResult`; never raise on
    expected failures (bad arguments, division by zero, no matches, etc).
    Unexpected exceptions are caught one level up by the registry.
    """

    name: ClassVar[str]
    permission: ClassVar[str] = "read"  # "read" | "write" -- write tools may require approval.
    definition: ClassVar[dict[str, Any]]

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with LLM-supplied arguments.

        Args:
            **kwargs: Keyword arguments matching the tool's JSON-schema parameters.

        Returns:
            A :class:`ToolResult` with ``status`` ``"ok"`` or ``"error"``.
        """
