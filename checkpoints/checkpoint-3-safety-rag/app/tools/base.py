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
    """Abstract base every concrete tool inherits from.

    Subclass and implement ``name``, ``definition``, and ``run``. The
    registry uses ``name`` as the lookup key and exposes ``definition`` to
    the LLM as a function declaration. ``permission`` drives the
    ``/safety/permissions`` endpoint and lets the agent gate write tools
    behind human approval.
    """

    name: ClassVar[str]
    permission: ClassVar[str] = "read"  # "read" | "write"
    definition: ClassVar[dict[str, Any]]

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Run the tool. Return a ToolResult — never raise on expected failure."""
