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
    """Abstract base class for every tool the agent can call.

    Concrete tools must define ``name``, ``definition`` (the JSON schema the
    LLM sees), and an async ``run`` method. They may optionally override
    ``permission`` to mark themselves as state-mutating (``"write"``).
    """

    #: Stable identifier the LLM uses to call the tool.
    name: ClassVar[str]
    #: ``"read"`` for safe lookups, ``"write"`` for tools that mutate state.
    #: The agent layer can interrupt before write tools run (HITL approval).
    permission: ClassVar[str] = "read"
    #: JSON-schema definition advertised to the LLM (matches ToolDefinition).
    definition: ClassVar[dict[str, Any]]

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the LLM-supplied arguments.

        Implementations MUST return a :class:`ToolResult` even on failure;
        raising propagates as an unrecoverable error and is reserved for
        truly exceptional conditions (programming bugs, OOM, etc.).
        """
