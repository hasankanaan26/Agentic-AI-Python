"""Clock — returns the current date/time."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class ClockTool(BaseTool):
    """Returns local wall-clock date/time. Useful for "what's today's date" queries."""

    name: ClassVar[str] = "clock"
    permission: ClassVar[str] = "read"
    definition: ClassVar[dict[str, Any]] = {
        "name": "clock",
        "description": "Get the current date and/or time.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["date", "time", "both"],
                    "description": "What to return: 'date', 'time', or 'both'.",
                },
            },
            "required": [],
        },
    }

    async def run(self, format: str = "both") -> ToolResult:
        """Return the current date, time, or both, as a formatted string.

        Args:
            format: One of ``date``, ``time``, ``both``. Defaults to ``both``.

        Returns:
            ``ToolResult.ok`` with the current value rendered as a string.
        """
        now = datetime.now()
        if format == "date":
            return ToolResult.ok(f"Current date: {now.strftime('%Y-%m-%d')}")
        if format == "time":
            return ToolResult.ok(f"Current time: {now.strftime('%H:%M:%S')}")
        return ToolResult.ok(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
