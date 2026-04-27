"""Clock — returns the current date/time.

A tiny example of a stateless tool that nonetheless reads from "the
outside world" (the system clock). Useful for showing how the LLM can
defer time-of-day questions to a tool rather than guessing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class ClockTool(BaseTool):
    """Returns the current local date, time, or both."""

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
        """Return the current wall-clock time in the requested format."""
        now = datetime.now()
        if format == "date":
            return ToolResult.ok(f"Current date: {now.strftime('%Y-%m-%d')}")
        if format == "time":
            return ToolResult.ok(f"Current time: {now.strftime('%H:%M:%S')}")
        # Default branch covers both "both" and any unexpected value.
        return ToolResult.ok(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
