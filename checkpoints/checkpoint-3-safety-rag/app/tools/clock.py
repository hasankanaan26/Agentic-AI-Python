"""Clock — returns the current date/time."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class ClockTool(BaseTool):
    """Returns the local current date/time — useful when the LLM needs ``today``."""

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
        """Return the current local date and/or time as a formatted string."""
        # ``datetime.now()`` uses the host's local timezone; this is fine for
        # the demo but a multi-region deployment should pass a tz explicitly.
        now = datetime.now()
        if format == "date":
            return ToolResult.ok(f"Current date: {now.strftime('%Y-%m-%d')}")
        if format == "time":
            return ToolResult.ok(f"Current time: {now.strftime('%H:%M:%S')}")
        return ToolResult.ok(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
