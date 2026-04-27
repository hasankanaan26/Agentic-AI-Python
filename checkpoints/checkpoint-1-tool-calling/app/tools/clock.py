"""Clock — returns the current date/time.

LLMs do not have access to wall-clock time on their own; exposing a clock
tool is a simple way to let the model answer "what's today's date?"
correctly without hallucinating.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class ClockTool(BaseTool):
    """Reports the current local date, time, or both.

    The `format` argument is optional — if omitted, the tool returns both
    the date and the time. Output is plain text so the LLM can quote it
    back to the user verbatim.
    """

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

    async def run(self, format: str = "both") -> ToolResult:  # noqa: A002
        """Return the current date and/or time as a formatted string.

        Args:
            format: One of "date", "time", or "both". Defaults to "both".

        Returns:
            A `ToolResult.ok` whose `output` contains the formatted value.
        """
        # `datetime.now()` returns local server time. For UTC use
        # `datetime.utcnow()` — kept local here for human readability.
        now = datetime.now()
        if format == "date":
            return ToolResult.ok(f"Current date: {now.strftime('%Y-%m-%d')}")
        if format == "time":
            return ToolResult.ok(f"Current time: {now.strftime('%H:%M:%S')}")
        # Default branch covers "both" and any unexpected value.
        return ToolResult.ok(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
