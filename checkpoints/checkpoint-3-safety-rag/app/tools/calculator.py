"""Calculator — the simplest tool. Pure async function, no I/O."""

from __future__ import annotations

from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """Pure-function arithmetic tool — the simplest possible ``BaseTool`` example."""

    name: ClassVar[str] = "calculator"
    permission: ClassVar[str] = "read"
    definition: ClassVar[dict[str, Any]] = {
        "name": "calculator",
        "description": (
            "Perform basic arithmetic operations: add, subtract, multiply, or divide two numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform.",
                },
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."},
            },
            "required": ["operation", "a", "b"],
        },
    }

    async def run(self, operation: str, a: float, b: float) -> ToolResult:
        """Apply ``operation`` to ``a`` and ``b`` and format the result."""
        # Guard against the one expected runtime failure; everything else is
        # a programming error and is caught by the registry's ``except``.
        if operation == "divide" and b == 0:
            return ToolResult.fail("Division by zero is not allowed.")

        match operation:
            case "add":
                value = a + b
            case "subtract":
                value = a - b
            case "multiply":
                value = a * b
            case "divide":
                value = a / b
            case _:
                # Defensive — JSON-schema enum already restricts values, but
                # this keeps us safe against schema/code drift.
                return ToolResult.fail(f"Unknown operation '{operation}'.")
        return ToolResult.ok(f"{a} {operation} {b} = {value}")
