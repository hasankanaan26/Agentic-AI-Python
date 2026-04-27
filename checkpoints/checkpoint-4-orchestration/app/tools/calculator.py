"""Calculator — the simplest tool. Pure async function, no I/O."""

from __future__ import annotations

from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """Two-operand arithmetic tool. Read-only -- never mutates state."""

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
        """Compute ``a operation b`` and return the result as text.

        Args:
            operation: One of ``add``, ``subtract``, ``multiply``, ``divide``.
            a: Left operand.
            b: Right operand.

        Returns:
            ``ToolResult.ok`` with the formatted answer, or ``ToolResult.fail``
            on division by zero / unknown operation.
        """
        # Guard against the one classic ValueError before dispatching.
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
                # Defensive: the JSON schema enum should keep us out of here,
                # but fail gracefully if the LLM ignores it.
                return ToolResult.fail(f"Unknown operation '{operation}'.")
        return ToolResult.ok(f"{a} {operation} {b} = {value}")
