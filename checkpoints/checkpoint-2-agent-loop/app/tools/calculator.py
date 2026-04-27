"""Calculator — the simplest tool. Pure async function, no I/O.

Useful as a "hello world" tool when learning how the agent loop wires
tool calls. Because it doesn't touch the network or the filesystem, it
also serves as a reliable smoke test that tool dispatch works.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """Performs the four basic arithmetic operations on two numbers."""

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
        """Compute ``a <operation> b`` and return the result as a ToolResult."""
        # Guard the only "expected" failure mode before dispatching.
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
                # Belt-and-braces: the schema's enum should already prevent this.
                return ToolResult.fail(f"Unknown operation '{operation}'.")
        return ToolResult.ok(f"{a} {operation} {b} = {value}")
