"""Calculator — the simplest tool. Pure async function, no I/O.

Useful as a teaching example: it shows the full tool contract (schema +
async `run` returning a `ToolResult`) without any networking, retries, or
state to distract from the basic shape.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """Performs one of four basic arithmetic operations on two numbers.

    Demonstrates the minimal tool: a JSON schema describing inputs, plus
    an async `run` that branches on the operation and returns a
    `ToolResult`. Division by zero is treated as an *expected* failure
    and reported via `ToolResult.fail` rather than raising.
    """

    name: ClassVar[str] = "calculator"
    permission: ClassVar[str] = "read"
    # JSON-schema description handed to the LLM. The `enum` on `operation`
    # constrains the model so it can't invent operations we don't support.
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
        """Compute ``a <operation> b`` and return the formatted result.

        Args:
            operation: One of "add", "subtract", "multiply", "divide".
            a: Left-hand operand.
            b: Right-hand operand.

        Returns:
            `ToolResult.ok` with a human-readable equation on success;
            `ToolResult.fail` for division-by-zero or unknown operations.
        """
        # Guard divide-by-zero before the match below so we don't raise
        # ZeroDivisionError — the LLM gets a clean error string instead.
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
                # Defence-in-depth: the schema's `enum` should prevent
                # unknown operations, but we don't trust the LLM to obey.
                return ToolResult.fail(f"Unknown operation '{operation}'.")
        return ToolResult.ok(f"{a} {operation} {b} = {value}")
