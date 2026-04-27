"""Build LangChain async tools from the registry.

LangChain's `StructuredTool.from_function(coroutine=...)` wraps an async
function as a tool LangGraph can `ainvoke`. We funnel every call through
`registry.execute(name, args)` so the SAME execution path is used by:
  - the raw agent loop,
  - the LangGraph ReAct agent,
  - the orchestrator's executor,
  - the LangGraph Studio entry point.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from app.tools.registry import ToolRegistry
from app.tools.schemas import CalculatorInput, ClockInput, EmployeeLookupInput

# Per-tool args_schema. Tools not listed here pass through with no schema —
# LangChain will accept any JSON the model produces.
_ARGS_SCHEMAS = {
    "calculator": CalculatorInput,
    "clock": ClockInput,
    "employee_lookup": EmployeeLookupInput,
}


def wrap_registry_tool(
    registry: ToolRegistry,
    name: str,
    *,
    args_schema=None,
    description: str | None = None,
) -> StructuredTool:
    """Build one :class:`StructuredTool` that proxies into the registry.

    Args:
        registry: The :class:`ToolRegistry` that owns the actual tool.
        name: Name of the tool to wrap (must exist in the registry).
        args_schema: Optional Pydantic model for input validation.
        description: Override description; defaults to the tool's own.

    Returns:
        A LangChain ``StructuredTool`` whose coroutine forwards to
        ``registry.execute(name, kwargs)``.
    """

    async def _runner(**kwargs):
        # Forward through the registry so the same execute path is used by
        # the raw loop, the LangGraph agent, and the orchestrator.
        result = await registry.execute(name, kwargs)
        return result.output

    # Pull description straight from the JSON-schema definition by default.
    tool_def = next(t.definition for t in registry.all() if t.name == name)
    return StructuredTool.from_function(
        coroutine=_runner,
        name=name,
        description=description or tool_def["description"],
        args_schema=args_schema,
    )


def build_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    """Wrap every tool in the registry. Order follows the registry's order."""
    return [
        wrap_registry_tool(registry, t.name, args_schema=_ARGS_SCHEMAS.get(t.name))
        for t in registry.all()
    ]
