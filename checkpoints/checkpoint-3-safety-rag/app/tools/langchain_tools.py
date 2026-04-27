"""Build LangChain async tools from the registry.

LangChain's `StructuredTool.from_function(coroutine=...)` wraps an async
function as a tool LangGraph can `ainvoke`. We funnel every call through
`registry.execute(name, args)` so the SAME execution path is used by:
  - the raw agent loop,
  - the LangGraph ReAct agent,
  - the orchestrator's executor.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from app.tools.registry import ToolRegistry
from app.tools.schemas import CalculatorInput, ClockInput, EmployeeLookupInput


def _wrap(registry: ToolRegistry, name: str, args_schema=None, description: str | None = None):
    """Wrap a registry tool as a LangChain ``StructuredTool``."""

    async def _runner(**kwargs):
        # Funnel into the registry so error handling, logging, and any future
        # cross-cutting concerns (rate limiting, traces, ...) live in one place.
        result = await registry.execute(name, kwargs)
        return result.output  # ToolResult -> string for the LLM

    tool_def = next(t.definition for t in registry.all() if t.name == name)
    return StructuredTool.from_function(
        coroutine=_runner,
        name=name,
        description=description or tool_def["description"],
        args_schema=args_schema,
    )


def build_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    """Build the LangChain tool list LangGraph hands to the LLM.

    Tools that benefit from input validation get an ``args_schema``; the
    others rely on the JSON-schema published in their ``ToolDefinition``.
    """
    return [
        _wrap(registry, "calculator", args_schema=CalculatorInput),
        _wrap(registry, "clock", args_schema=ClockInput),
        _wrap(registry, "knowledge_search"),
        _wrap(registry, "task_manager"),
        _wrap(registry, "employee_lookup", args_schema=EmployeeLookupInput),
    ]
