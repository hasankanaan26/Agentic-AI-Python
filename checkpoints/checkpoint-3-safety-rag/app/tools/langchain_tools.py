"""Build LangChain async tools from the registry — using the official `@tool` decorator.

We rely on the framework for as much as possible:
  - ``langchain_core.tools.tool`` decorates an async function as a Tool;
  - ``args_schema=...`` plugs in our Pydantic models for input validation;
  - ``ToolException`` + ``handle_tool_error=True`` lets the framework mark
    failed runs as ``ToolMessage(status="error")`` while still surfacing the
    error string to the LLM (preserving the "errors are data" contract).

We keep one layer of indirection: the body of each decorated function calls
``registry.execute(name, args)`` so the raw agent loop and the LangGraph
agent share the SAME execution path.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, ToolException, tool

from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    CalculatorInput,
    ClockInput,
    EmployeeLookupInput,
    KnowledgeSearchInput,
    TaskManagerInput,
)


def _make_tool(
    registry: ToolRegistry,
    name: str,
    args_schema,
    description: str | None = None,
) -> BaseTool:
    """Wrap a registry tool as a LangChain ``BaseTool`` via the ``@tool`` decorator."""
    tool_def = next(t.definition for t in registry.all() if t.name == name)

    @tool(
        name,
        args_schema=args_schema,
        description=description or tool_def["description"],
    )
    async def _runner(**kwargs) -> str:
        # Funnel into the registry so logging, retries, and future cross-cutting
        # concerns live in one place.
        result = await registry.execute(name, kwargs)
        if result.status == "error":
            # Raising ``ToolException`` is the official LangChain signal for a
            # recoverable tool failure. The framework converts it into a
            # ``ToolMessage(status="error")`` and — because we set
            # ``handle_tool_error=True`` below — still hands the error string
            # back to the LLM as the observation, so the agent can react.
            raise ToolException(result.output)
        return result.output

    # ``handle_tool_error=True`` keeps the "errors are data" behaviour: the LLM
    # sees the error string as a tool observation rather than the run aborting.
    _runner.handle_tool_error = True
    return _runner


def build_langchain_tools(registry: ToolRegistry) -> list[BaseTool]:
    """Build the LangChain tool list LangGraph hands to the LLM."""
    return [
        _make_tool(registry, "calculator", CalculatorInput),
        _make_tool(registry, "clock", ClockInput),
        _make_tool(registry, "knowledge_search", KnowledgeSearchInput),
        _make_tool(registry, "task_manager", TaskManagerInput),
        _make_tool(registry, "employee_lookup", EmployeeLookupInput),
    ]
