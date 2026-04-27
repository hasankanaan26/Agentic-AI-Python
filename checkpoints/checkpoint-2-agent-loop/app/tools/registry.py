"""Tool registry — the single source of truth for which tools exist.

Constructed once in lifespan and handed to routes via Depends.
CP2 has four tools; knowledge_search arrives in CP3 with the RAG stack.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models.tool import ToolResult
from app.settings import Settings
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.clock import ClockTool
from app.tools.employee_lookup import EmployeeLookupTool
from app.tools.task_manager import TaskManagerTool

log = get_logger(__name__)


class ToolRegistry:
    """Indexes tool instances by name and dispatches LLM-driven invocations.

    Construct via :meth:`build` so the configured cache sizes / file paths
    are wired in. Routes receive the registry through DI and use only the
    public methods — they never see individual tool classes.
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        """Wrap a pre-built list of tools in a name-indexed dict."""
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}

    @classmethod
    def build(cls, settings: Settings) -> ToolRegistry:
        """Construct the canonical CP2 registry from settings.

        This is the single place new tools should be registered as the
        codebase grows. Args are settings-driven so tests can swap in
        different cache sizes / data files without subclassing.
        """
        tools: list[BaseTool] = [
            CalculatorTool(),
            ClockTool(),
            EmployeeLookupTool(
                cache_ttl=settings.tool_cache_ttl_seconds,
                cache_max=settings.tool_cache_maxsize,
            ),
            TaskManagerTool(tasks_path=settings.tasks_data_path),
        ]
        log.info("tool_registry_ready", tools=[t.name for t in tools])
        return cls(tools)

    def all(self) -> list[BaseTool]:
        """Return every registered tool instance, in registration order."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return every registered tool name."""
        return list(self._tools.keys())

    def definitions(self, allowed: list[str] | None = None) -> list[dict]:
        """Return JSON-schema definitions, optionally filtered to a whitelist.

        Args:
            allowed: If provided, only definitions whose name is in this
                list are returned. ``None`` means "all tools".
        """
        return [t.definition for t in self.all() if allowed is None or t.name in allowed]

    def permissions(self) -> dict[str, str]:
        """Return the ``{tool_name: permission}`` map ("read" or "write")."""
        return {t.name: t.permission for t in self.all()}

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Dispatch a tool call by name with structured error handling.

        This method NEVER raises — every failure is converted into a
        :class:`ToolResult.fail` so the agent loop can keep running and
        feed the error back to the LLM.

        Args:
            name: Registered tool name (typically chosen by the LLM).
            arguments: Keyword arguments for the tool's ``run`` method.

        Returns:
            The tool's :class:`ToolResult`, or a ``fail`` result if the
            tool is unknown, the arguments don't match, or an unexpected
            exception escaped the tool's own ``run``.
        """
        tool = self._tools.get(name)
        if tool is None:
            # The LLM hallucinated a tool name; report it back so it can retry.
            return ToolResult.fail(f"Unknown tool '{name}'.")
        try:
            return await tool.run(**arguments)
        except TypeError as e:
            # Most likely cause: LLM passed wrong/missing argument names.
            return ToolResult.fail(f"Bad arguments for {name}: {e}")
        except Exception as e:
            # Last-ditch safety net. We log with traceback so we can debug
            # later, but still convert to a ToolResult so the loop survives.
            log.exception("tool_unexpected_error", tool=name)
            return ToolResult.fail(f"Unexpected tool error: {e}")
