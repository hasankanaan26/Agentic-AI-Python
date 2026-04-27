"""Tool registry — the single source of truth for which tools exist.

CP1 ships two tools: calculator + clock. More are added in CP2 and CP3.
The registry is built once during the FastAPI lifespan and shared across
all requests. It exposes the tool definitions to the LLM and provides a
single `execute` entrypoint that converts exceptions into structured
`ToolResult` failures — the agent loop must never crash on a bad tool
call.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models.tool import ToolResult
from app.settings import Settings
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.clock import ClockTool

log = get_logger(__name__)


class ToolRegistry:
    """Holds the runtime catalogue of tools and dispatches calls to them.

    The registry is constructed once via `ToolRegistry.build(settings)`
    during application startup. Routes look up tool definitions for the
    LLM and route tool-call requests through `execute`, which guarantees a
    `ToolResult` is always returned (never an unhandled exception).
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        """Index tools by name for O(1) lookup at execute-time."""
        # Build a name -> tool map so `execute` can resolve calls quickly.
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}

    @classmethod
    def build(cls, settings: Settings) -> ToolRegistry:
        """Build the default registry for the current environment.

        ``settings`` is accepted (and intentionally unused in CP1) so the
        signature is stable: later checkpoints will read it to decide
        which optional tools to load.

        Args:
            settings: Validated app settings. Currently unused.

        Returns:
            A populated `ToolRegistry` ready for use.
        """
        tools: list[BaseTool] = [CalculatorTool(), ClockTool()]
        log.info("tool_registry_ready", tools=[t.name for t in tools])
        return cls(tools)

    def all(self) -> list[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def definitions(self, allowed: list[str] | None = None) -> list[dict]:
        """Return JSON-schema tool definitions for the LLM.

        Args:
            allowed: Optional filter list. When provided, only tools
                whose name is in `allowed` are returned. When None, every
                registered tool is returned.

        Returns:
            A list of tool-definition dicts ready to send to the provider.
        """
        return [t.definition for t in self.all() if allowed is None or t.name in allowed]

    def permissions(self) -> dict[str, str]:
        """Return a `{tool_name: permission}` map for safety checks."""
        return {t.name: t.permission for t in self.all()}

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Run a tool by name with the LLM-supplied arguments.

        This is the single chokepoint between "the LLM said to call X"
        and "X actually runs". It converts every failure mode (unknown
        tool, bad arguments, raised exception) into a `ToolResult` so the
        agent loop can keep running.

        Args:
            name: Tool identifier as chosen by the LLM.
            arguments: Already-parsed kwargs to pass to the tool's `run`.

        Returns:
            A `ToolResult` — `ok` on success, `fail` for any error class.
        """
        tool = self._tools.get(name)
        if tool is None:
            # The LLM hallucinated a tool we don't have. Tell it cleanly.
            return ToolResult.fail(f"Unknown tool '{name}'.")
        try:
            return await tool.run(**arguments)
        except TypeError as e:
            # `TypeError` here almost always means the LLM produced an
            # argument shape that doesn't match the tool's signature.
            return ToolResult.fail(f"Bad arguments for {name}: {e}")
        except Exception as e:
            # Catch-all: a genuinely buggy tool. Log a stack trace for
            # the operator and return a structured error to the agent.
            log.exception("tool_unexpected_error", tool=name)
            return ToolResult.fail(f"Unexpected tool error: {e}")
