"""Tool registry — the single source of truth for which tools exist.

Constructed once in lifespan and handed to routes via Depends.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models.tool import ToolResult
from app.settings import Settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.clock import ClockTool
from app.tools.employee_lookup import EmployeeLookupTool
from app.tools.knowledge_search import KnowledgeSearchTool
from app.tools.task_manager import TaskManagerTool

log = get_logger(__name__)


class ToolRegistry:
    """Name-keyed catalogue of :class:`BaseTool` instances.

    The single execution choke point. Both the raw agent loop and the
    LangGraph runner ultimately call :meth:`execute`, so unknown-tool
    handling and bad-argument errors live in exactly one place.
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}

    @classmethod
    def build(
        cls, settings: Settings, embeddings: EmbeddingService, store: VectorStore
    ) -> ToolRegistry:
        """Construct the production registry with all five tools wired up.

        Caching parameters and data paths come from :class:`Settings` so tests
        can swap them without monkey-patching the tool classes.
        """
        tools: list[BaseTool] = [
            CalculatorTool(),
            ClockTool(),
            EmployeeLookupTool(
                cache_ttl=settings.tool_cache_ttl_seconds,
                cache_max=settings.tool_cache_maxsize,
            ),
            TaskManagerTool(tasks_path=settings.tasks_data_path),
            KnowledgeSearchTool(
                embeddings=embeddings,
                store=store,
                knowledge_path=settings.knowledge_data_path,
                cache_ttl=settings.tool_cache_ttl_seconds,
                cache_max=settings.tool_cache_maxsize,
            ),
        ]
        log.info("tool_registry_ready", tools=[t.name for t in tools])
        return cls(tools)

    def all(self) -> list[BaseTool]:
        """Return every registered tool in registration order."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return every registered tool name, in registration order."""
        return list(self._tools.keys())

    def definitions(self, allowed: list[str] | None = None) -> list[dict]:
        """Return JSON-schema definitions, optionally filtered by ``allowed`` names."""
        return [t.definition for t in self.all() if allowed is None or t.name in allowed]

    def permissions(self) -> dict[str, str]:
        """Return ``{tool_name: 'read'|'write'}`` for the safety/permissions endpoint."""
        return {t.name: t.permission for t in self.all()}

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Look up a tool by name and run it with ``arguments``.

        Wraps the tool's ``run`` so callers get a uniform :class:`ToolResult`
        even on unknown tools, bad LLM arguments, or unexpected exceptions.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"Unknown tool '{name}'.")
        try:
            return await tool.run(**arguments)
        except TypeError as e:
            # Bad arguments: surface as a structured error so the LLM can adapt.
            return ToolResult.fail(f"Bad arguments for {name}: {e}")
        except Exception as e:  # noqa: BLE001
            # Unexpected: log the full stack but still hand a structured error to the LLM.
            log.exception("tool_unexpected_error", tool=name)
            return ToolResult.fail(f"Unexpected tool error: {e}")
