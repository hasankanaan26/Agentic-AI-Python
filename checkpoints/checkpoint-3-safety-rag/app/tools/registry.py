"""Tool registry — the single source of truth for which tools exist.

Constructed once in lifespan and handed to routes via Depends.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models.tool import ToolResult
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.settings import Settings
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.clock import ClockTool
from app.tools.employee_lookup import EmployeeLookupTool
from app.tools.knowledge_search import KnowledgeSearchTool
from app.tools.task_manager import TaskManagerTool

log = get_logger(__name__)


class ToolRegistry:
    """Name -> tool lookup table shared by every agent runner.

    Exposes the JSON-schema definitions the LLM sees and the single
    ``execute`` entry point that funnels every tool invocation through the
    same error-handling path.
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}

    @classmethod
    def build(
        cls, settings: Settings, embeddings: EmbeddingService, store: VectorStore
    ) -> ToolRegistry:
        """Construct the canonical tool set for this app.

        Called once from ``app.lifespan`` after the embedding service and
        vector store are ready (knowledge_search depends on them).
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
        """Return every registered tool in insertion order."""
        return list(self._tools.values())

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name. Returns ``None`` if unknown."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def definitions(self, allowed: list[str] | None = None) -> list[dict]:
        """Return JSON-schema definitions for every (or allow-listed) tool."""
        return [t.definition for t in self.all() if allowed is None or t.name in allowed]

    def permissions(self) -> dict[str, str]:
        """Return ``{tool_name: 'read' | 'write'}`` for the safety endpoint."""
        return {t.name: t.permission for t in self.all()}

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool by name with structured error semantics.

        Every failure path returns a ``ToolResult`` rather than raising so
        the agent loop can keep going. Unknown tool, bad argument types,
        and unexpected exceptions are all converted into an error result
        the LLM can read.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"Unknown tool '{name}'.")
        try:
            return await tool.run(**arguments)
        except TypeError as e:
            # Bad arguments: surface as a structured error so the LLM can adapt.
            return ToolResult.fail(f"Bad arguments for {name}: {e}")
        except Exception as e:
            # Catch-all so a buggy tool can never bring down the agent.
            log.exception("tool_unexpected_error", tool=name)
            return ToolResult.fail(f"Unexpected tool error: {e}")
