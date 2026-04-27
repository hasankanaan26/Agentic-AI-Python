"""LangGraph Studio entry point — exposes a compiled graph for visual debug.

Studio loads this module at startup, picks up `executor_graph`, and renders
its nodes / edges / state in the browser. Run from the project root:

    pip install "langgraph-cli[inmem]"
    langgraph dev

Open the URL printed (usually http://localhost:8123) and select "executor"
from the graph dropdown.

We deliberately wire a slim 4-tool registry here (no `knowledge_search`)
so Studio doesn't need to boot Chroma or hit an embedding provider just
to draw the graph. The point of this module is the visualisation.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.agents.langgraph import build_chat_model
from app.prompts import SAFETY_SYSTEM_PROMPT
from app.settings import get_settings
from app.tools.calculator import CalculatorTool
from app.tools.clock import ClockTool
from app.tools.employee_lookup import EmployeeLookupTool
from app.tools.langchain_tools import build_langchain_tools
from app.tools.registry import ToolRegistry
from app.tools.task_manager import TaskManagerTool


def _studio_registry(settings) -> ToolRegistry:
    """Build the slim 4-tool registry exposed to LangGraph Studio.

    Excludes ``knowledge_search`` so Studio doesn't need Chroma or an
    embedding provider just to render the graph.
    """
    return ToolRegistry(
        [
            CalculatorTool(),
            ClockTool(),
            EmployeeLookupTool(
                cache_ttl=settings.tool_cache_ttl_seconds,
                cache_max=settings.tool_cache_maxsize,
            ),
            TaskManagerTool(tasks_path=settings.tasks_data_path),
        ]
    )


def _build():
    """Compile a fresh LangGraph ReAct agent for Studio to load."""
    settings = get_settings()
    registry = _studio_registry(settings)
    return create_react_agent(
        build_chat_model(settings),
        tools=build_langchain_tools(registry),
        prompt=SAFETY_SYSTEM_PROMPT,
        # MemorySaver -- Studio sessions are ephemeral, no need for sqlite here.
        checkpointer=MemorySaver(),
    )


# Module-level graph picked up by `langgraph dev` (referenced from langgraph.json).
executor_graph = _build()
