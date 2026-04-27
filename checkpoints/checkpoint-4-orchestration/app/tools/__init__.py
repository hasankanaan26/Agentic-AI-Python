"""Tool catalogue.

Each tool is a small async class deriving from :class:`app.tools.base.BaseTool`.
The :class:`app.tools.registry.ToolRegistry` collects them at startup and is
the single execution surface used by the raw loop, the LangGraph agent,
and the orchestrator's per-step executor.
"""
