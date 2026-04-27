"""Tool implementations and the registry that wires them together.

Each concrete tool subclasses ``app.tools.base.BaseTool`` and exposes an
async ``run`` method plus a JSON-schema ``definition`` that the LLM sees
as a function declaration. ``registry.ToolRegistry`` is the single source
of truth used by both the raw agent loop and the LangGraph runner.
"""
