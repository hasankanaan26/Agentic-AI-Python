"""Tool implementations and the registry that wires them together.

Each tool is a subclass of :class:`app.tools.base.BaseTool` and lives in
its own module. :class:`app.tools.registry.ToolRegistry` is the single
source of truth for what's available at runtime; routes only ever see
the registry, never individual tool classes.
"""
