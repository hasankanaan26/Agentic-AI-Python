"""Tool implementations and the registry that owns them.

A "tool" is a small async callable the LLM is allowed to invoke. Each tool
ships a JSON-schema definition (so the LLM knows how to call it) and a
`run()` coroutine that returns a `ToolResult`. The `ToolRegistry` is the
single source of truth for which tools the agent can see at runtime.
"""
