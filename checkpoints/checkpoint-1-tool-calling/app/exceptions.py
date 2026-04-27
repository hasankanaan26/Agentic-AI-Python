"""Domain-level exceptions.

Engineering standard: distinguish "the agent recovered" from "the agent
crashed". Tool failures are returned to the LLM as structured `ToolResult`s
so it can reason about them. These exceptions only fire for unrecoverable
problems (config wrong at boot, vector store unreachable, etc.).
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-layer errors that should bubble to the route.

    Anything inheriting from `AgentError` is considered an unrecoverable
    condition that the route handler (or FastAPI exception handler) will
    surface to the client. Tool-level failures should NOT subclass this —
    they are expected to be returned as `ToolResult.fail(...)` instead.
    """


class ProviderConfigError(AgentError):
    """Raised at startup when provider credentials are inconsistent."""


class InjectionDetected(AgentError):
    """Raised when the safety layer blocks a request before it runs."""


class VectorStoreUnavailable(AgentError):
    """Chroma collection is missing or unreachable."""
