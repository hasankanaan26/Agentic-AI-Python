"""Domain-level exceptions.

Engineering standard: distinguish "the agent recovered" from "the agent
crashed". Tool failures are returned to the LLM as structured `ToolResult`s
so it can reason about them. These exceptions only fire for unrecoverable
problems (config wrong at boot, vector store unreachable, etc.).
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-layer errors that should bubble to the route.

    Catch this in route handlers (or let FastAPI return a 500) when a problem
    is unrecoverable. Recoverable tool failures should be returned as
    ``ToolResult(status="error", ...)`` instead.
    """


class ProviderConfigError(AgentError):
    """Raised at startup when provider credentials are inconsistent."""


class InjectionDetected(AgentError):
    """Raised when the safety layer blocks a request before it runs."""


class VectorStoreUnavailable(AgentError):
    """Chroma collection is missing or unreachable."""
