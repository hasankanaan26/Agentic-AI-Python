"""Domain-level exceptions.

Engineering standard: distinguish "the agent recovered" from "the agent
crashed". Tool failures are returned to the LLM as structured `ToolResult`s
so it can reason about them. These exceptions only fire for unrecoverable
problems (config wrong at boot, vector store unreachable, etc.).
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-layer errors that should bubble to the route.

    Catch this in a request handler to translate any of the typed
    subclasses below into an HTTP error response.
    """


class ProviderConfigError(AgentError):
    """Raised at startup when provider credentials are inconsistent.

    Typically thrown by :class:`app.settings.Settings` validators when the
    selected ``LLM_PROVIDER`` doesn't have its companion credentials set.
    """


class InjectionDetected(AgentError):
    """Raised when the safety layer blocks a request before it runs.

    The route layer turns this into a 400 with the safety findings.
    """


class VectorStoreUnavailable(AgentError):
    """Chroma collection is missing or unreachable.

    Surfaced when a RAG-dependent tool can't reach its persistent store.
    """
