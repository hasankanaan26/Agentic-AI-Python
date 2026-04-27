"""Domain-level exceptions.

Engineering standard: distinguish "the agent recovered" from "the agent
crashed". Tool failures are returned to the LLM as structured `ToolResult`s
so it can reason about them. These exceptions only fire for unrecoverable
problems (config wrong at boot, vector store unreachable, etc.).
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-layer errors that should bubble up to the route handler.

    Catch this in route-level error handlers to translate domain failures
    into HTTP responses. Tool-level errors are NOT raised — they are returned
    as ``ToolResult(status="error", ...)`` so the agent can recover.
    """


class ProviderConfigError(AgentError):
    """Raised at startup when provider credentials are inconsistent.

    Example: ``LLM_PROVIDER=openai`` is set but ``OPENAI_API_KEY`` is empty.
    Triggered by the Pydantic ``model_validator`` on ``Settings``.
    """


class InjectionDetected(AgentError):
    """Raised when the safety layer blocks a request before it runs.

    Used by the (optional) prompt-injection guard so the agent loop never
    sends suspicious user content to the LLM.
    """


class VectorStoreUnavailable(AgentError):
    """Raised when the Chroma collection is missing or unreachable.

    Surfaced by the RAG layer (introduced in a later checkpoint) so the
    agent route can fail fast instead of running with an empty index.
    """
