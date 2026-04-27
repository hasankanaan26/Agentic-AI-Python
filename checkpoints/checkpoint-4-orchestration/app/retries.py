"""Async retry policies built on tenacity.

Engineering standard: every external call (LLM, embedding, tool API) is
wrapped with a timeout + bounded retry. When the budget is exhausted we
log structured detail and surface a typed error — never an unbounded
exception that takes down the request.
"""

from __future__ import annotations

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Errors we *do* retry on. Auth errors / 4xx-validation are not retried.
_TRANSIENT = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    ConnectionError,
)


def llm_retry(*, attempts: int = 3, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a tenacity retry controller for LLM chat-completion calls.

    Use as::

        async for attempt in llm_retry():
            with attempt:
                await client.chat.completions.create(...)

    Args:
        attempts: Maximum total attempts (initial call + retries).
        max_wait: Upper bound on the exponential backoff in seconds.

    Returns:
        An ``AsyncRetrying`` controller with ``reraise=True`` so the original
        exception bubbles after the budget is exhausted.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )


def embedding_retry(*, attempts: int = 4, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a retry controller for embedding endpoints.

    Embedding APIs typically rate-limit harder than chat APIs, so the
    defaults here are slightly more generous (one extra attempt and a
    larger backoff multiplier) than :func:`llm_retry`.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.75, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )
