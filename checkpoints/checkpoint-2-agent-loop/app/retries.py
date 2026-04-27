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

# Errors we *do* retry on. These represent transient network failures only.
# Auth errors, 4xx validation errors, and content-policy violations are NOT
# retried — retrying them just burns latency without changing the outcome.
_TRANSIENT = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    ConnectionError,
)


def llm_retry(*, attempts: int = 3, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a tenacity retry policy suited to chat-completion endpoints.

    The returned object is meant to be consumed as::

        async for attempt in llm_retry():
            with attempt:
                await llm_call(...)

    Args:
        attempts: Maximum total tries (initial attempt + retries).
        max_wait: Cap on the exponential backoff between attempts, in seconds.

    Returns:
        A configured ``AsyncRetrying`` that re-raises the last exception once
        the budget is exhausted (``reraise=True``).
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        # Exponential backoff: 0.5s, 1s, 2s, ... capped at ``max_wait``.
        wait=wait_exponential(multiplier=0.5, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )


def embedding_retry(*, attempts: int = 4, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a retry policy tuned for embedding endpoints.

    Embedding endpoints rate-limit harder than chat endpoints, so we give
    them an extra attempt and a slightly steeper backoff.

    Args:
        attempts: Maximum total tries.
        max_wait: Cap on the exponential backoff between attempts, in seconds.

    Returns:
        A configured ``AsyncRetrying`` that re-raises after exhausting the budget.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        # Slightly steeper multiplier because rate-limit windows are longer.
        wait=wait_exponential(multiplier=0.75, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )
