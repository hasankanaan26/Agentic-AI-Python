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

# Errors we *do* retry on. Auth errors / 4xx-validation are not retried —
# retrying a 401 just burns API budget. Network-level glitches (timeouts,
# resets) are exactly the class of failure where backoff helps.
_TRANSIENT = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    ConnectionError,
)


def llm_retry(*, attempts: int = 3, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a retry policy for LLM chat-completion calls.

    Use as ``async for attempt in llm_retry(): with attempt: await ...``.

    Args:
        attempts: Maximum total attempts before giving up (the original
            call counts as attempt #1).
        max_wait: Cap on the exponential backoff in seconds.

    Returns:
        A configured `AsyncRetrying` instance that re-raises the final
        exception if all attempts fail.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        # Exponential backoff: 0.5s, 1s, 2s, ... capped at `max_wait`.
        wait=wait_exponential(multiplier=0.5, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        # `reraise=True` surfaces the original exception type rather than
        # wrapping it in tenacity's RetryError — easier for callers.
        reraise=True,
    )


def embedding_retry(*, attempts: int = 4, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a retry policy tuned for embedding endpoints.

    Embedding endpoints rate-limit harder than chat endpoints, so we use
    a slightly more patient policy here (one extra attempt, larger
    backoff multiplier).

    Args:
        attempts: Maximum total attempts before giving up.
        max_wait: Cap on the exponential backoff in seconds.

    Returns:
        A configured `AsyncRetrying` instance.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.75, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )
