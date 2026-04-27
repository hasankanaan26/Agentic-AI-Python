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
# those are deterministic and would just burn our retry budget.
_TRANSIENT = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    ConnectionError,
)


def llm_retry(*, attempts: int = 3, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a tenacity retry policy for chat-completion LLM calls.

    Use as ``async for attempt in llm_retry(): with attempt: await ...``.

    Args:
        attempts: Total tries (including the first). After this many transient
            failures the exception is reraised to the caller.
        max_wait: Cap on the exponential backoff sleep, in seconds.

    Returns:
        An ``AsyncRetrying`` instance ready to drive an async-for loop.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )


def embedding_retry(*, attempts: int = 4, max_wait: float = 8.0) -> AsyncRetrying:
    """Build a tenacity retry policy tuned for embedding endpoints.

    Embedding endpoints rate-limit harder than chat endpoints, so we allow
    one extra attempt and a slightly slower backoff growth.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.75, max=max_wait),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )
