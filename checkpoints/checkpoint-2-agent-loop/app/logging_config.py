"""Structured logging via structlog. JSON in production, key/value locally.

Engineering standard: no print statements, no stdlib formatter strings
that fall apart when fields contain commas. structlog gives every log
line a typed dict you can ship straight to a log aggregator.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Wire stdlib logging and structlog together. Call once at startup.

    Args:
        level: Minimum log level name (``"DEBUG"``, ``"INFO"``, ...). Falls
            back to ``INFO`` if the name is unknown.
        json_output: When True, emit machine-readable JSON (production).
            When False, render with colors for local development.
    """
    # Resolve the textual level to the stdlib numeric value, defaulting to INFO.
    log_level = getattr(logging, level.upper(), logging.INFO)

    # All stdlib loggers (uvicorn, httpx, openai, etc.) flow through here.
    # ``force=True`` reconfigures even if some library already attached a handler.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Processors that run on every event regardless of output format.
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,  # pull bound contextvars in
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    # Pick a renderer based on environment: JSON for log shippers, console for humans.
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        # Caching is safe because configuration is fixed for the process lifetime.
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a module-scoped structlog logger.

    Args:
        name: Logger name, conventionally ``__name__`` of the caller.

    Returns:
        A bound logger that respects the configuration set by ``configure_logging``.
    """
    return structlog.get_logger(name)
