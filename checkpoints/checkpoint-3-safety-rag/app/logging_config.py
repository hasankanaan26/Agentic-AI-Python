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
    """Wire stdlib logging + structlog. Call once at startup.

    Args:
        level: Minimum level to emit (``"DEBUG"``, ``"INFO"``, ...).
        json_output: ``True`` for one-line JSON (production friendly), ``False``
            for the colorised dev renderer.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # All stdlib loggers (uvicorn, httpx, openai, etc.) flow through here.
    # ``force=True`` resets handlers so this works even when uvicorn has
    # already installed its own logging config.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Processors run in order on every log record before the renderer.
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    # Choose the final formatter: machine-readable JSON or human-friendly text.
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        # First call to get_logger(name) is cached forever — fast, but means
        # config changes after first use won't take effect until restart.
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, typically named after ``__name__``."""
    return structlog.get_logger(name)
