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
        level: One of ``DEBUG``/``INFO``/``WARNING``/``ERROR``. Falls back to
            ``INFO`` on an unknown value.
        json_output: ``True`` for one-JSON-object-per-line (production); ``False``
            for the colourised dev renderer (local).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # All stdlib loggers (uvicorn, httpx, openai, etc.) flow through here.
    # ``force=True`` rebinds the root handler in case uvicorn already attached one.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Processors run in order: each one transforms the event dict before
    # the final renderer turns it into a string.
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,  # request_id etc. injected via contextvars
        structlog.processors.add_log_level,  # adds "level": "info" key
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),  # renders stack info if asked for
    ]

    # Pick the terminal stage. JSON ships well to log aggregators; the dev
    # renderer is only nice when a human is reading the terminal.
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally tagged with ``name``."""
    return structlog.get_logger(name)
