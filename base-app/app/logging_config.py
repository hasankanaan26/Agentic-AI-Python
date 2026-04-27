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
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``). Falls
            back to ``INFO`` if the name isn't recognised.
        json_output: ``True`` for machine-parseable JSON lines (production /
            log aggregators); ``False`` for the colourful console renderer
            that's friendlier during local development.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # All stdlib loggers (uvicorn, httpx, openai, etc.) flow through here.
    # ``force=True`` evicts any handlers that imported libraries may have
    # attached so we have a single, predictable logging pipeline.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Processors run in order on every log record. ``merge_contextvars``
    # pulls in any context bound via ``structlog.contextvars.bind_contextvars``
    # (e.g. request_id) so every line within a request is correlated.
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    # Final processor decides the on-the-wire format.
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        # Caching the bound logger is safe because we don't reconfigure
        # at runtime; it makes ``get_logger`` essentially free per call.
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger, optionally namespaced by ``name``.

    Args:
        name: Usually ``__name__`` from the calling module so log lines
            are easy to filter by source.

    Returns:
        A bound logger that accepts arbitrary keyword fields, e.g.
        ``log.info("agent_step", step=1, tool="calculator")``.
    """
    return structlog.get_logger(name)
