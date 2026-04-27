"""Trace models surfaced by ``GET /traces`` and ``GET /traces/{id}``.

These mirror the in-memory dicts the :class:`app.services.tracer.TraceStore`
keeps -- no transformations, just a typed view at the API edge.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TraceEntry(BaseModel):
    """One event recorded against a running trace (planning, tool call, ...)."""

    action: str
    detail: dict[str, Any]
    timestamp: str


class AgentTrace(BaseModel):
    """Full trace returned by ``GET /traces/{trace_id}``.

    ``entries`` is in append order; ``end_time`` is ``None`` while running.
    """

    trace_id: str
    goal: str
    start_time: str
    end_time: str | None
    entries: list[TraceEntry]
    status: str


class TraceSummary(BaseModel):
    """Lightweight row returned by ``GET /traces`` (no entries)."""

    trace_id: str
    goal: str
    start_time: str
    end_time: str | None
    status: str
    entry_count: int
