"""In-memory trace store for the /traces endpoint.

LangSmith handles production tracing automatically when these env vars
are set:

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=project-4-agents       # optional

When LangSmith is on, every LLM call, tool call, and graph node from
LangChain/LangGraph is shipped to the LangSmith UI for free — latency,
token counts, prompt diffs, the lot. We don't need a custom callback
handler for that anymore.

The TraceStore below stays for the offline demo path and the existing
/traces routes — the orchestrator pushes events to it directly while
running, so students can `GET /traces/{trace_id}` without any external
service.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

from app.logging_config import get_logger

log = get_logger(__name__)


def langsmith_enabled() -> bool:
    """Return True iff LangSmith auto-tracing is configured.

    Used at startup to log whether the LangChain tracing callback will
    fire automatically; doesn't change behaviour, just observability.
    """
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.getenv("LANGCHAIN_API_KEY"))
    )


class TraceStore:
    """In-memory ring buffer of agent traces.

    One instance per process is created in lifespan. All mutations are
    guarded by an ``asyncio.Lock`` so concurrent requests don't trample
    each other. Capacity is bounded by ``max_size`` -- the oldest trace
    is evicted when the buffer fills.
    """

    def __init__(self, max_size: int = 100) -> None:
        self._lock = asyncio.Lock()
        self._traces: dict[str, dict] = {}
        self._max_size = max_size

    async def create(self, goal: str) -> str:
        """Open a new trace and return its short id.

        Args:
            goal: Goal string the agent/orchestrator was asked to satisfy.

        Returns:
            A ``trace_xxxxxxxx`` identifier used as the key for later writes.
        """
        trace_id = f"trace_{uuid.uuid4().hex[:8]}"
        async with self._lock:
            self._traces[trace_id] = {
                "trace_id": trace_id,
                "goal": goal,
                "start_time": _now(),
                "end_time": None,
                "entries": [],
                "status": "running",
            }
            # Evict oldest entries until we're back within capacity.
            while len(self._traces) > self._max_size:
                oldest = min(self._traces.keys(), key=lambda k: self._traces[k]["start_time"])
                del self._traces[oldest]
        log.info("trace_created", trace_id=trace_id, goal=goal[:100])
        return trace_id

    async def add_entry(self, trace_id: str, action: str, detail: dict) -> None:
        """Append a single event to a running trace.

        Silently drops the entry if ``trace_id`` was already evicted -- callers
        are expected to no-op rather than crash on missing traces.
        """
        async with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace["entries"].append(
                {
                    "action": action,
                    "detail": detail,
                    # Allow callers to override timestamp (e.g. when replaying).
                    "timestamp": detail.get("timestamp", _now()),
                }
            )

    async def complete(self, trace_id: str, status: str = "completed") -> None:
        """Mark a trace finished and stamp its end time."""
        async with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace["end_time"] = _now()
            trace["status"] = status
        log.info("trace_completed", trace_id=trace_id, status=status)

    async def get(self, trace_id: str) -> dict | None:
        """Return the full trace dict, or ``None`` if it was evicted/unknown."""
        async with self._lock:
            return self._traces.get(trace_id)

    async def list_summaries(self, limit: int = 20) -> list[dict]:
        """Return up to ``limit`` traces, newest first, with entry counts only."""
        async with self._lock:
            traces = sorted(
                self._traces.values(), key=lambda t: t["start_time"], reverse=True
            )[:limit]
            return [
                {
                    "trace_id": t["trace_id"],
                    "goal": t["goal"],
                    "start_time": t["start_time"],
                    "end_time": t["end_time"],
                    "status": t["status"],
                    "entry_count": len(t["entries"]),
                }
                for t in traces
            ]


def _now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()
