"""GET /traces — observability over agent runs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_traces
from app.models import AgentTrace, TraceSummary
from app.services.tracer import TraceStore

router = APIRouter(tags=["traces"])


@router.get("/traces", response_model=list[TraceSummary])
async def traces_list(
    traces: Annotated[TraceStore, Depends(get_traces)],
    limit: int = 20,
) -> list[TraceSummary]:
    """Return up to ``limit`` recent traces (newest first), summary view only."""
    return [TraceSummary(**t) for t in await traces.list_summaries(limit=limit)]


@router.get("/traces/{trace_id}", response_model=AgentTrace)
async def traces_detail(
    trace_id: str,
    traces: Annotated[TraceStore, Depends(get_traces)],
) -> AgentTrace:
    """Return the full trace (with all events) for ``trace_id`` or 404."""
    trace = await traces.get(trace_id)
    if trace is None:
        # 404 is more honest than returning an empty trace.
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return AgentTrace(**trace)
