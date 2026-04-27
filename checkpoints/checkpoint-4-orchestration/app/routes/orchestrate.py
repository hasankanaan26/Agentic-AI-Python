"""Multi-agent orchestration endpoints.

Three endpoints, each with one job:

  - POST /orchestrate                       blocking; supports `require_approval`.
  - POST /orchestrate/stream                Server-Sent Events; auto-mode only.
  - POST /orchestrate/resume/{trace_id}     continues an awaiting-approval run.

Streaming + HITL are kept on separate endpoints so the wire format of
each is easy to read.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.deps import get_orchestrator
from app.models import (
    OrchestrateResumeRequest,
    OrchestrationRequest,
    OrchestrationResponse,
)
from app.services.orchestrator import OrchestratorService

router = APIRouter(tags=["orchestration"])


@router.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(
    request: OrchestrationRequest,
    orchestrator: Annotated[OrchestratorService, Depends(get_orchestrator)],
) -> OrchestrationResponse:
    """Blocking plan + execute with optional approval gate."""
    result = await orchestrator.run(
        goal=request.goal,
        allowed_tools=request.allowed_tools,
        require_approval=request.require_approval,
    )
    return OrchestrationResponse(**result)


@router.post("/orchestrate/stream")
async def orchestrate_stream(
    request: OrchestrationRequest,
    orchestrator: Annotated[OrchestratorService, Depends(get_orchestrator)],
) -> StreamingResponse:
    """Server-Sent Events variant -- streams planning + per-step progress.

    Auto-mode only: the SSE channel does not support the approval gate
    because clients can't easily reply mid-stream.
    """

    async def event_source():
        # Each yielded dict from the orchestrator becomes one SSE ``data:`` line.
        async for event in orchestrator.astream_events(
            goal=request.goal,
            allowed_tools=request.allowed_tools,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.post("/orchestrate/resume/{trace_id}", response_model=OrchestrationResponse)
async def orchestrate_resume(
    trace_id: str,
    request: OrchestrateResumeRequest,
    orchestrator: Annotated[OrchestratorService, Depends(get_orchestrator)],
) -> OrchestrationResponse:
    """Approve or reject an orchestration paused at the approval gate.

    Returns 404 if no pending orchestration exists for ``trace_id`` (it
    was never paused, or already resumed/expired).
    """
    try:
        result = await orchestrator.resume(trace_id=trace_id, approved=request.approved)
    except KeyError as e:
        # Translate the orchestrator's KeyError into the right HTTP status.
        raise HTTPException(status_code=404, detail=str(e)) from e
    return OrchestrationResponse(**result)
