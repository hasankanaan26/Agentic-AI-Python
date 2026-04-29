"""/tasks — read-only view of the task_manager state.

Mutations go through the agent (``POST /agent/run``) so writes can be gated
behind ``require_approval`` and the HITL flow. This route only reads.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_registry
from app.tools.registry import ToolRegistry
from app.tools.task_manager import TaskManagerTool

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/list")
async def list_tasks(
    registry: Annotated[ToolRegistry, Depends(get_registry)],
) -> dict:
    """Return the live task list as structured JSON for the UI's grid."""
    tool = registry.get("task_manager")
    if not isinstance(tool, TaskManagerTool):
        raise HTTPException(status_code=500, detail="task_manager tool not registered.")
    tasks = await tool.list_raw()
    return {
        "tasks": tasks,
        "total": len(tasks),
        "open": sum(1 for t in tasks if not t["done"]),
        "done": sum(1 for t in tasks if t["done"]),
    }
