"""Task manager — a WRITE-capable, multi-action tool.

Permissions: classified as ``"write"`` because ``create`` and
``complete`` mutate state. The LangGraph agent can be configured to
interrupt before any write tool runs (``require_approval=True``).

The backing store is a JSON file loaded lazily on first use; mutations
are kept in memory and serialized only on flush (intentionally not
implemented in CP2 — the data resets between processes, which is the
desired behaviour for the demo).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

from app.models.tool import ToolResult
from app.tools.base import BaseTool


class TaskManagerTool(BaseTool):
    """List/create/complete/search tasks backed by a JSON seed file."""

    name: ClassVar[str] = "task_manager"
    permission: ClassVar[str] = "write"
    definition: ClassVar[dict[str, Any]] = {
        "name": "task_manager",
        "description": (
            "Manage tasks: list all tasks, create new ones, mark them complete, or "
            "search by keyword. Use this when the user wants to see, add, finish, "
            "or find tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "complete", "search"],
                    "description": "Which action to run.",
                },
                "title": {"type": "string", "description": "Required for 'create'."},
                "task_id": {"type": "integer", "description": "Required for 'complete'."},
                "query": {"type": "string", "description": "Required for 'search'."},
            },
            "required": ["action"],
        },
    }

    def __init__(self, tasks_path: Path) -> None:
        """Initialize the tool with the path to the seed JSON file.

        Args:
            tasks_path: Filesystem path to the tasks JSON used to bootstrap state.
        """
        self._path = tasks_path
        # All mutations go through this lock so two concurrent agent steps
        # can't race on the in-memory list (e.g., create + complete).
        self._lock = asyncio.Lock()
        self._tasks: list[dict] | None = None

    async def _load(self) -> list[dict]:
        """Lazily load tasks from disk on first access."""
        if self._tasks is None:
            self._tasks = json.loads(self._path.read_text(encoding="utf-8"))
        return self._tasks

    async def run(
        self,
        action: str,
        title: str | None = None,
        task_id: int | None = None,
        query: str | None = None,
    ) -> ToolResult:
        """Dispatch on ``action`` and run the requested sub-operation.

        Args:
            action: One of ``"list"``, ``"create"``, ``"complete"``, ``"search"``.
            title: Required when ``action == "create"``.
            task_id: Required when ``action == "complete"``.
            query: Required when ``action == "search"``.

        Returns:
            A :class:`ToolResult`. Validation problems (missing argument,
            unknown action, unknown id) come back as ``ToolResult.fail``
            so the LLM can recover by calling again with the right input.
        """
        async with self._lock:
            tasks = await self._load()
            if action == "list":
                return ToolResult.ok(_list_tasks(tasks))
            if action == "create":
                if not title:
                    return ToolResult.fail("A title is required to create a task.")
                return ToolResult.ok(_create_task(tasks, title))
            if action == "complete":
                if task_id is None:
                    return ToolResult.fail("A task_id is required to complete a task.")
                ok, msg = _complete_task(tasks, task_id)
                # Translate the (ok, msg) tuple back into a ToolResult.
                return ToolResult.ok(msg) if ok else ToolResult.fail(msg)
            if action == "search":
                if not query:
                    return ToolResult.fail("A query is required to search tasks.")
                return ToolResult.ok(_search_tasks(tasks, query))
            return ToolResult.fail(f"Unknown action '{action}'.")


def _list_tasks(tasks: list[dict]) -> str:
    """Render every task as ``[id] [STATUS] title``."""
    if not tasks:
        return "No tasks found."
    lines = [f"[{t['id']}] [{'DONE' if t['done'] else 'TODO'}] {t['title']}" for t in tasks]
    return f"Tasks ({len(tasks)} total):\n" + "\n".join(lines)


def _create_task(tasks: list[dict], title: str) -> str:
    """Append a new task with the next monotonically-increasing id."""
    # ``default=0`` handles the empty-list case so we still hand out id=1.
    new_id = max((t["id"] for t in tasks), default=0) + 1
    tasks.append({"id": new_id, "title": title, "done": False})
    return f"Created task [{new_id}]: {title}"


def _complete_task(tasks: list[dict], task_id: int) -> tuple[bool, str]:
    """Mark a task done; return ``(success, message)``."""
    for task in tasks:
        if task["id"] == task_id:
            if task["done"]:
                # Idempotent: completing an already-done task is still success.
                return True, f"Task [{task_id}] is already complete: {task['title']}"
            task["done"] = True
            return True, f"Completed task [{task_id}]: {task['title']}"
    return False, f"No task found with ID {task_id}."


def _search_tasks(tasks: list[dict], query: str) -> str:
    """Substring-match titles (case-insensitive) and render matches."""
    q = query.lower()
    matches = [t for t in tasks if q in t["title"].lower()]
    if not matches:
        return f"No tasks found matching '{query}'."
    lines = [f"[{t['id']}] [{'DONE' if t['done'] else 'TODO'}] {t['title']}" for t in matches]
    return f"Found {len(matches)} task(s) matching '{query}':\n" + "\n".join(lines)
