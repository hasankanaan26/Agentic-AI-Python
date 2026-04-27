"""Each tool, exercised in isolation. No LLM, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.tools.calculator import CalculatorTool
from app.tools.clock import ClockTool
from app.tools.employee_lookup import EmployeeLookupTool
from app.tools.task_manager import TaskManagerTool


@pytest.mark.asyncio
async def test_calculator_happy_path():
    """Calculator returns a successful ToolResult containing the product."""
    tool = CalculatorTool()
    out = await tool.run(operation="multiply", a=147, b=23)
    assert out.status == "ok"
    # 147 * 23 = 3381 — assert the number appears in the rendered output
    # rather than pinning the exact format string.
    assert "3381" in out.output


@pytest.mark.asyncio
async def test_calculator_returns_structured_error_on_divide_by_zero():
    """Divide-by-zero is reported as a ToolResult error, not an exception."""
    tool = CalculatorTool()
    out = await tool.run(operation="divide", a=10, b=0)
    # Tools must NEVER raise into the agent loop; they return status="error"
    # so the LLM gets to see the failure and react.
    assert out.status == "error"
    assert "zero" in out.error.lower()


@pytest.mark.asyncio
async def test_calculator_unknown_op_is_structured_error():
    """An unsupported operation is rejected with a structured error."""
    tool = CalculatorTool()
    out = await tool.run(operation="exponentiate", a=2, b=3)
    assert out.status == "error"


@pytest.mark.asyncio
async def test_clock_returns_text():
    """ClockTool's date format produces a human-readable success string."""
    out = await ClockTool().run(format="date")
    assert out.status == "ok"
    assert "Current date" in out.output


@pytest.mark.asyncio
async def test_employee_lookup_caches_repeats():
    """Repeated identical queries are served from the in-memory cache."""
    # Tight TTL/size are fine here — the second call happens immediately.
    tool = EmployeeLookupTool(cache_ttl=60, cache_max=8)
    first = await tool.run(query="engineering")
    second = await tool.run(query="engineering")
    assert first.status == "ok"
    # The cache hit is signalled via metadata so callers / traces can
    # distinguish a cached response from a fresh one.
    assert second.metadata.get("cached") is True


@pytest.mark.asyncio
async def test_employee_lookup_rejects_short_query():
    """Single-character queries are rejected to avoid useless scans."""
    out = await EmployeeLookupTool().run(query="a")
    assert out.status == "error"


@pytest.mark.asyncio
async def test_task_manager_create_then_list(tmp_path: Path):
    """A created task appears in the subsequent list call."""
    # ``tmp_path`` is a per-test temp dir, so each test gets a fresh
    # task store and cannot pollute the others.
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps([]), encoding="utf-8")
    tool = TaskManagerTool(tasks_path=tasks_file)

    created = await tool.run(action="create", title="Review onboarding doc")
    assert created.status == "ok"
    # The created task is the first one, so its ID renders as ``[1]``.
    assert "[1]" in created.output

    listed = await tool.run(action="list")
    assert "Review onboarding doc" in listed.output


@pytest.mark.asyncio
async def test_task_manager_complete_unknown_id(tmp_path: Path):
    """Completing a non-existent task ID surfaces a structured error."""
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps([]), encoding="utf-8")
    tool = TaskManagerTool(tasks_path=tasks_file)

    out = await tool.run(action="complete", task_id=999)
    assert out.status == "error"
