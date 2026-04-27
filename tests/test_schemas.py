"""Validate Pydantic boundary models with valid + invalid inputs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.agent import AgentRequest, SafeAgentRequest
from app.models.orchestration import AgentPlan, OrchestrationRequest, PlanStep
from app.models.tool import ToolResult


def test_agent_request_rejects_empty_goal():
    """An empty goal string is rejected at the boundary."""
    # Empty goals would let an HTTP client trigger an agent run with
    # nothing to do; the model should refuse it before any work starts.
    with pytest.raises(ValidationError):
        AgentRequest(goal="")


def test_agent_request_rejects_max_steps_too_high():
    """``max_steps`` above the configured ceiling is rejected."""
    # Hard ceiling protects against runaway loops / pathological costs.
    with pytest.raises(ValidationError):
        AgentRequest(goal="hi", max_steps=999)


def test_safe_agent_request_inherits_validation():
    """SafeAgentRequest accepts all parent fields and the allow-list."""
    req = SafeAgentRequest(goal="test", allowed_tools=["calculator"], max_steps=5)
    assert req.allowed_tools == ["calculator"]


def test_orchestration_request_validation():
    """OrchestrationRequest mirrors AgentRequest's empty-goal rule."""
    OrchestrationRequest(goal="do a thing")  # ok
    with pytest.raises(ValidationError):
        OrchestrationRequest(goal="")


def test_agent_plan_round_trip():
    """An AgentPlan survives a model_dump() round-trip with field names intact."""
    plan = AgentPlan(
        steps=[PlanStep(step_number=1, description="x", tool_needed="clock", reasoning="y")],
        overall_strategy="ok",
    )
    # If a future refactor renames ``tool_needed`` we need the
    # serialised JSON shape to break loudly so prompts can be updated.
    assert plan.model_dump()["steps"][0]["tool_needed"] == "clock"


def test_tool_result_factories():
    """``ToolResult.ok`` / ``.fail`` build correctly-shaped results."""
    ok = ToolResult.ok("hello", count=3)
    fail = ToolResult.fail("boom")
    assert ok.status == "ok"
    # Extra kwargs to ``ok`` are stashed in metadata, not at the top
    # level — keeps the public schema stable as we add new attributes.
    assert ok.metadata["count"] == 3
    assert fail.status == "error"
    assert fail.error == "boom"
