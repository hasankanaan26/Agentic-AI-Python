"""Models for the planner+executor orchestration pipeline.

`AgentPlan` is the structured output the planner produces via
`call_llm_structured()` — provider-native JSON, schema-validated.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One step the planner emits (parsed straight out of structured output)."""

    step_number: int = Field(..., ge=1, description="1-based ordinal of this step in the plan.")
    description: str = Field(..., description="What this step accomplishes, in natural language.")
    tool_needed: str = Field(..., description="Name of the tool the executor should run.")
    reasoning: str = Field(..., description="Why this step is necessary toward the goal.")


class AgentPlan(BaseModel):
    """The planner's contract. Returned via provider-native structured output.

    ``with_structured_output(AgentPlan)`` validates the LLM's JSON against
    this schema, so executors downstream can trust the shape.
    """

    steps: list[PlanStep] = Field(..., description="Ordered list of plan steps.")
    overall_strategy: str = Field(..., description="One-paragraph strategy summary.")


class ExecutionStepResult(BaseModel):
    """Outcome of executing a single :class:`PlanStep`.

    ``status`` is one of ``"completed"``, ``"failed"``, or ``"rejected"``
    (operator declined the approval gate).
    """

    step_number: int = Field(..., description="Matches the originating ``PlanStep.step_number``.")
    description: str = Field(..., description="Echo of the plan step description.")
    tool_used: str = Field(..., description="Tool name that actually ran (or was rejected).")
    result: str = Field(..., description="Final answer/string the executor returned.")
    status: str = Field(..., description="``completed`` | ``failed`` | ``rejected``.")


class OrchestrationRequest(BaseModel):
    """Body for ``POST /orchestrate`` and ``POST /orchestrate/stream``."""

    goal: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language goal to plan + execute.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="If set, restrict executor steps to this tool whitelist.",
    )
    require_approval: bool = Field(
        default=False,
        description="Pause before any write tool runs; resume via /orchestrate/resume.",
    )


class OrchestrationResponse(BaseModel):
    """Response shape for the blocking and resume orchestrate endpoints.

    The ``pending_*`` fields are populated only when ``status ==
    "awaiting_approval"`` -- the caller uses them to decide what to confirm
    via ``POST /orchestrate/resume/{trace_id}``.
    """

    goal: str = Field(..., description="Echo of the input goal.")
    plan: dict[str, Any] | None = Field(
        ..., description="Serialised :class:`AgentPlan`, or ``None`` if planning was skipped."
    )
    execution_results: list[ExecutionStepResult] = Field(
        ..., description="Per-step outcomes, in plan order."
    )
    final_summary: str = Field(..., description="Human-readable run summary.")
    trace_id: str = Field(..., description="In-memory trace id for the /traces endpoints.")
    status: str = Field(
        ...,
        description="``completed`` | ``awaiting_approval`` | ``blocked`` | ``error``.",
    )
    # Populated only when status == "awaiting_approval".
    pending_thread_id: str | None = Field(
        default=None, description="Trace id to pass to /orchestrate/resume."
    )
    pending_step: int | None = Field(
        default=None, description="Step number waiting on approval."
    )
    pending_tool: str | None = Field(
        default=None, description="Tool that requires approval before it can run."
    )


class OrchestrateResumeRequest(BaseModel):
    """Body for ``POST /orchestrate/resume/{thread_id}``."""

    approved: bool = Field(
        ..., description="True to run the paused step, false to skip it as ``rejected``."
    )
