"""Planner + executor orchestration service.

The two-phase pattern this checkpoint introduces:

  1. **Planner** — an LLM call that returns a STRUCTURED `AgentPlan`
     (Pydantic) via `with_structured_output()`. Every plan field is
     validated; no JSON parsing failures.
  2. **Executor** — the LangGraph ReAct agent (CP3's `LangGraphAgentRunner`)
     run once per plan step, restricted to the tool the planner picked.

This file deliberately stays imperative and short. The interesting
control flow — "plan first, then loop the steps" — is something every
junior should be able to read top-to-bottom in one sitting.

Three things the orchestrator can do:

  - `run()` — blocking; supports the optional approval gate.
  - `astream_events()` — yields events for SSE; auto-mode only.
  - `resume()` — continues an orchestration paused by the approval gate.

Tracing: every interesting moment is pushed to the in-memory `TraceStore`
under a single `trace_id`. When `LANGCHAIN_TRACING_V2=true` is set,
LangSmith additionally captures every LangChain/LangGraph event for free.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.agents.langgraph import WRITE_TOOLS, LangGraphAgentRunner
from app.logging_config import get_logger
from app.models.orchestration import AgentPlan, ExecutionStepResult
from app.prompts import EXECUTOR_PROMPT_TEMPLATE, PLANNER_PROMPT
from app.services.safety import check_prompt_injection
from app.services.tracer import TraceStore
from app.settings import Settings

log = get_logger(__name__)


@dataclass
class _Pending:
    """Snapshot of an orchestration paused at the approval gate.

    Stored in :attr:`OrchestratorService._pending` keyed by ``trace_id`` so
    ``resume()`` can pick the work up exactly where it stopped.
    """

    goal: str
    plan: AgentPlan
    completed: list[ExecutionStepResult]
    next_step_index: int
    allowed_tools: list[str] | None


class OrchestratorService:
    """Plan-then-execute orchestrator built on the LangGraph ReAct runner.

    Wraps three behaviours:

    * :meth:`run` -- blocking plan + execute, with optional approval gate.
    * :meth:`astream_events` -- same workflow, yields SSE-shaped events.
    * :meth:`resume` -- continue an orchestration paused on an approval gate.

    The structured planner is built once via ``with_structured_output`` so
    every plan call returns a validated :class:`AgentPlan`.
    """

    def __init__(
        self,
        *,
        chat_model,
        runner: LangGraphAgentRunner,
        traces: TraceStore,
        settings: Settings,
    ) -> None:
        # ``with_structured_output`` binds AgentPlan as the response schema.
        # We keep the original ``chat_model`` reference inside it; no extra HTTP client.
        self._planner = chat_model.with_structured_output(AgentPlan)
        self._runner = runner
        self._traces = traces
        self._settings = settings
        self._pending: dict[str, _Pending] = {}  # keyed by trace_id

    # ---------------------------------------------------------------- run

    async def run(
        self,
        *,
        goal: str,
        allowed_tools: list[str] | None = None,
        require_approval: bool = False,
    ) -> dict[str, Any]:
        """Plan and execute ``goal`` end-to-end.

        Args:
            goal: Natural-language goal from the user.
            allowed_tools: Optional whitelist; ``None`` means all tools.
            require_approval: If true, pause before any write tool runs.

        Returns:
            A dict shaped like :class:`app.models.OrchestrationResponse`.
        """
        trace_id = await self._traces.create(goal)
        try:
            # Safety gate runs first; if it blocks, we never plan or execute.
            if blocked := await self._safety_block(trace_id, goal):
                return blocked
            plan = await self._plan(trace_id, goal)
            return await self._execute(
                trace_id=trace_id,
                goal=goal,
                plan=plan,
                completed=[],
                start_index=0,
                allowed_tools=allowed_tools,
                require_approval=require_approval,
            )
        except Exception as e:  # noqa: BLE001
            # Catch-all so an unexpected provider blowup still closes the trace cleanly.
            log.exception("orchestration_failed")
            await self._traces.complete(trace_id, status="error")
            return _error_response(goal, trace_id, str(e))

    # ------------------------------------------------------------- resume

    async def resume(self, *, trace_id: str, approved: bool) -> dict[str, Any]:
        """Continue (or skip) the step that was waiting on operator approval.

        Args:
            trace_id: The trace id returned in the prior ``awaiting_approval``
                response.
            approved: True to execute the paused step, false to record it as
                rejected and move on.

        Raises:
            KeyError: If no pending orchestration exists for ``trace_id``.
        """
        pending = self._pending.pop(trace_id, None)
        if pending is None:
            raise KeyError(f"No pending orchestration for trace_id={trace_id!r}.")

        step = pending.plan.steps[pending.next_step_index]
        await self._traces.add_entry(
            trace_id,
            "approval_granted" if approved else "approval_rejected",
            {"step": step.step_number, "tool": step.tool_needed},
        )

        if not approved:
            # Operator declined: record the step as rejected and resume from the next one.
            pending.completed.append(
                ExecutionStepResult(
                    step_number=step.step_number,
                    description=step.description,
                    tool_used=step.tool_needed,
                    result="Rejected by operator.",
                    status="rejected",
                )
            )
            return await self._execute(
                trace_id=trace_id,
                goal=pending.goal,
                plan=pending.plan,
                completed=pending.completed,
                start_index=pending.next_step_index + 1,
                allowed_tools=pending.allowed_tools,
                require_approval=True,
            )

        # Approved: run the paused step (this one only, no re-approval), then continue.
        await self._run_step(
            trace_id=trace_id,
            step=step,
            allowed_tools=pending.allowed_tools,
            results=pending.completed,
        )
        return await self._execute(
            trace_id=trace_id,
            goal=pending.goal,
            plan=pending.plan,
            completed=pending.completed,
            start_index=pending.next_step_index + 1,
            allowed_tools=pending.allowed_tools,
            require_approval=True,
        )

    # ----------------------------------------------------- streaming (SSE)

    async def astream_events(
        self,
        *,
        goal: str,
        allowed_tools: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield event dicts as the orchestration runs. Auto-mode only.

        Sequence:
            {type: "trace", trace_id}
            {type: "planning_start"}
            {type: "planning_complete", strategy, steps}
            (per step)
                {type: "step_start", step, description, tool}
                {type: "token" | "tool_start" | "tool_end", ...}   (proxied from executor)
                {type: "step_complete", step, result, status}
            {type: "summary", final_summary, status}
            {type: "done"}
        """
        trace_id = await self._traces.create(goal)
        yield {"type": "trace", "trace_id": trace_id}
        try:
            if blocked := await self._safety_block(trace_id, goal):
                yield {"type": "summary", "final_summary": blocked["final_summary"], "status": "blocked"}
                yield {"type": "done"}
                return

            yield {"type": "planning_start"}
            plan = await self._plan(trace_id, goal)
            yield {
                "type": "planning_complete",
                "strategy": plan.overall_strategy,
                "steps": [s.model_dump() for s in plan.steps],
            }

            results: list[ExecutionStepResult] = []
            for step in plan.steps:
                yield {
                    "type": "step_start",
                    "step": step.step_number,
                    "description": step.description,
                    "tool": step.tool_needed,
                }
                step_allowed = _resolve_step_tools(step.tool_needed, allowed_tools)
                step_goal = EXECUTOR_PROMPT_TEMPLATE.format(
                    step_number=step.step_number,
                    description=step.description,
                    tool_needed=step.tool_needed,
                )
                final_answer: str | None = None
                async for ev in self._runner.astream_events(
                    goal=step_goal,
                    allowed_tools=step_allowed,
                    require_approval=False,
                ):
                    if ev["type"] == "done":
                        final_answer = ev.get("final_answer")
                    else:
                        yield ev
                results.append(
                    ExecutionStepResult(
                        step_number=step.step_number,
                        description=step.description,
                        tool_used=step.tool_needed,
                        result=final_answer or "No result",
                        status="completed",
                    )
                )
                yield {
                    "type": "step_complete",
                    "step": step.step_number,
                    "status": "completed",
                    "result": final_answer,
                }

            summary = _summarise(results)
            await self._traces.complete(trace_id, status="completed")
            yield {"type": "summary", "final_summary": summary, "status": "completed"}
        except Exception as e:  # noqa: BLE001
            log.exception("orchestration_stream_failed")
            await self._traces.complete(trace_id, status="error")
            yield {"type": "summary", "final_summary": f"Orchestration failed: {e}", "status": "error"}
        finally:
            yield {"type": "done"}

    # ---------------------------------------------------------- internals

    async def _safety_block(self, trace_id: str, goal: str) -> dict | None:
        """Run the heuristic safety check and return a blocked response on hit."""
        if not self._settings.enable_injection_detection:
            return None
        check = check_prompt_injection(goal)
        await self._traces.add_entry(
            trace_id,
            "safety_check",
            {"flagged": check["flagged"], "risk_level": check["risk_level"]},
        )
        if not check["flagged"]:
            return None
        # Close the trace with status="blocked" so /traces shows the outcome.
        await self._traces.complete(trace_id, status="blocked")
        return {
            "goal": goal,
            "plan": None,
            "execution_results": [],
            "final_summary": "Request blocked: potential prompt injection detected.",
            "trace_id": trace_id,
            "status": "blocked",
            "pending_thread_id": None,
            "pending_step": None,
            "pending_tool": None,
        }

    async def _plan(self, trace_id: str, goal: str) -> AgentPlan:
        """Call the structured planner and record start/end markers in the trace."""
        await self._traces.add_entry(trace_id, "planning_start", {"goal": goal})
        # ``with_structured_output(AgentPlan)`` returns a validated AgentPlan.
        # No JSON parsing or fallback handling needed -- if the LLM produces
        # invalid JSON, the underlying client raises and we hit the run() catch.
        plan: AgentPlan = await self._planner.ainvoke(
            [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": f"Create a plan to accomplish this goal: {goal}"},
            ]
        )
        await self._traces.add_entry(
            trace_id,
            "planning_complete",
            {"steps": len(plan.steps), "strategy": plan.overall_strategy},
        )
        log.info("orchestrator_plan_ready", steps=len(plan.steps), trace_id=trace_id)
        return plan

    async def _execute(
        self,
        *,
        trace_id: str,
        goal: str,
        plan: AgentPlan,
        completed: list[ExecutionStepResult],
        start_index: int,
        allowed_tools: list[str] | None,
        require_approval: bool,
    ) -> dict[str, Any]:
        """Execute steps starting at ``start_index``; pause on approval if needed.

        Mutates ``completed`` in-place so ``resume()`` can continue using the
        same accumulator across calls.
        """
        for i in range(start_index, len(plan.steps)):
            step = plan.steps[i]
            # Approval gate: stash a snapshot and return early if this step writes.
            if require_approval and step.tool_needed in WRITE_TOOLS:
                self._pending[trace_id] = _Pending(
                    goal=goal,
                    plan=plan,
                    completed=completed,
                    next_step_index=i,
                    allowed_tools=allowed_tools,
                )
                await self._traces.add_entry(
                    trace_id,
                    "awaiting_approval",
                    {"step": step.step_number, "tool": step.tool_needed},
                )
                summary = (
                    f"Paused before step {step.step_number}: awaiting approval to "
                    f"call '{step.tool_needed}'. POST /orchestrate/resume/{trace_id} "
                    f"with {{\"approved\": true}} to continue."
                )
                return _build_response(
                    trace_id=trace_id,
                    goal=goal,
                    plan=plan,
                    results=completed,
                    summary=summary,
                    status="awaiting_approval",
                    pending_thread_id=trace_id,
                    pending_step=step.step_number,
                    pending_tool=step.tool_needed,
                )
            await self._run_step(
                trace_id=trace_id,
                step=step,
                allowed_tools=allowed_tools,
                results=completed,
            )

        # All steps done. Stamp the trace and return.
        summary = _summarise(completed)
        await self._traces.complete(trace_id, status="completed")
        return _build_response(
            trace_id=trace_id,
            goal=goal,
            plan=plan,
            results=completed,
            summary=summary,
            status="completed",
        )

    async def _run_step(
        self,
        *,
        trace_id: str,
        step,
        allowed_tools: list[str] | None,
        results: list[ExecutionStepResult],
    ) -> None:
        """Run one plan step on the LangGraph executor, recording trace events."""
        await self._traces.add_entry(
            trace_id,
            "execution_step_start",
            {"step": step.step_number, "description": step.description, "tool": step.tool_needed},
        )
        # Constrain the executor to the planner's chosen tool when possible so
        # the LLM doesn't go shopping during a step.
        step_allowed = _resolve_step_tools(step.tool_needed, allowed_tools)
        step_goal = EXECUTOR_PROMPT_TEMPLATE.format(
            step_number=step.step_number,
            description=step.description,
            tool_needed=step.tool_needed,
        )
        try:
            step_result = await self._runner.run(
                goal=step_goal,
                allowed_tools=step_allowed,
                require_approval=False,
            )
            # Mirror each tool call into the trace for the /traces endpoint.
            for s in step_result.get("steps", []):
                await self._traces.add_entry(
                    trace_id,
                    "tool_call",
                    {
                        "step": step.step_number,
                        "name": s["tool_name"],
                        "input": s["tool_input"],
                        # Truncate to keep trace storage bounded.
                        "output": str(s["tool_output"])[:500],
                    },
                )
            results.append(
                ExecutionStepResult(
                    step_number=step.step_number,
                    description=step.description,
                    tool_used=step.tool_needed,
                    result=step_result.get("final_answer") or "No result",
                    status="completed",
                )
            )
            await self._traces.add_entry(
                trace_id,
                "execution_step_complete",
                {"step": step.step_number, "status": "completed"},
            )
        except Exception as e:  # noqa: BLE001
            # One failed step doesn't fail the orchestration -- record and move on.
            log.exception("orchestrator_step_failed", step=step.step_number)
            results.append(
                ExecutionStepResult(
                    step_number=step.step_number,
                    description=step.description,
                    tool_used=step.tool_needed,
                    result=f"Failed: {e}",
                    status="failed",
                )
            )
            await self._traces.add_entry(
                trace_id,
                "execution_step_failed",
                {"step": step.step_number, "error": str(e)},
            )


# ----------------------------------------------------------------- helpers


def _resolve_step_tools(needed: str | None, allowed: list[str] | None) -> list[str] | None:
    """Narrow the executor's tool whitelist to the single tool the planner picked.

    Falls back to the caller-provided ``allowed`` if ``needed`` would conflict
    with the user's whitelist (e.g. planner picked something not on it).
    """
    if needed and (allowed is None or needed in allowed):
        return [needed]
    return allowed


def _summarise(results: list[ExecutionStepResult]) -> str:
    """Build a one-line human summary from per-step statuses."""
    done = sum(1 for r in results if r.status == "completed")
    rejected = sum(1 for r in results if r.status == "rejected")
    failed = sum(1 for r in results if r.status == "failed")
    parts = [f"Completed {done}/{len(results)} plan steps."]
    if rejected:
        parts.append(f"{rejected} rejected.")
    if failed:
        parts.append(f"{failed} failed.")
    return " ".join(parts)


def _build_response(
    *,
    trace_id: str,
    goal: str,
    plan: AgentPlan,
    results: list[ExecutionStepResult],
    summary: str,
    status: str,
    pending_thread_id: str | None = None,
    pending_step: int | None = None,
    pending_tool: str | None = None,
) -> dict[str, Any]:
    """Build the dict that ``/orchestrate`` and ``/orchestrate/resume`` return."""
    return {
        "goal": goal,
        "plan": {
            "strategy": plan.overall_strategy,
            "steps": [s.model_dump() for s in plan.steps],
        },
        "execution_results": results,
        "final_summary": summary,
        "trace_id": trace_id,
        "status": status,
        "pending_thread_id": pending_thread_id,
        "pending_step": pending_step,
        "pending_tool": pending_tool,
    }


def _error_response(goal: str, trace_id: str, err: str) -> dict[str, Any]:
    """Shape an error result that the response model can still validate."""
    return {
        "goal": goal,
        "plan": None,
        "execution_results": [],
        "final_summary": f"Orchestration failed: {err}",
        "trace_id": trace_id,
        "status": "error",
        "pending_thread_id": None,
        "pending_step": None,
        "pending_tool": None,
    }
