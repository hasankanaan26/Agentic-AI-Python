"""End-to-end test of the agent loop with a fake LLM.

The fake LLM returns scripted tool calls and a final answer. Together
with real tool implementations this exercises the entire control flow
without burning a real API key.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.raw_loop import run_agent
from app.models.tool import ToolResult


class FakeLLM:
    """Drop-in stand-in for `LLMService` used by the raw agent loop.

    The class mimics only the surface area the agent loop actually
    touches (``model_name`` attribute and ``call_with_tools`` coroutine).
    Each entry in the ``scripted`` list is the LLM response for one
    iteration of the loop — tests pre-load the script to cover specific
    branches (tool call, tool error, final answer, etc.).
    """

    model_name = "fake-llm-1"

    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        # Copy the script so callers can reuse the original list across
        # tests without seeing it mutated by ``pop(0)``.
        self._script = list(scripted)

    async def call_with_tools(self, messages, tools, system_prompt):  # noqa: ARG002
        """Return the next scripted response, or a default 'no script' stop.

        The signature mirrors the real LLM service. ``noqa: ARG002`` is
        applied because we deliberately ignore ``messages``/``tools``/
        ``system_prompt`` — the script encodes the desired behaviour.
        """
        if not self._script:
            # Defensive fallback: if the test under-specified the script,
            # we end the loop cleanly instead of raising IndexError.
            return {"response_text": "no script left", "tool_calls": [], "finish_reason": "stop"}
        return self._script.pop(0)


class FakeRegistry:
    """Tool registry stub that records calls and answers a hardcoded case.

    The agent loop expects a registry exposing ``definitions()`` (for the
    LLM's tool schema) and ``execute()`` (to run a chosen tool). This
    fake records every call so tests can assert *what* the agent invoked
    and *with which arguments*.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def definitions(self, allowed=None):  # noqa: ARG002
        """Return a single-tool catalogue. ``allowed`` is ignored on purpose."""
        return [
            {
                "name": "calculator",
                "description": "math",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Record the call and return a hardcoded success or failure."""
        self.calls.append((name, arguments))
        # Only a very specific input is treated as a "happy path"; any
        # other shape returns a structured failure to exercise the
        # error-handling branch of the agent loop.
        if name == "calculator" and arguments.get("a") == 3 and arguments.get("b") == 4:
            return ToolResult.ok("3 + 4 = 7")
        return ToolResult.fail(f"unsupported call: {name} {arguments}")


@pytest.mark.asyncio
async def test_agent_runs_one_tool_then_answers():
    """Agent calls the calculator once, then returns the LLM's final answer."""
    # Two-step script: (1) ask to call the calculator, (2) answer based
    # on the tool result. Mirrors the textbook ReAct trajectory.
    llm = FakeLLM(
        [
            {
                "response_text": None,
                "tool_calls": [
                    {"name": "calculator", "arguments": {"operation": "add", "a": 3, "b": 4}}
                ],
                "finish_reason": "tool_calls",
            },
            {
                "response_text": "The answer is 7.",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        ]
    )
    registry = FakeRegistry()

    result = await run_agent(goal="What is 3+4?", llm=llm, registry=registry, max_steps=5)

    assert result["final_answer"] == "The answer is 7."
    # ``steps_completed`` should be 1 because exactly one tool call ran;
    # the second LLM turn produced the final answer (not a step).
    assert result["steps_completed"] == 1
    # And the registry must have seen exactly the call the LLM scripted.
    assert registry.calls == [("calculator", {"operation": "add", "a": 3, "b": 4})]


@pytest.mark.asyncio
async def test_agent_surfaces_tool_error_to_llm():
    """When a tool returns ToolResult.fail, the loop continues; it does NOT raise."""
    # The first scripted call uses arguments the FakeRegistry doesn't
    # support, forcing a ToolResult.fail. The loop must hand that error
    # back to the LLM rather than aborting with an exception.
    llm = FakeLLM(
        [
            {
                "response_text": None,
                "tool_calls": [
                    {"name": "calculator", "arguments": {"operation": "x", "a": 1, "b": 2}}
                ],
                "finish_reason": "tool_calls",
            },
            {
                "response_text": "I couldn't compute that.",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        ]
    )
    result = await run_agent(goal="x", llm=llm, registry=FakeRegistry(), max_steps=5)
    assert result["final_answer"] == "I couldn't compute that."
    # The recorded step trace should preserve the failure so observability
    # / debugging surfaces it instead of silently swallowing the error.
    assert result["steps"][0]["tool_status"] == "error"
