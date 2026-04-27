"""The from-scratch agent loop, async.

Kept in the codebase deliberately — it's the simplest possible agent
written without LangChain so engineers reading the code can see the
control flow before opening `langgraph.py`.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.prompts import AGENT_LOOP_PROMPT
from app.services.llm import LLMService
from app.tools.registry import ToolRegistry

log = get_logger(__name__)


async def run_agent(
    *,
    goal: str,
    llm: LLMService,
    registry: ToolRegistry,
    tool_names: list[str] | None = None,
    max_steps: int = 10,
) -> dict:
    """Iterate think -> act -> observe until the LLM returns a final answer.

    On each iteration the LLM either:

    1. Picks one or more tool calls, in which case we execute them,
       append the observations to the conversation, and loop, OR
    2. Emits a final natural-language answer (``finish_reason == "stop"``),
       in which case we exit and return the result.

    Args:
        goal: The user's goal in natural language.
        llm: Singleton LLM service used for each step.
        registry: Tool registry providing schemas + execution.
        tool_names: Optional allow-list to restrict tool exposure.
        max_steps: Maximum think/act/observe iterations before bailing.

    Returns:
        A dict matching ``AgentResponse``: goal, steps[], final_answer,
        steps_completed, model.
    """
    tools = registry.definitions(allowed=tool_names)
    # Conversation state shared across iterations; grows by two messages
    # per tool call (assistant intent + user observation).
    messages = [{"role": "user", "content": goal}]
    steps: list[dict] = []
    final_answer: str | None = None

    for step_num in range(max_steps):
        log.info("agent_step_llm_call", step=step_num + 1, tools=len(tools))
        response = await llm.call_with_tools(
            messages=messages, tools=tools, system_prompt=AGENT_LOOP_PROMPT
        )

        # The model decided to answer directly; we're done.
        if response["finish_reason"] == "stop":
            final_answer = response["response_text"]
            break

        # Otherwise the model wants one or more tool calls; run them all
        # before looping back for the next think step.
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]

            # Tool errors are returned as ``ToolResult(status="error", ...)``
            # — they never raise. The error becomes another observation
            # the LLM can react to in the next iteration.
            result = await registry.execute(tool_name, tool_args)
            log.info(
                "agent_step_tool_done",
                step=step_num + 1,
                tool=tool_name,
                status=result.status,
            )
            steps.append(
                {
                    "step": step_num + 1,
                    "tool_name": tool_name,
                    "tool_input": tool_args,
                    "tool_output": result.output,
                    "tool_status": result.status,
                }
            )

            # Feed the result back as a turn the LLM can read. We use plain
            # role=assistant/user instead of OpenAI's tool-call role so this
            # works uniformly across providers.
            messages.append({"role": "assistant", "content": f"Calling tool: {tool_name}"})
            messages.append(
                {"role": "user", "content": f"Tool '{tool_name}' returned: {result.output}"}
            )
    else:
        # ``for`` loop ran to ``max_steps`` without ``break`` — agent didn't
        # converge. Return a clear placeholder so the caller doesn't see ``None``.
        final_answer = "Agent reached maximum steps without completing the goal."

    return {
        "goal": goal,
        "steps": steps,
        "final_answer": final_answer,
        "steps_completed": len(steps),
        "model": llm.model_name,
    }
