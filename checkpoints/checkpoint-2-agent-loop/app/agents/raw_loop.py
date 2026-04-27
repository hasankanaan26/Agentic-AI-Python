"""The from-scratch agent loop, async.

Kept in the codebase deliberately — it's the simplest possible agent
written without LangChain so engineers reading the code can see the
control flow before opening ``langgraph.py``.

The pattern is the canonical "ReAct"-style loop:

    1. Send the conversation + tool catalog to the LLM.
    2. If the LLM returns a final answer, stop.
    3. Otherwise execute the requested tool(s), append the results back
       into the message list, and loop again.

Bounded by ``max_steps`` so a confused LLM cannot spin forever.
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

    Args:
        goal: Natural-language objective from the user.
        llm: Shared :class:`LLMService` instance (injected by the route).
        registry: :class:`ToolRegistry` exposing the available tools.
        tool_names: Optional whitelist; when set, only these tools are
            advertised to the LLM. ``None`` means "all tools".
        max_steps: Hard cap on the number of think/act iterations. If the
            agent doesn't produce a final answer within this budget we
            return a graceful fallback message instead of raising.

    Returns:
        A dict containing the original goal, every tool step taken, the
        final answer (if produced), the count of completed steps, and the
        LLM model identifier. Shape matches :class:`AgentResponse`.
    """
    # Filter the tool catalogue and seed the conversation with the user's goal.
    tools = registry.definitions(allowed=tool_names)
    messages = [{"role": "user", "content": goal}]
    steps: list[dict] = []
    final_answer: str | None = None

    # ``for/else``: the ``else`` branch fires only if the loop completed
    # all iterations without ``break`` — i.e. we hit the step budget.
    for step_num in range(max_steps):
        log.info("agent_step_llm_call", step=step_num + 1, tools=len(tools))
        response = await llm.call_with_tools(
            messages=messages, tools=tools, system_prompt=AGENT_LOOP_PROMPT
        )

        # finish_reason="stop" means the model is done thinking and gave a final answer.
        if response["finish_reason"] == "stop":
            final_answer = response["response_text"]
            break

        # Otherwise we have one or more tool calls to execute this step.
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]

            # ``registry.execute`` never raises — it always returns a ToolResult.
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

            # Feed the result back as a turn the LLM can read.
            # NOTE: a more idiomatic implementation would use provider-specific
            # tool-result message types; we use plain user/assistant text here
            # for portability across all three providers.
            messages.append({"role": "assistant", "content": f"Calling tool: {tool_name}"})
            messages.append(
                {"role": "user", "content": f"Tool '{tool_name}' returned: {result.output}"}
            )
    else:
        # Step budget exhausted without the LLM ever returning "stop".
        final_answer = "Agent reached maximum steps without completing the goal."

    return {
        "goal": goal,
        "steps": steps,
        "final_answer": final_answer,
        "steps_completed": len(steps),
        "model": llm.model_name,
    }
