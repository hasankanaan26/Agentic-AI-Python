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

    The whole agent fits in this one function on purpose -- it's the
    canonical pedagogical example of "what does an agent loop actually do
    if you don't pull in a framework?".

    Args:
        goal: Natural-language goal from the user.
        llm: Async LLM service that supports ``call_with_tools``.
        registry: Tool registry used to execute LLM-chosen tool calls.
        tool_names: Optional whitelist; ``None`` means all tools.
        max_steps: Hard cap on iterations to prevent runaway loops.

    Returns:
        ``{goal, steps, final_answer, steps_completed, model}`` -- shaped to
        validate as :class:`app.models.AgentResponse`.
    """
    tools = registry.definitions(allowed=tool_names)
    messages = [{"role": "user", "content": goal}]
    steps: list[dict] = []
    final_answer: str | None = None

    # The classic ReAct loop: each iteration is one LLM turn possibly
    # followed by tool calls. Break when the LLM signals "stop".
    for step_num in range(max_steps):
        log.info("agent_step_llm_call", step=step_num + 1, tools=len(tools))
        response = await llm.call_with_tools(
            messages=messages, tools=tools, system_prompt=AGENT_LOOP_PROMPT
        )

        # Terminal condition: LLM returned a plain answer instead of tool calls.
        if response["finish_reason"] == "stop":
            final_answer = response["response_text"]
            break

        # Otherwise execute every tool the LLM asked for in this turn.
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]

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
            # Note: we use plain user/assistant messages (rather than
            # provider-specific tool-result messages) so this loop stays
            # provider-agnostic and easy to read.
            messages.append({"role": "assistant", "content": f"Calling tool: {tool_name}"})
            messages.append(
                {"role": "user", "content": f"Tool '{tool_name}' returned: {result.output}"}
            )
    else:
        # ``for/else`` -- only runs if the loop finishes without ``break``.
        final_answer = "Agent reached maximum steps without completing the goal."

    return {
        "goal": goal,
        "steps": steps,
        "final_answer": final_answer,
        "steps_completed": len(steps),
        "model": llm.model_name,
    }
