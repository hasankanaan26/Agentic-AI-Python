"""LangGraph ReAct agent — the same agent, reimplemented with LangGraph.

LangGraph gives us, for free:
  - state persistence per thread_id (MemorySaver / SqliteSaver),
  - human-in-the-loop via interrupt_before=["tools"],
  - streaming and consistent message handling,
  - first-class callbacks (used by AgentTracer).

What we still own:
  - the chat-model factory (`build_chat_model`),
  - the tool list (built from our async registry),
  - permission filtering before the agent is constructed.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.logging_config import get_logger
from app.prompts import SAFETY_SYSTEM_PROMPT
from app.settings import Settings
from app.tools.langchain_tools import build_langchain_tools
from app.tools.registry import ToolRegistry

log = get_logger(__name__)

# Process-wide monotonic counter used to generate readable thread IDs.
# Each /agent/run call gets a fresh ``thread_<n>`` for checkpointing.
_thread_seq = itertools.count(1)


def _checkpointer():
    """Pick a checkpointer based on env: SQLite if configured, else in-memory.

    LangGraph persists agent state per ``thread_id`` so a future request can
    resume mid-run (used by the human-in-the-loop approval flow). For dev
    we keep state in memory; production sets ``CHECKPOINT_DB_PATH`` to
    persist across restarts.
    """
    db_path = os.getenv("CHECKPOINT_DB_PATH", "")
    if db_path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            log.info("checkpointer_sqlite", path=db_path)
            return SqliteSaver.from_conn_string(db_path)
        except ImportError:
            # Optional dep not installed — fall back rather than crash boot.
            log.warning("sqlite_saver_missing_falling_back_to_memory")
    return MemorySaver()


def build_chat_model(settings: Settings):
    """Construct the LangChain chat model matching ``settings.llm_provider``.

    LangGraph drives a LangChain chat model rather than the bare SDK clients
    used by ``LLMService``; this factory hides the per-provider construction.
    """
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.3,
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        )
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.3,
    )


class LangGraphAgentRunner:
    """Wraps a LangGraph ReAct agent with per-request configuration.

    One instance per process; created in lifespan. The chat model, tool list,
    and checkpointer are constructed once and reused. ``run`` builds a fresh
    agent graph for each request because the allowed-tool list and
    interrupt-before-tools flag can vary per call.
    """

    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._llm = build_chat_model(settings)
        # LangChain's StructuredTool wrappers around our registry — built
        # once because the underlying tool implementations don't change.
        self._lc_tools = build_langchain_tools(registry)
        self._checkpointer = _checkpointer()

    def _build_agent(self, *, allowed_tools: list[str] | None, require_approval: bool):
        """Build a fresh ReAct agent graph for this request.

        Args:
            allowed_tools: Names to expose to the LLM. ``None`` exposes all.
            require_approval: When true, the graph pauses before invoking
                any tool so a human can approve via the resume API.
        """
        # Tool gating happens here, before construction — the LLM never sees
        # the disallowed tools, so it cannot invoke them.
        tools = (
            self._lc_tools
            if allowed_tools is None
            else [t for t in self._lc_tools if t.name in allowed_tools]
        )
        # ``interrupt_before=["tools"]`` is LangGraph's hook for HITL approval.
        interrupt_before = ["tools"] if require_approval else None
        return create_react_agent(
            self._llm,
            tools=tools,
            prompt=SAFETY_SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
            interrupt_before=interrupt_before,
        )

    async def run(
        self,
        *,
        goal: str,
        allowed_tools: list[str] | None = None,
        require_approval: bool = False,
        max_steps: int | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """Execute the agent against ``goal`` and return a serializable result.

        Args:
            goal: User's goal in natural language.
            allowed_tools: Optional allow-list of tool names.
            require_approval: Pause before each tool call when true.
            max_steps: Override of ``settings.max_agent_steps``.
            callbacks: Optional LangChain callback handlers (tracing, etc.).

        Returns:
            A dict matching ``LangGraphAgentResponse``: goal, steps[],
            final_answer, steps_completed, thread_id, model, engine.
        """
        max_steps = max_steps or self._settings.max_agent_steps
        # Thread IDs partition checkpointer state — each run gets its own.
        thread_id = f"thread_{next(_thread_seq)}"

        agent = self._build_agent(allowed_tools=allowed_tools, require_approval=require_approval)
        # ``recursion_limit`` is a LangGraph safety net independent of our
        # logical max_steps; doubled because each step counts as multiple
        # graph node visits (agent -> tools -> agent).
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max_steps * 2,
        }
        if callbacks:
            config["callbacks"] = callbacks

        log.info("langgraph_run", thread_id=thread_id, goal=goal[:100])

        steps: list[dict] = []
        final_answer: str | None = None
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": goal}]}, config=config
            )
            messages = result.get("messages", [])

            # Reconstruct a step list from the LangGraph message log: each
            # AIMessage with tool_calls produces N steps, the next ToolMessage
            # fills in tool_output for the most recently appended step.
            step_num = 0
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step_num += 1
                        steps.append(
                            {
                                "step": step_num,
                                "tool_name": tc["name"],
                                "tool_input": tc["args"],
                                "tool_output": "",
                                "tool_status": "ok",
                            }
                        )
                elif msg.type == "tool" and steps:
                    steps[-1]["tool_output"] = str(msg.content)

            # The final answer is the last AI message with non-empty content.
            for msg in reversed(messages):
                if msg.type == "ai" and msg.content:
                    final_answer = msg.content
                    break

        except Exception as e:
            # Catch-all so a buggy tool / LLM glitch never crashes the route;
            # surface the error string in the final_answer for the caller.
            log.exception("langgraph_run_failed", error=str(e))
            final_answer = f"Agent encountered an error: {e}"

        return {
            "goal": goal,
            "steps": steps,
            "final_answer": final_answer,
            "steps_completed": len(steps),
            "thread_id": thread_id,
            "model": self._settings.model_name(),
            "engine": "langgraph",
        }
