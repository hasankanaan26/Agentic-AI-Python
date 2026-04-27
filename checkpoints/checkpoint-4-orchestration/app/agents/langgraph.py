"""LangGraph ReAct agent — the same agent, reimplemented with LangGraph.

LangGraph gives us, for free:
  - state persistence per thread_id (MemorySaver / SqliteSaver),
  - human-in-the-loop via interrupt_before=["tools"],
  - streaming via astream_events,
  - first-class observability (LangSmith auto-instruments LangGraph nodes
    when LANGCHAIN_TRACING_V2=true).

What we still own:
  - the chat-model factory (`build_chat_model`),
  - the tool list (built from our async registry),
  - permission filtering (one-line list comprehension).
"""

from __future__ import annotations

import itertools
import os
from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.logging_config import get_logger
from app.prompts import SAFETY_SYSTEM_PROMPT
from app.settings import Settings
from app.tools.langchain_tools import build_langchain_tools
from app.tools.registry import ToolRegistry

log = get_logger(__name__)

_thread_seq = itertools.count(1)

# Write tools must be approved when require_approval=True. Read tools never
# pause. Keep this list small and explicit — it's the safety contract.
WRITE_TOOLS: set[str] = {"task_manager"}


def _checkpointer():
    """Pick a LangGraph checkpointer based on env config.

    Uses Sqlite when ``CHECKPOINT_DB_PATH`` is set and the saver is
    importable; otherwise falls back to an in-memory saver. Sqlite gives
    durability across process restarts -- handy for production HITL flows.
    """
    db_path = os.getenv("CHECKPOINT_DB_PATH", "")
    if db_path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            log.info("checkpointer_sqlite", path=db_path)
            return SqliteSaver.from_conn_string(db_path)
        except ImportError:
            # Sqlite saver lives in an optional extra; warn but keep going.
            log.warning("sqlite_saver_missing_falling_back_to_memory")
    return MemorySaver()


def build_chat_model(settings: Settings):
    """Construct the LangChain chat model that matches ``settings.llm_provider``.

    Lazy imports keep optional packages out of the import graph. Same
    provider switch as :class:`app.services.llm.LLMService`, but using
    LangChain's chat model wrappers because LangGraph wants those.
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
    """LangGraph-based ReAct agent runner.

    One instance per process is built in lifespan. Wraps everything you
    need to run, resume, and stream a single LangGraph agent: chat model,
    LangChain-shaped tools, and a checkpointer keyed by ``thread_id``.
    """

    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._llm = build_chat_model(settings)
        # Wrap our async tools so LangGraph can schedule them.
        self._lc_tools = build_langchain_tools(registry)
        self._checkpointer = _checkpointer()

    @property
    def chat_model(self):
        """Expose the underlying chat model so callers (e.g. the orchestrator's
        planner) can reuse one HTTP client per process."""
        return self._llm

    def _build_agent(self, *, allowed_tools: list[str] | None, require_approval: bool):
        """Compile a fresh ReAct graph with the requested tool subset/HITL flag."""
        tools = (
            self._lc_tools
            if allowed_tools is None
            else [t for t in self._lc_tools if t.name in allowed_tools]
        )
        # interrupt_before=["tools"] is LangGraph's HITL primitive: the
        # graph pauses after the LLM emits tool_calls but before they execute.
        interrupt_before = ["tools"] if require_approval else None
        return create_react_agent(
            self._llm,
            tools=tools,
            prompt=SAFETY_SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
            interrupt_before=interrupt_before,
        )

    @staticmethod
    def _new_thread_id() -> str:
        """Allocate a process-unique thread id using the module-level counter."""
        return f"thread_{next(_thread_seq)}"

    @staticmethod
    def _config(thread_id: str, max_steps: int, callbacks: list | None) -> dict[str, Any]:
        """Build the LangGraph runtime config dict for a single invocation."""
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            # ``recursion_limit`` is per-edge; multiply by 2 since each step is
            # a model->tools edge plus a tools->model edge.
            "recursion_limit": max_steps * 2,
        }
        if callbacks:
            config["callbacks"] = callbacks
        return config

    @staticmethod
    def _shape_response(
        *,
        goal: str,
        messages: list,
        thread_id: str,
        model: str,
        final_answer: str | None,
        status: str = "completed",
        pending_tool: str | None = None,
    ) -> dict[str, Any]:
        """Convert a LangGraph message list into our :class:`AgentResponse` shape."""
        # LangGraph's message list interleaves AIMessage (may contain tool_calls)
        # and ToolMessage (the result for the prior call). We pair them up.
        steps: list[dict] = []
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
            elif getattr(msg, "type", None) == "tool" and steps:
                # Tool message comes immediately after the AIMessage that called it.
                steps[-1]["tool_output"] = str(msg.content)
        return {
            "goal": goal,
            "steps": steps,
            "final_answer": final_answer,
            "steps_completed": len(steps),
            "thread_id": thread_id,
            "model": model,
            "engine": "langgraph",
            "status": status,
            "pending_tool": pending_tool,
        }

    async def run(
        self,
        *,
        goal: str,
        allowed_tools: list[str] | None = None,
        require_approval: bool = False,
        max_steps: int | None = None,
        callbacks: list | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute the agent for ``goal`` and return a response dict.

        Args:
            goal: User goal.
            allowed_tools: Optional whitelist; ``None`` means all tools.
            require_approval: If true, pause before executing tool calls.
            max_steps: Override for ``settings.max_agent_steps``.
            callbacks: Optional LangChain callback handlers (e.g. a tracer).
            thread_id: Reuse an existing thread id; otherwise a fresh one is allocated.

        Returns:
            A dict shaped like :class:`app.models.LangGraphAgentResponse`.
            ``status`` will be ``"awaiting_approval"`` if the run paused on
            an approval gate, otherwise ``"completed"`` or ``"error"``.
        """
        max_steps = max_steps or self._settings.max_agent_steps
        thread_id = thread_id or self._new_thread_id()
        agent = self._build_agent(allowed_tools=allowed_tools, require_approval=require_approval)
        config = self._config(thread_id, max_steps, callbacks)
        log.info("langgraph_run", thread_id=thread_id, goal=goal[:100])

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": goal}]}, config=config
            )
            messages = result.get("messages", [])
            # Inspect the graph state to see if we're paused for approval.
            status, pending_tool = await self._inspect_pause(agent, config, messages)
            final_answer = _last_ai_text(messages) if status == "completed" else None
        except Exception as e:  # noqa: BLE001
            # Catch-all so a provider blowup still returns a structured 200 with
            # status="error" instead of crashing the route.
            log.exception("langgraph_run_failed", error=str(e))
            return self._shape_response(
                goal=goal,
                messages=[],
                thread_id=thread_id,
                model=self._settings.model_name(),
                final_answer=f"Agent encountered an error: {e}",
                status="error",
            )

        return self._shape_response(
            goal=goal,
            messages=messages,
            thread_id=thread_id,
            model=self._settings.model_name(),
            final_answer=final_answer,
            status=status,
            pending_tool=pending_tool,
        )

    async def resume(
        self,
        *,
        thread_id: str,
        approved: bool,
        allowed_tools: list[str] | None = None,
        max_steps: int | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """Continue (or reject) a run that was paused on the approval gate.

        Args:
            thread_id: Thread id from the prior ``awaiting_approval`` response.
            approved: True to execute the pending tool, false to abort with
                ``status="rejected"``.
            allowed_tools: Same whitelist semantics as :meth:`run`.
            max_steps: Optional override.
            callbacks: Optional LangChain callbacks.

        Returns:
            The same response shape as :meth:`run`.
        """
        max_steps = max_steps or self._settings.max_agent_steps
        agent = self._build_agent(allowed_tools=allowed_tools, require_approval=True)
        config = self._config(thread_id, max_steps, callbacks)

        # Pull the prior conversation out of the checkpointer so we can echo
        # the original goal back in the response shape.
        snapshot = await agent.aget_state(config)
        goal = _first_human_text(snapshot.values.get("messages", [])) or ""

        if not approved:
            log.info("langgraph_resume_rejected", thread_id=thread_id)
            return self._shape_response(
                goal=goal,
                messages=snapshot.values.get("messages", []),
                thread_id=thread_id,
                model=self._settings.model_name(),
                final_answer="Tool call was rejected by the operator. No changes were made.",
                status="rejected",
            )

        log.info("langgraph_resume_approved", thread_id=thread_id)
        # Passing None continues from the checkpoint (LangGraph standard pattern).
        result = await agent.ainvoke(None, config=config)
        messages = result.get("messages", [])
        # Could pause again at the next write tool; re-inspect.
        status, pending_tool = await self._inspect_pause(agent, config, messages)
        final_answer = _last_ai_text(messages) if status == "completed" else None
        return self._shape_response(
            goal=goal,
            messages=messages,
            thread_id=thread_id,
            model=self._settings.model_name(),
            final_answer=final_answer,
            status=status,
            pending_tool=pending_tool,
        )

    async def astream_events(
        self,
        *,
        goal: str,
        allowed_tools: list[str] | None = None,
        require_approval: bool = False,
        max_steps: int | None = None,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield event dicts (suitable for SSE) as the agent runs.

        Translates LangGraph's verbose ``astream_events`` output into the
        compact event vocabulary the orchestrator and ``/orchestrate/stream``
        expose: ``token``, ``tool_start``, ``tool_end``, ``done``.
        """
        max_steps = max_steps or self._settings.max_agent_steps
        thread_id = thread_id or self._new_thread_id()
        agent = self._build_agent(allowed_tools=allowed_tools, require_approval=require_approval)
        config = self._config(thread_id, max_steps, callbacks=None)

        # ``version="v2"`` is the stable event schema.
        async for ev in agent.astream_events(
            {"messages": [{"role": "user", "content": goal}]},
            config=config,
            version="v2",
        ):
            kind = ev.get("event")
            if kind == "on_chat_model_stream":
                # Token-by-token streaming from the LLM.
                chunk = ev["data"]["chunk"]
                text = getattr(chunk, "content", "") or ""
                if text:
                    yield {"type": "token", "thread_id": thread_id, "text": text}
            elif kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "thread_id": thread_id,
                    "name": ev.get("name"),
                    "input": ev.get("data", {}).get("input"),
                }
            elif kind == "on_tool_end":
                yield {
                    "type": "tool_end",
                    "thread_id": thread_id,
                    "name": ev.get("name"),
                    # Truncate to keep streamed payloads bounded.
                    "output": str(ev.get("data", {}).get("output", ""))[:500],
                }

        # After the stream finishes, fetch the final state to learn whether
        # we're paused or done.
        snapshot = await agent.aget_state(config)
        messages = snapshot.values.get("messages", [])
        status, pending_tool = await self._inspect_pause(agent, config, messages)
        final_answer = _last_ai_text(messages) if status == "completed" else None
        yield {
            "type": "done",
            "thread_id": thread_id,
            "status": status,
            "pending_tool": pending_tool,
            "final_answer": final_answer,
        }

    @staticmethod
    async def _inspect_pause(agent, config, messages) -> tuple[str, str | None]:
        """Return ('awaiting_approval', tool_name) if paused, else ('completed', None)."""
        snapshot = await agent.aget_state(config)
        # ``snapshot.next`` is empty only when the graph has finished.
        if not snapshot.next:
            return "completed", None
        # Pending tool name lives on the most recent AIMessage's tool_calls.
        for msg in reversed(messages):
            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                return "awaiting_approval", tcs[0].get("name")
        return "awaiting_approval", None


def _last_ai_text(messages: list) -> str | None:
    """Return the most recent AIMessage content, or ``None`` if there isn't one."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and msg.content:
            return msg.content
    return None


def _first_human_text(messages: list) -> str | None:
    """Return the first HumanMessage content (used to recover the original goal)."""
    for msg in messages:
        if getattr(msg, "type", None) == "human" and msg.content:
            return msg.content
    return None
