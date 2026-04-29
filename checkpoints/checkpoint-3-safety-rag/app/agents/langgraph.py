"""LangGraph ReAct agent — the same agent, reimplemented with LangGraph.

LangGraph gives us, for free:
  - state persistence per thread_id (MemorySaver / SqliteSaver),
  - human-in-the-loop via ``interrupt_before=["tools"]``,
  - streaming and consistent message handling,
  - first-class callbacks (used by AgentTracer).

What we still own:
  - the chat-model factory (``build_chat_model``),
  - the tool list (built from our async registry),
  - permission filtering before the agent is constructed,
  - a small in-memory record of paused threads so ``resume()`` knows what
    ``allowed_tools`` to rebuild the agent with.
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
    """Pick a checkpointer based on env: SQLite if configured, else in-memory."""
    db_path = os.getenv("CHECKPOINT_DB_PATH", "")
    if db_path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            log.info("checkpointer_sqlite", path=db_path)
            return SqliteSaver.from_conn_string(db_path)
        except ImportError:
            log.warning("sqlite_saver_missing_falling_back_to_memory")
    return MemorySaver()


def build_chat_model(settings: Settings):
    """Construct the LangChain chat model matching ``settings.llm_provider``."""
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


def _reconstruct_steps(messages: list) -> list[dict]:
    """Turn the LangGraph message log into our flat ``steps[]`` shape.

    Each ``AIMessage.tool_calls`` entry produces a step; the matching
    ``ToolMessage`` (correlated by ``tool_call_id``) fills in the output
    and status. ``ToolMessage.status`` is set to ``"error"`` by LangChain
    when our wrapper raises ``ToolException``.
    """
    steps: list[dict] = []
    step_num = 0
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                step_num += 1
                steps.append(
                    {
                        "step": step_num,
                        "tool_call_id": tc.get("id"),
                        "tool_name": tc["name"],
                        "tool_input": tc["args"],
                        "tool_output": "",
                        "tool_status": "ok",
                    }
                )
        elif getattr(msg, "type", None) == "tool":
            target = next(
                (s for s in steps if s.get("tool_call_id") == msg.tool_call_id),
                steps[-1] if steps else None,
            )
            if target is not None:
                target["tool_output"] = str(msg.content)
                target["tool_status"] = (
                    "error" if getattr(msg, "status", "success") == "error" else "ok"
                )
    for s in steps:
        s.pop("tool_call_id", None)
    return steps


def _final_answer(messages: list) -> str | None:
    """Return the last non-empty AI message content, or ``None``."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            return msg.content
    return None


def _pending_tool_call(messages: list) -> dict | None:
    """If the run paused at ``interrupt_before=["tools"]``, return the proposed call.

    The signature of an interrupt is: the last message is an ``AIMessage``
    with non-empty ``tool_calls`` and no ``ToolMessage`` follows it.
    """
    if not messages:
        return None
    last = messages[-1]
    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return None
    tc = last.tool_calls[0]
    return {"name": tc["name"], "args": tc["args"], "id": tc.get("id")}


def _format_message(msg) -> dict:
    """Compact, JSON-safe view of a LangChain message."""
    out: dict[str, Any] = {
        "type": getattr(msg, "type", "unknown"),
        "content": str(getattr(msg, "content", "")),
    }
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        out["tool_calls"] = [
            {"name": tc["name"], "args": tc["args"], "id": tc.get("id")}
            for tc in msg.tool_calls
        ]
    if getattr(msg, "tool_call_id", None):
        out["tool_call_id"] = msg.tool_call_id
    if getattr(msg, "name", None) and out["type"] == "tool":
        out["name"] = msg.name
    if getattr(msg, "status", None) == "error":
        out["status"] = "error"
    return out


def _format_snapshot(snap) -> dict:
    """Project a LangGraph ``StateSnapshot`` into a JSON-safe dict for the UI."""
    md = snap.metadata or {}
    cfg = snap.config or {}
    cp_id = cfg.get("configurable", {}).get("checkpoint_id")
    parent_cfg = snap.parent_config or {}
    parent_id = parent_cfg.get("configurable", {}).get("checkpoint_id")
    messages = (snap.values or {}).get("messages", []) if snap.values else []
    writes = md.get("writes") or {}
    return {
        "checkpoint_id": cp_id,
        "parent_checkpoint_id": parent_id,
        "step": md.get("step"),
        "source": md.get("source"),
        "next": list(snap.next) if snap.next else [],
        # Just the node names that wrote at this step — full payload is huge.
        "writes": list(writes.keys()) if writes else [],
        "created_at": str(snap.created_at) if snap.created_at else None,
        "messages": [_format_message(m) for m in messages],
        "message_count": len(messages),
    }


class LangGraphAgentRunner:
    """Wraps a LangGraph ReAct agent with per-request configuration.

    One instance per process; created in lifespan. Holds a tiny in-memory
    record of paused threads (``_pending``) so ``resume`` can rebuild the
    agent with the same ``allowed_tools`` / ``max_steps`` as the original
    request — without that, an approved tool call could re-pause or hit a
    different tool list.
    """

    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._llm = build_chat_model(settings)
        self._lc_tools = build_langchain_tools(registry)
        self._checkpointer = _checkpointer()
        # thread_id -> {"goal", "allowed_tools", "max_steps"} for resume().
        self._pending: dict[str, dict[str, Any]] = {}

    def _build_agent(self, *, allowed_tools: list[str] | None, require_approval: bool):
        """Build a fresh ReAct agent graph for this request."""
        tools = (
            self._lc_tools
            if allowed_tools is None
            else [t for t in self._lc_tools if t.name in allowed_tools]
        )
        interrupt_before = ["tools"] if require_approval else None
        return create_react_agent(
            self._llm,
            tools=tools,
            prompt=SAFETY_SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
            interrupt_before=interrupt_before,
        )

    def pending_threads(self) -> list[dict]:
        """Snapshot of paused threads — used by the UI's HITL inbox."""
        return [
            {"thread_id": tid, **{k: v for k, v in ctx.items() if k != "max_steps"}}
            for tid, ctx in self._pending.items()
        ]

    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        """Return the current state plus full checkpoint history for a thread.

        Reads come straight from the checkpointer attached to the runner —
        any agent built with the same checkpointer can read any thread it
        has touched, regardless of the original ``allowed_tools``.
        """
        agent = self._build_agent(allowed_tools=None, require_approval=False)
        config = {"configurable": {"thread_id": thread_id}}
        current = await agent.aget_state(config)
        if current is None or not current.values:
            raise KeyError(thread_id)

        history: list[dict] = []
        async for snap in agent.aget_state_history(config):
            history.append(_format_snapshot(snap))

        ctx = self._pending.get(thread_id)
        return {
            "thread_id": thread_id,
            "current": _format_snapshot(current),
            "history": history,
            "is_paused": ctx is not None,
            "pending_context": (
                {"goal": ctx["goal"], "allowed_tools": ctx.get("allowed_tools")}
                if ctx else None
            ),
        }

    async def run(
        self,
        *,
        goal: str,
        allowed_tools: list[str] | None = None,
        require_approval: bool = False,
        max_steps: int | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """Execute the agent against ``goal`` and return a serializable result."""
        max_steps = max_steps or self._settings.max_agent_steps
        thread_id = f"thread_{next(_thread_seq)}"
        agent = self._build_agent(
            allowed_tools=allowed_tools, require_approval=require_approval
        )
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max_steps * 2,
        }
        if callbacks:
            config["callbacks"] = callbacks

        log.info("langgraph_run", thread_id=thread_id, goal=goal[:100])

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": goal}]}, config=config
            )
            messages = result.get("messages", [])
            steps = _reconstruct_steps(messages)
            final = _final_answer(messages)
            proposed = _pending_tool_call(messages) if require_approval else None

            if proposed is not None:
                # Stash enough context for resume() to rebuild the same agent.
                self._pending[thread_id] = {
                    "goal": goal,
                    "allowed_tools": allowed_tools,
                    "max_steps": max_steps,
                }
                status = "paused"
                # The agent hasn't said anything final yet — null out the
                # placeholder so the UI doesn't render it as "done".
                final = None
            else:
                status = "completed"
        except Exception as e:
            log.exception("langgraph_run_failed", error=str(e))
            steps, final, proposed, status = [], f"Agent encountered an error: {e}", None, "completed"

        return {
            "goal": goal,
            "steps": steps,
            "final_answer": final,
            "steps_completed": len(steps),
            "thread_id": thread_id,
            "model": self._settings.model_name(),
            "engine": "langgraph",
            "status": status,
            "proposed_tool_call": proposed,
        }

    async def resume(self, *, thread_id: str, approved: bool) -> dict[str, Any]:
        """Continue or abort a paused thread.

        On approve, we build a fresh graph WITHOUT ``interrupt_before`` so
        the run goes to completion; LangGraph picks up from the checkpoint
        because the thread_id is the same. On reject we discard the pending
        record and return a minimal response so the UI can close the loop.
        """
        ctx = self._pending.pop(thread_id, None)
        if ctx is None:
            raise KeyError(thread_id)

        goal = ctx["goal"]
        if not approved:
            log.info("langgraph_resume_rejected", thread_id=thread_id)
            return {
                "goal": goal,
                "steps": [],
                "final_answer": "Run rejected by reviewer; the proposed tool call was not executed.",
                "steps_completed": 0,
                "thread_id": thread_id,
                "model": self._settings.model_name(),
                "engine": "langgraph",
                "status": "rejected",
                "proposed_tool_call": None,
            }

        # Build a fresh agent with the SAME allow-list but interrupts off.
        agent = self._build_agent(
            allowed_tools=ctx["allowed_tools"], require_approval=False
        )
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": ctx["max_steps"] * 2,
        }
        log.info("langgraph_resume", thread_id=thread_id)

        try:
            # ``ainvoke(None, ...)`` resumes from the checkpoint without
            # appending a new user turn — the state already has the pending
            # AIMessage with tool_calls; LangGraph runs the tool node next.
            result = await agent.ainvoke(None, config=config)
            messages = result.get("messages", [])
            steps = _reconstruct_steps(messages)
            final = _final_answer(messages)
            status = "completed"
        except Exception as e:
            log.exception("langgraph_resume_failed", error=str(e))
            steps, final, status = [], f"Resume error: {e}", "completed"

        return {
            "goal": goal,
            "steps": steps,
            "final_answer": final,
            "steps_completed": len(steps),
            "thread_id": thread_id,
            "model": self._settings.model_name(),
            "engine": "langgraph",
            "status": status,
            "proposed_tool_call": None,
        }
