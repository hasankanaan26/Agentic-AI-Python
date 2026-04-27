# Instructor Notes — Checkpoint 4 (Orchestration & Observability)

**Duration:** ~2 hours
**Prerequisite:** CP3.

## Learning goals

- Build a **planner + executor** multi-agent pipeline.
- Use **structured outputs** (week 2 idea, applied here): the planner returns a Pydantic `AgentPlan` via `chat_model.with_structured_output(AgentPlan)` — provider-native, no parsing failures.
- See **three takes on the same problem** at three levels of abstraction:
  - CP2 `raw_loop.py` — every line of the agent loop, in pure Python.
  - CP3 `create_react_agent` — the same loop as a single LangGraph call.
  - CP4 orchestrator — a planner-then-executor pipeline that *coordinates* multiple agent runs.
- Add **streaming responses** with Server-Sent Events.
- Add **human-in-the-loop approval** for write tools, with `interrupt_before` available at the executor level and an explicit `awaiting_approval` gate at the orchestrator level.
- Use **LangSmith for production tracing**: free, automatic, no code change. Keep an in-memory `TraceStore` for the offline demo so `/traces` still works without an API key.

## Why this checkpoint exists

CP3 ended with a useful agent. CP4 answers the two questions a tech lead asks the day after it ships:

1. Why did the agent pick X at step 3? (observability — LangSmith)
2. Can it handle goals that need a deliberate plan rather than reactive tool-picking? (orchestration — the planner)

## What this checkpoint deliberately does NOT build

- A custom `BaseCallbackHandler` to capture executor events. LangSmith does this for free; for offline mode we use `astream_events()` directly. There is no `AgentTracer` class anymore.
- Hand-rolled HITL plumbing. We use LangGraph's built-in `interrupt_before=["tools"]` plus a small approval gate inside the orchestrator.
- A `permissions.py` module. Filtering tools by name is a one-line list comprehension; pulling it into a module made the abstraction look more important than it is.

This is the one checkpoint where "use the framework" beats "build it yourself" — the patterns being demonstrated *are* the framework's headline features.

## Live coding sequence

| Step | What | File |
|------|------|------|
| 1 | Define `PlanStep`, `AgentPlan`, `OrchestrationResponse`, `OrchestrateResumeRequest`. Show how `with_structured_output(AgentPlan)` removes JSON parsing entirely. | [`models/orchestration.py`](../checkpoints/checkpoint-4-orchestration/app/models/orchestration.py) |
| 2 | Walk through the planner prompt and the call: `await self._planner.ainvoke([...])` returns a fully-typed `AgentPlan`. | [`prompts/planner.py`](../checkpoints/checkpoint-4-orchestration/app/prompts/planner.py), [`services/orchestrator.py::_plan`](../checkpoints/checkpoint-4-orchestration/app/services/orchestrator.py) |
| 3 | Build `OrchestratorService.run()`. The pipeline is: optional safety gate → plan → for each step, run the LangGraph executor restricted to the picked tool → summarise. Read it top-to-bottom; every line is doing one thing. | [`services/orchestrator.py`](../checkpoints/checkpoint-4-orchestration/app/services/orchestrator.py) |
| 4 | Add the approval gate. When `require_approval=True` and the planner picked a write tool, snapshot the orchestration into `_pending` and return `status="awaiting_approval"` with the `trace_id` to resume on. | `OrchestratorService._execute` |
| 5 | Add `OrchestratorService.astream_events()`. Yields `planning_*`, `step_*`, and proxies the executor's `token` / `tool_start` / `tool_end` events. The route wraps this in `StreamingResponse`. | `OrchestratorService.astream_events`, [`routes/orchestrate.py`](../checkpoints/checkpoint-4-orchestration/app/routes/orchestrate.py) |
| 6 | Walk through `TraceStore`: bounded, async-locked, no callback handler. Then show how setting `LANGCHAIN_TRACING_V2=true` flips on full LangSmith capture for free. | [`services/tracer.py`](../checkpoints/checkpoint-4-orchestration/app/services/tracer.py) |
| 7 | Demo all three endpoints. | demo |

## Demo script

```bash
# 1. Auto mode — the original demo, end-to-end.
POST /orchestrate {"goal":"Find the remote work policy, calculate 15% of 230, and tell me the time"}
# Returns plan.steps[], execution_results[], trace_id, status="completed".

# 2. Streaming — same goal, watch the events arrive.
curl -N -X POST http://localhost:8005/orchestrate/stream \
  -H 'content-type: application/json' \
  -d '{"goal":"Find the remote work policy, calculate 15% of 230, and tell me the time"}'
# Stream of `data: {...}` SSE frames: trace, planning_*, step_*, token,
# tool_start, tool_end, summary, done.

# 3. Human-in-the-loop — pause before a write.
POST /orchestrate {"goal":"Create a task to onboard Alice","require_approval":true}
# Returns status="awaiting_approval", pending_thread_id=<trace_id>,
# pending_tool="task_manager".

POST /orchestrate/resume/<trace_id> {"approved":true}
# Continues the orchestration, returns status="completed".

# Try {"approved":false} on a fresh paused run — the step is recorded as
# rejected and the orchestration finishes the remaining steps anyway.

# 4. Drill into the trace (offline path; works without LangSmith).
GET /traces
GET /traces/<trace_id>
# Read out the entries: safety_check (if enabled), planning_start,
# planning_complete, execution_step_start, tool_call, execution_step_complete
# (or awaiting_approval / approval_granted if HITL was used).

# 5. With LangSmith on — set in .env:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=ls__...
# Restart, run the same orchestration, click into the LangSmith UI.
# Every LLM call, tool call, and node transition is captured automatically.

# 6. Compare with the reactive agent on the same goal:
POST /agent/run {"goal":"Find the remote work policy, calculate 15% of 230, and tell me the time"}
# The reactive agent often interleaves steps; the orchestrator commits
# to a plan first and then executes.
```

## Things to call out explicitly

- **Provider-native structured output, not prompt engineering.** `chat_model.with_structured_output(AgentPlan)` returns a fully validated Pydantic instance. The provider guarantees the response parses.
- **Per-step tool restriction is real safety.** Each plan step picks one tool; the executor is built with `allowed_tools=[step.tool_needed]`. Even if the LLM tries to call something else inside the executor, it can't reach it.
- **Approval gate is at the orchestrator, not the executor.** We could push approval into the executor's `interrupt_before` (and the runner still supports that), but the orchestrator-level gate is one `if step.tool_needed in WRITE_TOOLS:` — easier to read, easier to extend (e.g. require approval for tools touching specific records).
- **Streaming and HITL are on separate endpoints.** `/orchestrate/stream` is auto-mode only. Combining them would mean negotiating SSE pause/resume semantics, which is a distraction from the patterns we're teaching.
- **Async lock in the trace store.** Two concurrent orchestration runs would otherwise interleave entries. Show `async with self._lock`.
- **LangSmith vs the in-memory store are not redundant.** The in-memory store backs `/traces` so the demo works air-gapped. LangSmith is what you'd actually use in prod — and the only setup is two env vars.

## Common mistakes

| Mistake | Why it's wrong | Right answer |
|---------|----------------|--------------|
| Asking the planner to "return JSON" in the prompt | ~5% of responses fail to parse cleanly | `chat_model.with_structured_output(AgentPlan)` |
| Recording trace events with `print()` | Loses structure, can't query | `await store.add_entry(trace_id, action, detail)` |
| Letting one failed step abort the entire orchestration | Loses partial useful work | Catch per-step, record `execution_step_failed`, continue |
| Writing your own `BaseCallbackHandler` to capture LLM events | LangSmith already does this; for local capture, `astream_events()` is built in | Use the framework. Custom callbacks are the wrong layer to start at. |
| `_traces[trace_id] = ...` without the lock | Race condition under concurrent requests | `async with self._lock: ...` |
| Resuming a paused run with the wrong `trace_id` | The orchestrator can't find the snapshot, raises 404 | The `trace_id` in the response IS the resume key |

## Optional 5-minute extension: LangGraph Studio

LangGraph Studio is a visual debugger for any compiled LangGraph agent — the nodes, edges, current state, and pending interrupts, drawn as a graph in the browser. Useful for answering "why did the agent take this path?" without reading messages.

This is the *closest analogue* in the LangChain ecosystem to Microsoft Agent Framework's workflow studio (or Copilot Studio). The crucial difference: Studio is a **debugger** — you write the graph in Python, Studio visualises it. Drag-and-drop *builders* (LangFlow, Flowise) belong in a different conversation, ideally a later week.

### One-time setup

```bash
# In your project venv:
uv pip install "langgraph-cli[inmem]"
```

The repo ships a [`langgraph.json`](../langgraph.json) at the project root and a Studio entry point at [`app/agents/studio.py`](../checkpoints/checkpoint-4-orchestration/app/agents/studio.py) that exposes the executor graph (4 lightweight tools — no Chroma boot needed). `.env` must have a valid provider key (same as for `python run_checkpoint.py ...`).

### Run it

```bash
# From the project root, in a separate terminal:
langgraph dev
```

You'll see something like `Server: http://localhost:2024` and `Studio UI: https://smith.langchain.com/studio?baseUrl=http://localhost:2024`. Open the Studio URL — it's a hosted UI that talks to your local server.

### What to point at on screen (5 min)

1. **The graph itself** — top-left dropdown, pick `executor`. You'll see three nodes: `__start__`, `agent` (the LLM), `tools` (every tool call), with the conditional edge from `agent` going either to `tools` (if the LLM produced a tool call) or to `__end__` (if it produced a final answer). This IS the ReAct loop. Compare it to CP2's [`raw_loop.py`](../checkpoints/checkpoint-4-orchestration/app/agents/raw_loop.py) — same shape, drawn vs hand-coded.
2. **Run a goal** — paste `"Look up Bob in engineering and create a task to schedule a 1:1 with him"` into the input panel and hit Submit. Watch the highlighted node move: `agent` → `tools` (employee_lookup) → `agent` → `tools` (task_manager) → `agent` → `__end__`.
3. **State inspector** — right-hand panel shows the full message list at every step. Pick any step, see the exact `messages` array (system, human, AI with tool_calls, tool result, AI). This is what students would print from inside CP2's loop, but without the print statements.
4. **Time travel** — click any prior step, hit "Resume from here". The agent rewinds to that state and replays. Useful for "what if the tool had failed differently?" experiments.
5. **Interrupts** — Studio honours `interrupt_before`. Add `?force_interrupt_before=tools` to a fresh run via the UI's settings, and you'll see the graph pause before each tool call with an "Approve" button. (This is the same `interrupt_before=["tools"]` we use programmatically for HITL in the orchestrator.)

### The talking point

> "You wrote a Python file. The framework gave you a graph viewer, a state inspector, time travel, and human-in-the-loop approval — for free. This is what 'use the framework' actually buys you."

Skip if short on time; it's pure observability sugar, no new code concepts.

## Lab exercise (30 min)

Add a **POST /orchestrate/replan** endpoint. Given a `trace_id`, fetch the trace, replace the failed step with a new sub-plan from the planner (using the trace's prior context), and re-execute just that step. Append results to the same `trace_id`. Demonstrate it on a goal where step 2 normally fails.

## Wrap-up question (or capstone setup)

> "You now have a working agent that can plan, execute, retrieve, stream, pause for approval, and observe — running on async FastAPI with all the engineering rigour of a real service. Your capstone: build a domain-specific autonomous agent on top of this scaffold. IT helpdesk? Onboarding assistant? Expense approvals? Pick one, swap the tools, ship it."
