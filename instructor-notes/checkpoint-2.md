# Instructor Notes — Checkpoint 2 (Agent Loop)

**Duration:** ~2 hours
**Prerequisite:** CP1.

## Learning goals

- See the agent loop pattern in ~80 lines of pure Python.
- Understand why we async/await everything in an I/O-bound system.
- Internalise that **tool failures are data, not exceptions** — the LLM should reason about them.
- Add stateful tools (task_manager) and discover the concurrency questions they raise.

## Why this checkpoint exists

Frameworks (LangGraph, AutoGen, CrewAI) hide the loop. Hide it too early and students think it's magic. We build it from scratch *once*, with all the warts (state management, message history, stopping conditions), then in CP3 we hand the same problem to LangGraph and they appreciate what it bought.

## Live coding sequence

| Step | What | File |
|------|------|------|
| 1 | Add `task_manager` and `employee_lookup` tools. Make sure both are async. | [`tools/task_manager.py`](../checkpoints/checkpoint-2-agent-loop/app/tools/task_manager.py), [`tools/employee_lookup.py`](../checkpoints/checkpoint-2-agent-loop/app/tools/employee_lookup.py) |
| 2 | Write the agent loop. `messages = [...]; while ...; if finish_reason == "stop": break`. | [`agents/raw_loop.py`](../checkpoints/checkpoint-2-agent-loop/app/agents/raw_loop.py) |
| 3 | Wire `/agent/run`. | [`routes/agent.py`](../checkpoints/checkpoint-2-agent-loop/app/routes/agent.py) |
| 4 | Demo a multi-tool goal. Read the `steps` array out loud — call out each tool input and output. | demo |
| 5 | Force a tool failure (e.g. `task_manager` with no title). Show that the LLM tries again rather than 500ing. | demo |

## Demo script

```bash
# Two tools chained:
POST /agent/run  {"goal":"Look up Bob in engineering, then create a task to schedule a 1:1 with him"}

# A goal where the LLM has to retry after a structured error:
POST /agent/run  {"goal":"Mark task 999 as done, then list everything"}
# Step 1: tool returns `tool_status=error` (no task with ID 999)
# Step 2: LLM observes the error, calls `list` instead, summarises.
```

## Things to call out explicitly

- **The whole stack is async.** Walk through one request from `/agent/run` -> `run_agent()` -> `LLMService.call_with_tools()` -> back. Find every `await`.
- **`while step_num in range(max_steps): ... else:`** — Python's loop-else handles "we ran out of steps without finishing".
- **The conversation history grows.** Each tool call appends both an assistant turn and a user turn. Show a 3-step run and count the messages.
- **`asyncio.Lock` in task_manager.** Two concurrent `create` calls would otherwise race.
- **TTL caching in employee_lookup.** Run the same lookup twice; the second response includes `metadata.cached=true`. Discuss when caching is worth it (deterministic, expensive, low churn) and when it's a footgun (per-user data, fast-moving APIs).

## Common mistakes

| Mistake | Why it's wrong | Right answer |
|---------|----------------|--------------|
| `time.sleep` between retries | Blocks the event loop | `await asyncio.sleep`, or let tenacity handle it |
| Using `requests.post` inside a tool | Blocks the event loop | `async with httpx.AsyncClient(): ...` |
| Tool raises on bad arguments | Crashes the agent run | Return `ToolResult.fail(...)` — the registry already wraps `TypeError` for us |
| Re-implementing the registry inside the loop | Duplicates execution path | The loop calls `registry.execute(name, args)` |
| No `max_steps` bound | Infinite loop on a confused LLM | Always bound the loop |

## Lab exercise (20 min)

Make `task_manager` actually persist to disk. Convert the in-memory list to JSON writes. Mind the `asyncio.Lock` so two concurrent writes don't corrupt the file. Add a test that runs `create` 100x concurrently and asserts every task is present in the file.

## Wrap-up question

> "Your agent works, but `knowledge_search` doesn't exist yet — and if we
> built it with keyword counting, `how many days off do I get?` would
> miss `Annual Leave`. We also have no way to gate write tools behind
> approval. Next: safety, the LangGraph rebuild, and proper RAG."
