# Checkpoint 2 — The Agent Loop

Adds the from-scratch async agent loop. The LLM can now chain tool calls until the goal is met. Tool errors become structured data the LLM reasons about, not exceptions that crash the run.

## What's new

| Layer | File(s) | Why |
|-------|---------|-----|
| Async agent loop | [`app/agents/raw_loop.py`](app/agents/raw_loop.py) | `think -> act -> observe`, awaitable, no LangChain. |
| Two more tools | [`task_manager.py`](app/tools/task_manager.py), [`employee_lookup.py`](app/tools/employee_lookup.py) | A write-capable tool and a read-capable directory. |
| TTL caching | inside `employee_lookup.py` | Same lookup within the TTL doesn't re-run; demonstrates `cachetools.TTLCache`. |
| Structured tool errors throughout | (already in registry from CP1) | The new `task_manager` tool surfaces `ToolResult.fail("title required ...")` cleanly. |
| Concurrency-safe state | `task_manager.py:asyncio.Lock` | Two concurrent `create` calls won't collide. |

## Try it

```bash
python run_checkpoint.py checkpoint-2-agent-loop --reload
```

Three demo flows that exercise the loop:

```bash
# Two tool calls — knowledge_search isn't here yet, so the agent picks task_manager + employee_lookup:
curl -X POST localhost:8000/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Find Bob in engineering and create a task to schedule a 1:1 with him."}'

# A goal that requires the calculator:
curl -X POST localhost:8000/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"goal":"What time is it, and how many minutes until 5pm?"}'
```

The response includes a `steps` array — every tool call is visible with input, output, and `tool_status`.

## Engineering details worth pointing out

- **The whole loop is async.** No `time.sleep`, no `requests`. Look at [`raw_loop.py`](app/agents/raw_loop.py) — every external call is awaited.
- **The LLM never sees Python exceptions.** When `task_manager` is called without a `title`, the registry returns `ToolResult.fail(...)` and the loop feeds that string back to the LLM as the next observation.
- **Retries live at the LLM boundary.** Open [`app/services/llm.py`](app/services/llm.py) — every external call is wrapped with `llm_retry()`.

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | Liveness + active provider |
| GET | `/tools/list` | All four tool schemas |
| POST | `/tools/call` | One-turn tool call (CP1) |
| POST | `/agent/run` | Multi-step async agent loop |

## Connection to next

> "Your loop works. But: anyone can ask anything, write tools run with no
> approval, and `knowledge_search` doesn't exist yet — keyword matching
> would fail on `how many days off do I get?` anyway. Next: safety, the
> LangGraph rebuild, and proper RAG retrieval."
