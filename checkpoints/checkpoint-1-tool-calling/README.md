# Checkpoint 1 — Async Tool Calling

The minimum viable agentic primitive: **the LLM picks a tool, we run it, return the result**. No loops, no orchestration, no RAG. This checkpoint exists to lock in the engineering foundation that every later checkpoint builds on.

## What's new

| Layer | File(s) | Why |
|-------|---------|-----|
| Typed config | [`app/settings.py`](app/settings.py) | One `Settings` class, validated at startup. No `os.getenv` anywhere else. |
| Structured logging | [`app/logging_config.py`](app/logging_config.py) | structlog with JSON output for production, key/value for local. |
| Lifespan singletons | [`app/lifespan.py`](app/lifespan.py) | LLM client and tool registry are built once, disposed cleanly. |
| Dependency injection | [`app/deps.py`](app/deps.py) | Routes declare what they need with `Depends(...)`. |
| Async LLM client | [`app/services/llm.py`](app/services/llm.py) | `AsyncOpenAI`, `AsyncAzureOpenAI`, Gemini's `client.aio.*`. Tenacity retries. |
| Tool contract | [`app/tools/base.py`](app/tools/base.py), [`app/models/tool.py`](app/models/tool.py) | Every tool is `async def run() -> ToolResult`. Failures are structured, never raised. |
| Two tools | [`calculator.py`](app/tools/calculator.py), [`clock.py`](app/tools/clock.py) | Smallest illustrative pair. |
| Routes | [`app/routes/`](app/routes/) | `/tools/list`, `/tools/call`, `/health`. All async, all DI. |

## Try it

```bash
# from project-4-agents/
uv sync
cp .env.example .env       # set GEMINI_API_KEY (or OPENAI_API_KEY)

python run_checkpoint.py checkpoint-1-tool-calling --reload
```

Then:

```bash
curl localhost:8000/tools/list
curl -X POST localhost:8000/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is 147 times 23?"}'
```

The LLM picks `calculator`, we execute it, the response includes `tool_status` and the structured result.

## Try this too — the unhappy path

```bash
curl -X POST localhost:8000/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is 10 divided by 0?"}'
```

The tool returns `ToolResult.fail("Division by zero ...")`. Status is `"error"`, the agent sees a structured message — nothing crashes.

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | Liveness + active provider |
| GET | `/tools/list` | All tool JSON schemas |
| POST | `/tools/call` | One-turn tool call (no loop) |

## What you can't do yet

- Chain multiple tool calls. (CP2)
- Search the Acme knowledge base. (CP3)
- Plan with a separate agent. (CP4)

## Connection to next

> "One tool call per request is fine for `What time is it?`. Real questions like
> `What is our vacation policy and how many days are left if I took 5?` need
> two tools in sequence — knowledge_search, then calculator. Next checkpoint:
> the agent loop."
