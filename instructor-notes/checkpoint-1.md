# Instructor Notes — Checkpoint 1 (Tool Calling)

**Duration:** ~2 hours
**Mode:** Live coding from `base-app/`, plus reading [`checkpoints/checkpoint-1-tool-calling/`](../checkpoints/checkpoint-1-tool-calling/).

## Learning goals

- Understand the **tool-calling primitive**: a JSON schema sent to the LLM, plus an execute function we run locally.
- Internalise the **engineering foundation** every later checkpoint reuses:
  - one typed `Settings` validated at startup,
  - structured logging,
  - lifespan-managed singletons,
  - dependency injection via `Depends`,
  - async LLM client with bounded retries,
  - structured `ToolResult` so failures don't crash the loop.

## Why this checkpoint exists

Most students reach for `requests`, scatter `os.getenv` calls, instantiate a new OpenAI client per request, and `print()` for diagnostics. We fix that here, *before* there's any agent loop or RAG to distract from it. By CP4, none of it can rot back in.

## Live coding sequence

| Step | What | File |
|------|------|------|
| 1 | Show the missing-key failure mode. Set `LLM_PROVIDER=openai` with no key — the app refuses to start. | [`settings.py`](../checkpoints/checkpoint-1-tool-calling/app/settings.py) |
| 2 | Build `LLMService` and explain why it lives in lifespan, not on every request. | [`services/llm.py`](../checkpoints/checkpoint-1-tool-calling/app/services/llm.py), [`lifespan.py`](../checkpoints/checkpoint-1-tool-calling/app/lifespan.py) |
| 3 | Write the calculator tool — show the JSON schema *and* the execute function side by side. | [`tools/calculator.py`](../checkpoints/checkpoint-1-tool-calling/app/tools/calculator.py) |
| 4 | Wire `/tools/call`. Walk through `Depends(get_llm)` and `Depends(get_registry)`. | [`routes/tools.py`](../checkpoints/checkpoint-1-tool-calling/app/routes/tools.py) |
| 5 | Demo `divide by 0` — show the structured error response. | call the endpoint |

## Demo script

```bash
# 1. The LLM picks calculator
POST /tools/call  {"message": "What is 147 times 23?"}
#    -> tool_called=true, tool_status=ok, tool_result="147 multiply 23 = 3381"

# 2. The LLM picks clock
POST /tools/call  {"message": "What time is it?"}

# 3. The LLM doesn't pick a tool
POST /tools/call  {"message": "What is the capital of France?"}
#    -> tool_called=false, llm_response="Paris"

# 4. The tool returns a structured error
POST /tools/call  {"message": "What is 10 divided by 0?"}
#    -> tool_status=error, tool_result="Tool error: Division by zero ..."
```

## Things to call out explicitly

- **`Settings.model_validator` fails fast.** A missing key isn't a 500 on the third request; it's a refusal to boot.
- **Singletons via lifespan, not module globals.** `get_llm()` returns the same instance to every request, but it's swappable in tests via `app.dependency_overrides`.
- **Async LLM client.** Open `services/llm.py` and find the `await`. Open the OpenAI SDK to confirm `AsyncOpenAI` exists. Compare with the sync version students may have seen elsewhere.
- **`ToolResult.fail` vs raising.** Show what would happen if calculator raised `ZeroDivisionError`. Then show why we don't.
- **Tenacity retries on `_TRANSIENT` errors only.** Auth errors aren't retried — that's intentional.

## Common mistakes

| Mistake | Why it's wrong | Right answer |
|---------|----------------|--------------|
| `client = OpenAI(...)` at the top of `llm.py` | Module-level instantiation runs at import time, before settings are loaded | Build the client in `LLMService.__init__`, instantiate the service in lifespan. |
| `os.getenv("OPENAI_API_KEY")` inside a route | Untyped, unvalidated, scattered | Read `Settings` once, inject via `Depends`. |
| `requests.post(...)` from a tool | Blocks the event loop | Use `httpx.AsyncClient`, await the call. |
| `try: ... except Exception: return "error"` inside a tool | Loses the error type | Return `ToolResult.fail(error_text)` — typed, structured, the LLM can read it. |
| Building the agent client per request | Wastes connection pooling, hits provider rate limits faster | Lifespan + DI. |

## Lab exercise (15 min)

Add a third tool: a **temperature converter** (Celsius ↔ Fahrenheit). Subclass `BaseTool`, return `ToolResult.fail(...)` when the unit is unknown. Wire it into the registry. Test with `/tools/call`. The student should *not* need to touch `services/llm.py` or `routes/tools.py`.

## Wrap-up question

> "If we want one LLM call to solve `What is our vacation policy and how
> many days are left if I took 5?` we'd need TWO tool calls, not one.
> What's the simplest control flow that lets the LLM decide to call
> another tool after seeing the first result?"

That's CP2.
