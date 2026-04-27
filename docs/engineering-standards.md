# Engineering Standards

These rules are enforced across every checkpoint. They aren't aspirational — every file in the project follows them.

## 1. Async, all the way down

- **Routes** are `async def`. See [`routes/`](../checkpoints/checkpoint-4-orchestration/app/routes/).
- **Tools** are `async def run(...)`. See [`tools/base.py`](../checkpoints/checkpoint-4-orchestration/app/tools/base.py).
- **LLM calls** use the SDKs' async clients (`AsyncOpenAI`, `AsyncAzureOpenAI`, `genai.Client.aio.*`). See [`services/llm.py`](../checkpoints/checkpoint-4-orchestration/app/services/llm.py).
- **HTTP** uses `httpx` with `Timeout` configured.
- **Sync libraries** (e.g. ChromaDB) are wrapped with `fastapi.concurrency.run_in_threadpool`. See [`services/vector_store.py`](../checkpoints/checkpoint-4-orchestration/app/services/vector_store.py).
- **No `time.sleep`, no `requests`** anywhere on a request path.

## 2. Dependency injection — use `Depends`

Routes declare what they need:

```python
async def agent_run(
    request: SafeAgentRequest,
    runner: Annotated[LangGraphAgentRunner, Depends(get_langgraph_runner)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> LangGraphAgentResponse:
```

The provider functions live in [`deps.py`](../checkpoints/checkpoint-4-orchestration/app/deps.py). In tests, `app.dependency_overrides[get_llm] = ...` swaps the LLM client without monkey-patching globals.

## 3. Singletons via lifespan

Things that should exist exactly once per process — LLM client, embedding client, vector store, tool registry, agent runner, trace store — live on `app.state` and are built in [`lifespan.py`](../checkpoints/checkpoint-4-orchestration/app/lifespan.py):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    llm = LLMService(settings)
    embeddings = EmbeddingService(settings)
    # ...
    app.state.app_state = AppState(llm=llm, ...)
    try:
        yield
    finally:
        await llm.aclose()
        await embeddings.aclose()
```

Disposed cleanly on shutdown.

## 4. Caching where it pays off

- `@lru_cache(maxsize=1)` on `get_settings()` — deterministic, called everywhere, must return the same instance.
- `cachetools.TTLCache` on tool responses where it makes sense:
  - `KnowledgeSearchTool` caches identical queries (saves an embedding call + a vector search).
  - `EmployeeLookupTool` caches the same lookup (directory churn is rare).
- We do **not** cache:
  - clock (time-sensitive, point of the tool),
  - calculator (cheap),
  - task_manager (mutating, per-call).

## 5. Configuration — pydantic-settings

One `Settings` class. Required keys for the active provider are validated by a `model_validator` at startup. The app refuses to boot if `LLM_PROVIDER=openai` and `OPENAI_API_KEY` is empty. See [`settings.py`](../checkpoints/checkpoint-4-orchestration/app/settings.py).

```python
@model_validator(mode="after")
def _require_provider_credentials(self) -> Settings:
    if self.llm_provider == "openai" and not self.openai_api_key:
        raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")
    ...
```

## 6. Type hints + Pydantic at the boundary

Every function has type hints. Every external boundary is a Pydantic model:

- HTTP request bodies: `ToolCallRequest`, `OrchestrationRequest`, …
- LLM structured outputs: `AgentPlan` (used by the planner via `call_llm_structured`).
- Tool I/O: `ToolResult` (every tool returns one — never raises on expected failure).

Pydantic is the fence. Inside the fence, types are trusted; we don't validate the same thing five times.

## 7. Errors, retries, failure isolation

- **External calls** (LLM, embedding) are wrapped with timeouts + tenacity retries on transient errors. See [`retries.py`](../checkpoints/checkpoint-4-orchestration/app/retries.py).
- **Auth errors are not retried** — that's intentional. Retrying a 401 is wasted budget.
- **Tool failures don't raise.** They return `ToolResult.fail(error_text)`. The agent loop feeds the structured error back to the LLM; the LLM decides what to do (e.g. retry with different arguments, switch tools, give up gracefully).
- **Orchestration failures are partial.** When step 2 of a 3-step plan fails, steps 1 and 3 still run. The trace records `execution_step_failed` for the bad step.

## 8. Code hygiene

- A real layout: `routes/`, `services/`, `agents/`, `tools/`, `models/`, `prompts/`, `rag/`. **No 600-line `main.py`** — the longest file is `services/llm.py` (~250 lines, three providers).
- Logging via [structlog](https://www.structlog.org/), JSON in production. **No `print` statements.** See [`logging_config.py`](../checkpoints/checkpoint-4-orchestration/app/logging_config.py).
- `ruff` for lint + format, configured in [`pyproject.toml`](../pyproject.toml).
- `pre-commit` runs ruff on every commit. See [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).
- `.env.example` lists every key.
- This document and [`architecture.md`](architecture.md) explain the architecture, not just how to run.

## 9. Tests — at least the critical path

Critical-path coverage in [`tests/`](../tests/):

- **Each tool in isolation** ([`test_tools.py`](../tests/test_tools.py)) — exercises happy paths, structured errors, caching.
- **Pydantic schemas** ([`test_schemas.py`](../tests/test_schemas.py)) — valid + invalid inputs.
- **End-to-end agent loop with a fake LLM** ([`test_agent_with_fake_llm.py`](../tests/test_agent_with_fake_llm.py)) — scripted tool calls + final answer, no network.
- **Safety heuristics** ([`test_safety.py`](../tests/test_safety.py)) — known injections + clean text.

Tests run on every push via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). No API keys required.

## 10. Don't break these (pragmatic warnings)

| Don't | Do |
|-------|----|
| Instantiate clients inside route handlers | Build in lifespan, inject via Depends |
| `os.getenv("X")` in a service module | Read once via `Settings`, inject |
| `requests.post` from a tool | `httpx.AsyncClient` with timeouts |
| `time.sleep` between retries | tenacity handles wait times asynchronously |
| Raise inside a tool on user-error | `ToolResult.fail(...)` — let the LLM see the error |
| Skip type hints "to ship faster" | Type hints + Pydantic ARE shipping faster — they catch the bug at boot |
| `print()` for diagnostics | `log = get_logger(__name__); log.info("event", key=value)` |
