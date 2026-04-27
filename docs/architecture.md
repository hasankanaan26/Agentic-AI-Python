# Architecture

## Layout

```
project-4-agents/
├── pyproject.toml             # uv-managed workspace
├── Dockerfile                 # multi-stage uv build (CHECKPOINT build-arg)
├── docker-compose.yml         # one-command run with persistent Chroma volume
├── run_checkpoint.py          # launch any checkpoint on :8000
├── .pre-commit-config.yaml    # ruff + housekeeping hooks
├── .github/workflows/ci.yml   # lint + format-check + tests on every push
├── data/                      # Acme knowledge + sample tasks (shared)
├── base-app/app/              # minimal scaffold (settings, logging, lifespan)
├── checkpoints/
│   ├── checkpoint-1-tool-calling/app/
│   ├── checkpoint-2-agent-loop/app/
│   ├── checkpoint-3-safety-rag/app/
│   └── checkpoint-4-orchestration/app/   # the final shape
├── instructor-notes/          # one teaching guide per checkpoint
├── docs/                      # this file + engineering standards
└── tests/                     # critical-path tests, run against CP4
```

Each checkpoint's `app/` directory is its own Python package — fully runnable, fully self-contained. Run any of them with `python run_checkpoint.py <name>`.

## Inside a checkpoint

```
app/
├── main.py              # FastAPI(app=...), router includes
├── lifespan.py          # async startup/shutdown — singletons live here
├── deps.py              # Depends() providers
├── settings.py          # one Settings class (pydantic-settings)
├── logging_config.py    # structlog wiring
├── exceptions.py        # AgentError, ProviderConfigError, etc.
├── retries.py           # tenacity policies
├── models/              # Pydantic at every boundary
├── services/            # singletons: LLM, embeddings, vector store, tracer, orchestrator, safety
├── agents/              # the loops: raw_loop, langgraph
├── prompts/             # system prompts as plain strings
├── rag/                 # ingest pipeline (CP3+)
├── tools/               # one file per tool + base.py + registry.py
└── routes/              # one router per concern
```

## Request flow (CP4, `/orchestrate`)

```
client
  │
  ▼
FastAPI route                  routes/orchestrate.py
  │  injects via Depends:
  │    - orchestrator
  ▼
OrchestratorService.run        services/orchestrator.py
  ├─► safety check             services/safety.py
  ├─► trace_id = await traces.create(goal)
  ├─► plan = await planner.ainvoke(...)        <-- chat_model.with_structured_output(AgentPlan)
  └─► for each step:
        ├─► if require_approval and step.tool_needed in WRITE_TOOLS:
        │       store snapshot in _pending[trace_id]
        │       return status="awaiting_approval"
        │       (client calls POST /orchestrate/resume/{trace_id})
        ├─► await runner.run(                  agents/langgraph.py
        │     allowed_tools=[step.tool_needed])
        │     └─► LangGraph create_react_agent
        │           └─► registry.execute(...)  tools/registry.py
        │                 └─► tool.run(**args) -> ToolResult
        └─► push tool calls to TraceStore + record execution_step_(complete|failed)
```

For SSE (`POST /orchestrate/stream`), `OrchestratorService.astream_events()` yields `planning_*`, `step_*`, and per-token / per-tool events proxied from `runner.astream_events()`.

Every `await` in that path is a real async hop — no thread blocked, no event loop stalled.

## Why this layout

- **Routes are thin.** They convert HTTP → service call → HTTP. Logic lives in services.
- **Services are singletons.** `LLMService`, `EmbeddingService`, `VectorStore`, `ToolRegistry`, `LangGraphAgentRunner`, `OrchestratorService`, `TraceStore` — each created once in `lifespan`, exposed via `Depends`.
- **Tools are unit-testable.** `await CalculatorTool().run(operation="add", a=1, b=2)` works in a test with no fixtures.
- **Models are at every boundary.** HTTP, tool I/O, LLM structured output. Once data crosses in, types are trusted.

## Engines

Two agent engines coexist on purpose:

- [`agents/raw_loop.py`](../checkpoints/checkpoint-4-orchestration/app/agents/raw_loop.py) — ~80 lines of pure Python. The agent loop without LangChain.
- [`agents/langgraph.py`](../checkpoints/checkpoint-4-orchestration/app/agents/langgraph.py) — `create_react_agent()`, checkpointing, interrupts, async (`ainvoke`).

Same tools, same prompts, same response shape. Engineers can compare directly.

## RAG pipeline (CP3+)

```
acme-knowledge.json
        │
        ▼
ingest_knowledge():           rag/ingest.py
   chunks = _load_entries()
   vectors = await embeddings.embed_texts(...)   services/embeddings.py
   await store.add_chunks(chunks, vectors)       services/vector_store.py

KnowledgeSearchTool.run(query):                   tools/knowledge_search.py
   await ensure_indexed()
   chunks = await retrieve(query, ...)            rag/retriever.py
   format and cache
```

Chroma is sync; the `VectorStore` wrapper hops to the threadpool with `run_in_threadpool`, so every public method is `async def`.

## Tracing

- **In-memory trace store** ([`services/tracer.py`](../checkpoints/checkpoint-4-orchestration/app/services/tracer.py)) — bounded by `TRACE_STORE_MAX`, async-locked. Backs `GET /traces` and `GET /traces/{trace_id}`. The orchestrator pushes events to it directly while running.
- **LangSmith** (production) — set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY=...` and every LangChain / LangGraph node, LLM call, and tool call streams to the LangSmith UI for free. There is no custom callback handler; LangSmith does this via the `langsmith` package that ships transitively with `langchain`.

## Configuration

One `Settings` (pydantic-settings) loaded at startup. Required keys for the active provider are validated by a `model_validator` — missing keys raise at import time, not on the third request. See [`settings.py`](../checkpoints/checkpoint-4-orchestration/app/settings.py).
