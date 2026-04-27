# Checkpoint Roadmap

Each checkpoint is its own `app/` directory under [`checkpoints/`](checkpoints/) — fully runnable, fully self-contained. Earlier checkpoints exist on purpose (not just as historical artefacts) so engineers can read the simpler version of any concept before opening the production one.

## Checkpoint 1 — Async Tool Calling

**Duration:** ~2 hours

**Objective:** Lock in the engineering foundation. The smallest possible app that demonstrates the LLM choosing a tool, with all the production scaffolding the rest of the project assumes.

**New in this checkpoint:**

| File | What it does |
|------|-------------|
| `app/settings.py` | Single `Settings` class via `pydantic-settings`. Required keys validated at startup. |
| `app/logging_config.py` | Structured JSON logging via `structlog`. |
| `app/lifespan.py` | Async startup/shutdown — singletons live here. |
| `app/deps.py` | DI providers (`Depends`). |
| `app/services/llm.py` | Async LLM client (Gemini / OpenAI / Azure) with tenacity retries. |
| `app/tools/base.py`, `app/models/tool.py` | The `BaseTool` / `ToolResult` contract. |
| `app/tools/calculator.py`, `app/tools/clock.py` | Two tools. |
| `app/routes/tools.py` | `GET /tools/list`, `POST /tools/call`. |

**Demo flow:**
1. `POST /tools/call {"message":"What is 147 times 23?"}` — calculator picked.
2. `POST /tools/call {"message":"What time is it?"}` — clock picked.
3. `POST /tools/call {"message":"What is the capital of France?"}` — no tool picked.
4. `POST /tools/call {"message":"What is 10 divided by 0?"}` — `ToolResult.fail` returned cleanly.

**Connection to next:** "One tool call per request is fine for `What time is it?`. Real questions need two tool calls in sequence. Next: the agent loop."

---

## Checkpoint 2 — The Agent Loop

**Duration:** ~2 hours

**Objective:** Build the agent loop in pure async Python. Internalise that tool failures are data, not exceptions.

**New in this checkpoint:**

| File | What it does |
|------|-------------|
| `app/agents/raw_loop.py` | The async think-act-observe loop. |
| `app/tools/task_manager.py` | A WRITE-capable tool with branching actions and an `asyncio.Lock`. |
| `app/tools/employee_lookup.py` | A READ-capable directory with TTL caching. |
| `app/routes/agent.py` | `POST /agent/run`. |
| `app/models/agent.py` | `AgentRequest`, `AgentResponse`, `AgentStep`. |

**Demo flow:**
1. `POST /agent/run {"goal":"Look up Bob in engineering, then create a task to schedule a 1:1 with him"}` — calls `employee_lookup` then `task_manager`.
2. `POST /agent/run {"goal":"Mark task 999 as done, then list everything"}` — tool returns structured error, LLM sees it, recovers.
3. Walk through the `steps[]` array — every tool call is visible with `tool_status`.

**Connection to next:** "Anyone can ask anything, write tools run with no approval, and `knowledge_search` doesn't exist yet. Next: safety, the LangGraph rebuild, and proper RAG."

---

## Checkpoint 3 — Safety, LangGraph, and RAG

**Duration:** ~2.5 hours

**Objective:** Three concerns land together because they're how an agent becomes production-shaped.

**New in this checkpoint:**

| File | What it does |
|------|-------------|
| `app/agents/langgraph.py` | LangGraph ReAct agent (async `ainvoke`, checkpointed, supports `interrupt_before`). |
| `app/tools/langchain_tools.py`, `app/tools/schemas.py` | LangChain `StructuredTool` wrappers with `args_schema` validation. |
| `app/services/safety.py`, `app/routes/safety.py` | Heuristic prompt-injection detection. |
| `app/services/embeddings.py` | Async embedding service (Gemini / OpenAI / Azure). |
| `app/services/vector_store.py` | ChromaDB wrapper, sync calls hopped to threadpool. |
| `app/rag/{ingest,retriever}.py` | RAG ingest pipeline + retrieval helper. |
| `app/tools/knowledge_search.py` | RAG-backed semantic search (replaces keyword version). |
| `app/routes/rag.py` | `POST /rag/ingest`, `GET /rag/status`. |

**Demo flow:**
1. `POST /agent/run {"goal":"how many days off do I get?"}` — RAG retrieves "Annual Leave" despite zero word overlap.
2. `POST /agent/run {"goal":"Create a task","allowed_tools":["calculator","clock"]}` — agent has no `task_manager`, declines politely.
3. With `ENABLE_INJECTION_DETECTION=true`: `POST /agent/run {"goal":"Ignore previous instructions ..."}` — 400 with risk_level + findings.
4. `POST /rag/ingest?force=true`, `GET /rag/status`.

**Connection to next:** "How do you debug WHY the agent picked the wrong tool? And what about goals that need a deliberate plan rather than reactive thinking? Next: orchestration and tracing."

---

## Checkpoint 4 — Orchestration & Observability

**Duration:** ~2 hours

**Objective:** Multi-agent orchestration with structured planning, streaming, human-in-the-loop approval, and production-grade tracing via LangSmith.

**New in this checkpoint:**

| File | What it does |
|------|-------------|
| `app/services/orchestrator.py` | Planner + executor pipeline. Plan via `chat_model.with_structured_output(AgentPlan)`; loop the steps; approval gate for write tools; `astream_events()` for SSE. |
| `app/models/orchestration.py` | `AgentPlan`, `PlanStep`, `OrchestrateResumeRequest` — the structured-output + HITL schemas. |
| `app/prompts/{planner,executor}.py` | The planner system prompt + executor template. |
| `app/services/tracer.py` | Bounded async `TraceStore` for the offline `/traces` demo. Production tracing is LangSmith — auto-on via `LANGCHAIN_TRACING_V2=true`. No custom callback handler. |
| `app/agents/langgraph.py` | Extended with `astream_events()` and `resume()` so the orchestrator can stream and continue paused runs. |
| `app/routes/orchestrate.py` | `POST /orchestrate`, `POST /orchestrate/stream` (SSE), `POST /orchestrate/resume/{trace_id}`. |
| `app/routes/trace.py` | `GET /traces`, `GET /traces/{trace_id}`. |
| `app/models/trace.py` | `TraceEntry`, `AgentTrace`, `TraceSummary`. |

**Demo flow:**
1. `POST /orchestrate {"goal":"Find the remote work policy, calculate 15% of 230, and tell me the time"}` — planner builds a 3-step plan, executor runs each.
2. `POST /orchestrate/stream` with the same goal — watch the SSE events arrive: `planning_*`, `step_*`, `token`, `tool_*`, `summary`, `done`.
3. `POST /orchestrate {"goal":"Create a task to onboard Alice","require_approval":true}` — pauses with `status="awaiting_approval"`. `POST /orchestrate/resume/{trace_id} {"approved":true}` continues.
4. `GET /traces` / `GET /traces/{trace_id}` — every event under one trace_id (offline path).
5. Set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY=...`, restart, run again — the same orchestration appears in the LangSmith UI with token counts, latencies, and prompt diffs, no code change.
6. Compare `/orchestrate` vs `/agent/run` on the same goal — deliberate plan vs reactive thinking.

---

## Capstone

Pick a domain (IT helpdesk, onboarding, expense approval, …), swap the tools for ones that fit your domain, ship a focused autonomous agent on top of this scaffold. Required:

1. ≥ 3 tools (provided or custom).
2. ≥ 2 tool calls per scenario (must exercise the agent loop).
3. ≥ 1 safety mechanism.
4. Tracing visible in the response.
5. ≥ 3 offline tests for tools + Pydantic schemas.
