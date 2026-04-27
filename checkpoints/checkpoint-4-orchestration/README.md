# Checkpoint 4 — Orchestration & Observability

The final state. Two new ideas:

1. **Multi-agent orchestration.** A planner agent produces a structured `AgentPlan` (Pydantic, via provider-native JSON mode — week-2's structured outputs applied here), and an executor agent (the LangGraph ReAct agent from CP3) runs each plan step.
2. **Full observability.** Every orchestration run produces a trace_id. `/traces/{trace_id}` returns every LLM call, tool call, plan step, and safety check with timestamps and timings.

## What's new

| Layer | File(s) | Why |
|-------|---------|-----|
| Orchestrator service | [`app/services/orchestrator.py`](app/services/orchestrator.py) | Plan -> validate -> execute step-by-step -> trace. |
| Structured planner | [`app/models/orchestration.py`](app/models/orchestration.py), [`app/prompts/planner.py`](app/prompts/planner.py) | `call_llm_structured(... AgentPlan)` — schema-validated JSON, no parsing. |
| Trace store | [`app/services/tracer.py`](app/services/tracer.py) | Async-safe in-memory store + `AgentTracer` callback handler that pushes LangChain events into the same trace. |
| /orchestrate route | [`app/routes/orchestrate.py`](app/routes/orchestrate.py) | One call: plan + execute, returns plan + results + trace_id. |
| /traces routes | [`app/routes/trace.py`](app/routes/trace.py) | List recent runs, drill into one. |

## Try it

```bash
python run_checkpoint.py checkpoint-4-orchestration --reload
```

```bash
# A goal that needs three steps:
curl -X POST localhost:8000/orchestrate \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Find the remote work policy, calculate 15% of 230, and tell me the time"}'

# Drill into the trace for that run (use the trace_id returned above):
curl localhost:8000/traces
curl localhost:8000/traces/trace_xxxxxxxx
```

The trace has rows like:
```
{"action":"safety_check", ...}
{"action":"planning_start", "detail":{"goal":"..."}}
{"action":"planning_complete", "detail":{"steps":3, "strategy":"..."}}
{"action":"execution_step_start", "detail":{"step":1, "tool":"knowledge_search"}}
{"action":"tool_call_start", ...}
{"action":"tool_call_end", "detail":{"duration_ms":421, ...}}
{"action":"execution_step_complete", ...}
... etc
```

## Why this is the final shape

- **Planner = deliberation, executor = action.** When goals get fuzzy, a deliberate plan beats reactive ReAct because every step has a stated intent.
- **Structured outputs guarantee plans we can iterate over.** No prompt-engineering JSON repair loops.
- **Per-step tool restriction.** The executor is constructed with `allowed_tools=[plan_step.tool_needed]` — even if the LLM hallucinates a different tool inside the executor, it can't reach it.
- **One trace per goal.** Even when the planner runs the LLM and each executor runs its own LangGraph agent, every event lands under the same `trace_id` because the orchestrator threads `AgentTracer(traces, trace_id)` into every executor's `callbacks`.

## Endpoints (full list, all checkpoints cumulative)

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | Liveness + active provider |
| GET | `/tools/list` | Five tool schemas |
| POST | `/tools/call` | One-turn tool call |
| POST | `/agent/run` | LangGraph ReAct agent |
| POST | `/agent/run-raw` | From-scratch loop (comparison) |
| POST | `/agent/approve` | HITL placeholder |
| POST | `/safety/check-prompt` | Injection heuristic |
| GET | `/safety/permissions` | Read/write classification |
| POST | `/orchestrate` | Plan + execute a multi-step goal |
| GET | `/traces` | Recent traces |
| GET | `/traces/{trace_id}` | Full trace timeline |
| POST | `/rag/ingest` | (Re-)index the knowledge base |
| GET | `/rag/status` | Index size |

## Connection back to weeks 1–3

- **Week 1 (FastAPI / Docker):** `Dockerfile`, lifespan startup, `/health` endpoint.
- **Week 2 (Structured outputs):** the planner uses `call_llm_structured(... AgentPlan)` — provider-native JSON, schema-validated. Open [`app/services/orchestrator.py`](app/services/orchestrator.py).
- **Week 3 (RAG):** `knowledge_search` is the same RAG pipeline from week 3 — embedder, Chroma store, retriever, ingest. Compare the same query against the keyword-only week-4 baseline to see the win.
