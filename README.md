# Project 4 — Agentic AI Service

A standalone, production-shaped FastAPI service that teaches the four core primitives of agent engineering across **four runnable checkpoints**, on top of the same Acme Corp domain used through weeks 1–3 of the bootcamp.

| Checkpoint | Focus | Endpoints |
|------------|-------|-----------|
| [1 — Tool Calling](checkpoints/checkpoint-1-tool-calling/) | Async tool calling primitive + the engineering foundation (settings, lifespan, DI, structlog, async LLM client, structured `ToolResult`) | `/tools/list`, `/tools/call` |
| [2 — Agent Loop](checkpoints/checkpoint-2-agent-loop/) | From-scratch async agent loop with structured tool errors and tenacity retries | `+ /agent/run` |
| [3 — Safety + RAG + LangGraph](checkpoints/checkpoint-3-safety-rag/) | LangGraph ReAct agent, prompt-injection detection, tool permissions, RAG-backed `knowledge_search` (week-3 integration) | `+ /agent/run` (LangGraph), `/safety/*`, `/rag/*` |
| [4 — Orchestration & Observability](checkpoints/checkpoint-4-orchestration/) | Planner+executor multi-agent pipeline (week-2 structured outputs) + per-run trace store with LangChain callback bridge | `+ /orchestrate`, `/traces/*` |

Each checkpoint has its own `app/` directory and is fully runnable. Each checkpoint folder has a README; each checkpoint has matching teaching notes in [`instructor-notes/`](instructor-notes/).

## What's "production-shaped"?

This project applies the engineering standards a real service is held to — see [docs/engineering-standards.md](docs/engineering-standards.md). The headlines:

- **Async, all the way down.** FastAPI handlers, tool functions, LLM SDK calls, embedding calls. ChromaDB is sync, so the wrapper hops to the threadpool. No `requests`, no `time.sleep`.
- **Dependency injection via `Depends`.** Routes never instantiate clients. Singletons are built once in [lifespan](checkpoints/checkpoint-4-orchestration/app/lifespan.py).
- **One typed `Settings` class** built on `pydantic-settings`. Required keys are validated at startup; the app refuses to boot if `OPENAI_API_KEY` is missing.
- **Pydantic at every boundary.** HTTP bodies, tool I/O (`ToolResult`), LLM structured output (`AgentPlan`).
- **Bounded retries with tenacity** on every external call.
- **Tool errors are returned to the LLM as structured data** (`ToolResult.fail`), not raised as Python exceptions. The agent loop never crashes because a single tool's API hiccupped.
- **Structured logging** via `structlog`. JSON in production, key/value locally. No `print` statements.
- **Linters and a CI pipeline.** `ruff` + pre-commit, GitHub Actions runs lint + format-check + tests on every push.
- **Tests for the critical path.** Each tool in isolation, Pydantic schemas with valid + invalid inputs, end-to-end agent loop with a fake LLM. No API keys, no network.

## Cross-week integrations

- **Week 1 (FastAPI / Docker):** the Dockerfile is multi-stage with `uv`, the lifespan handles startup/shutdown, `/health` is wired in.
- **Week 2 (Structured outputs):** the planner in CP4 returns a Pydantic `AgentPlan` via provider-native JSON mode. No prompt-engineering JSON repair.
- **Week 3 (RAG):** the `knowledge_search` tool in CP3 is the same RAG pipeline from week 3 — async embedder, ChromaDB store, retriever, ingest. Drop-in replacement for the keyword version. See [`app/rag/`](checkpoints/checkpoint-3-safety-rag/app/rag/).

## Quick start

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
# 2. Sync deps
cd project-4-agents
uv sync

# 3. Configure provider keys
cp .env.example .env  # set GEMINI_API_KEY (or OPENAI_API_KEY, AZURE_*)

# 4. Run any checkpoint
python run_checkpoint.py checkpoint-1-tool-calling --reload
python run_checkpoint.py checkpoint-4-orchestration --reload

# 5. Open the docs
open http://localhost:8000/docs
```

### Tests

```bash
uv run pytest -q
```

### Linting

```bash
uv run ruff check .
uv run ruff format --check .

# Or, install the pre-commit hook so it runs automatically:
uv run pre-commit install
```

### Docker

```bash
# Build + run the final checkpoint:
docker compose up --build

# Or build a different checkpoint:
docker compose build --build-arg CHECKPOINT=checkpoint-2-agent-loop
docker compose up
```

## Repo layout

See [docs/architecture.md](docs/architecture.md) for the full picture. In short:

```
project-4-agents/
├── pyproject.toml             # uv-managed
├── Dockerfile                 # multi-stage uv build
├── docker-compose.yml
├── run_checkpoint.py          # launches any checkpoint
├── data/                      # acme-knowledge.json, sample-tasks.json
├── base-app/app/              # minimal scaffold for live coding
├── checkpoints/
│   ├── checkpoint-1-tool-calling/{app,README.md}
│   ├── checkpoint-2-agent-loop/{app,README.md}
│   ├── checkpoint-3-safety-rag/{app,README.md}
│   └── checkpoint-4-orchestration/{app,README.md}
├── instructor-notes/          # one teaching guide per checkpoint
├── docs/                      # architecture + engineering-standards
└── tests/                     # critical-path tests, run against CP4
```

## Where to start reading

- The end state: [`checkpoints/checkpoint-4-orchestration/app/main.py`](checkpoints/checkpoint-4-orchestration/app/main.py) → [`lifespan.py`](checkpoints/checkpoint-4-orchestration/app/lifespan.py) → any route → its service.
- The agent control flow: [`agents/raw_loop.py`](checkpoints/checkpoint-4-orchestration/app/agents/raw_loop.py) (the simple version) and [`agents/langgraph.py`](checkpoints/checkpoint-4-orchestration/app/agents/langgraph.py) (the production version).
- Why we made each engineering choice: [`docs/engineering-standards.md`](docs/engineering-standards.md).
