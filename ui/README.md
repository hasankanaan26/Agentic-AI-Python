# Checkpoint 3 Workshop UI

A React + Vite + Tailwind UI for poking at every endpoint exposed by
`checkpoints/checkpoint-3-safety-rag/app`. Built to *teach* the
concepts, not just call them — every page has inline explainers, presets,
and an animated step timeline that visualises the agent loop.

## What it covers

| Page              | Endpoints exercised                                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| Overview          | `/health`                                                                                            |
| Agent             | `POST /agent/run`, `POST /agent/approve`, `GET /safety/permissions`                                  |
| HITL flow         | `POST /agent/run` (require_approval), `POST /agent/approve`, `GET /agent/pending`                    |
| Tasks             | `GET /tasks/list`, `POST /agent/run` (write), `POST /agent/approve`                                  |
| Safety            | `POST /safety/check-prompt`, `GET /safety/permissions`                                               |
| RAG               | `GET /rag/status`, `POST /rag/ingest`, `POST /agent/run` scoped to `knowledge_search`                |
| Tools             | `GET /tools/list`, `POST /tools/call`                                                                |
| Compare engines   | `POST /agent/run` and `POST /agent/run-raw` side-by-side                                             |

Features worth noticing:

- **Tool permission toggles** — pick which tools the agent is allowed to
  call. The UI hits `GET /safety/permissions` so each tool is also tagged
  `read` or `write`.
- **HITL flow, end-to-end** — flipping `require_approval` wires
  LangGraph's `interrupt_before=["tools"]`. The response shows the
  paused state, the **proposed tool call** with its full args, and
  **Approve** / **Reject** buttons. Approving rebuilds the graph
  without the interrupt and calls `ainvoke(None, …)` with the same
  `thread_id` — the MemorySaver picks up from the checkpoint and
  finishes the run. The dedicated `HITL flow` tab walks through the
  whole sequence with a stage tracker and a pending-threads inbox.
- **Tasks grid** — live view of `task_manager`'s state via
  `GET /tasks/list`. Every create / complete is routed through
  `/agent/run` with `require_approval=true`, so each mutation surfaces
  the same approval gate the HITL page demonstrates.
- **Animated agent timeline** — steps reveal one by one, so you can
  follow the think-act-observe rhythm even though `/agent/run` itself is
  non-streaming in CP3.
- **Curated presets** — every page has a row of preset prompts that load
  scenarios designed to fail / succeed / pause in instructive ways
  (semantic retrieval, multi-tool chains, scope refusals, error recovery,
  HITL pause, etc.).
- **Request preview** — the Agent page shows the exact JSON body the UI
  is about to POST, so you can copy it into `curl` or your tests.

## How to run

You need **two** processes: the FastAPI backend, then this UI.

### 1. Start the checkpoint-3 backend

From the project root:

```bash
python run_checkpoint.py checkpoint-3-safety-rag --reload
```

This boots FastAPI on `http://localhost:8000`. Confirm it's alive:

```bash
curl http://localhost:8000/health
```

If you want the prompt-injection guard active (the "Safety" page demos
will then also block `/agent/run` with HTTP 400), start it with:

```bash
ENABLE_INJECTION_DETECTION=true python run_checkpoint.py checkpoint-3-safety-rag --reload
```

### 2. Install and start the UI

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:5173`.

That's it — Vite proxies `/agent`, `/safety`, `/rag`, `/tools`, and
`/health` to `http://localhost:8000`, so there's no CORS to configure
and no `VITE_*` env vars to set.

### Pointing at a different backend

Set `BACKEND_URL` before `npm run dev`:

```bash
BACKEND_URL=http://192.168.1.10:8000 npm run dev
```

## Suggested walkthrough (~10 min)

1. **Overview** — read the three pillars, glance at the endpoint table.
2. **Agent + HITL** — click `RAG: semantic retrieval`, run, watch the
   timeline. Then click `HITL: pause before tools`, run, observe the
   paused state. Hit **Approve** to see the placeholder response.
3. **Safety** — try the four injection examples; only the last one
   should pass clean.
4. **RAG** — verify chunks are indexed, run a couple of natural-language
   queries against `knowledge_search`. Look at the cosine distances.
5. **Tools** — confirm every tool's args schema looks right. Try
   `What is 147 times 23?` to see a single-turn calculator call.
6. **Compare engines** — run the same multi-tool goal on both. Notice
   the `thread_id` only appears on the LangGraph response.

## File layout

```
ui/
  index.html
  vite.config.ts          # proxy config — backend URL lives here
  package.json
  tsconfig.json
  src/
    main.tsx
    App.tsx               # tab shell + health pill
    api.ts                # typed fetch client
    types.ts              # mirrors of the FastAPI Pydantic models
    components/
      ui.tsx              # Card, Button, Badge, Toggle, Spinner, JsonView, ...
      AgentTimeline.tsx   # the animated step list
    views/
      DashboardView.tsx
      AgentRunView.tsx    # the centerpiece: run + HITL
      SafetyView.tsx
      RagView.tsx
      ToolsView.tsx
      CompareView.tsx
```

## Troubleshooting

- **"backend offline" badge in the header** — the backend isn't running
  on `localhost:8000`, or `npm run dev` was started before the backend
  was up. Restart `npm run dev` once the backend is alive, or set
  `BACKEND_URL` if it's elsewhere.
- **`POST /rag/ingest` returns a 500 with an Azure 404** — the embedding
  deployment isn't configured. Set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
  in your `.env` to the deployment name (not the model name) and
  restart.
- **`/agent/run` returns 400 with `risk_level`** — that's working as
  intended; injection detection is on. Try a clean prompt or turn off
  `ENABLE_INJECTION_DETECTION` to bypass.
