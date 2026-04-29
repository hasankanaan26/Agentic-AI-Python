import { Tab } from "../App";
import { BookIcon, HitlIcon, ListIcon, ShieldIcon, SparkIcon } from "../components/icons";
import { Button, Callout, Card, Section } from "../components/ui";

const PILLARS = [
  {
    icon: <SparkIcon size={16} />,
    title: "LangGraph ReAct agent",
    body: (
      <>
        <code className="font-mono text-sky-300">create_react_agent</code> gives
        us state checkpointing per <code>thread_id</code>, streaming, and the{" "}
        <code className="font-mono text-sky-300">interrupt_before</code> hook
        used for HITL.
      </>
    ),
    cta: "Run an agent",
    target: "agent" as Tab,
  },
  {
    icon: <ShieldIcon size={16} />,
    title: "Safety layer",
    body: (
      <>
        Heuristic injection detection + per-call tool permission scoping. The
        LLM only sees the tools it's allowed to call.
      </>
    ),
    cta: "Inspect",
    target: "safety" as Tab,
  },
  {
    icon: <BookIcon size={16} />,
    title: "Real RAG",
    body: (
      <>
        Embeddings + cosine search via Chroma. "How many days off?" matches
        the "Annual Leave" doc despite zero word overlap.
      </>
    ),
    cta: "Try it",
    target: "rag" as Tab,
  },
  {
    icon: <HitlIcon size={16} />,
    title: "Human-in-the-loop",
    body: (
      <>
        Pause before any tool, surface the proposed call, approve or reject —
        the same <code>thread_id</code> resumes from the checkpoint.
      </>
    ),
    cta: "See the flow",
    target: "hitl" as Tab,
  },
  {
    icon: <ListIcon size={16} />,
    title: "Tasks (writes via HITL)",
    body: (
      <>
        Live grid backed by <code className="font-mono">/tasks/list</code>.
        Every create / complete is routed through the agent with{" "}
        <code className="font-mono">require_approval=true</code>.
      </>
    ),
    cta: "Open the grid",
    target: "tasks" as Tab,
  },
];

export function DashboardView({ onJump }: { onJump: (t: Tab) => void }) {
  return (
    <div className="space-y-8">
      <Section
        title="What CP3 actually does"
        subtitle="In CP2 you had a working agent. CP3 makes it production-shaped: safe to expose, scoped to a tool subset, and able to answer questions keyword search would miss."
      >
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {PILLARS.map((p) => (
            <Card key={p.title} className="flex h-full flex-col p-5">
              <div className="flex items-center gap-2">
                <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-sky-500/20 text-sky-300">
                  {p.icon}
                </span>
                <h3 className="text-base font-semibold text-slate-50">{p.title}</h3>
              </div>
              <p className="mt-2.5 flex-1 text-sm leading-relaxed text-slate-300">
                {p.body}
              </p>
              <div className="mt-4">
                <Button variant="secondary" size="sm" onClick={() => onJump(p.target)}>
                  {p.cta} →
                </Button>
              </div>
            </Card>
          ))}
        </div>
      </Section>

      <Section
        title="Endpoint coverage"
        subtitle="Every CP3 endpoint has a page below. Two new ones land with this UI: /agent/pending and /tasks/list."
      >
        <div className="grid gap-2 md:grid-cols-2">
          <Endpoint method="POST" path="/agent/run" purpose="LangGraph agent (allowed_tools, require_approval)" />
          <Endpoint method="POST" path="/agent/run-raw" purpose="Original CP2 loop (kept for comparison)" />
          <Endpoint method="POST" path="/agent/approve" purpose="Resume or reject a paused thread" />
          <Endpoint method="GET" path="/agent/pending" purpose="List threads awaiting human review" />
          <Endpoint method="POST" path="/safety/check-prompt" purpose="Run injection heuristics on a string" />
          <Endpoint method="GET" path="/safety/permissions" purpose="Read/write classification per tool" />
          <Endpoint method="GET" path="/tools/list" purpose="JSON schemas for every registered tool" />
          <Endpoint method="POST" path="/tools/call" purpose="One-turn tool call (LLM picks a tool)" />
          <Endpoint method="POST" path="/rag/ingest" purpose="Embed and index the knowledge file" />
          <Endpoint method="GET" path="/rag/status" purpose="Chunk count + Chroma path" />
          <Endpoint method="GET" path="/tasks/list" purpose="Live task grid for the UI" />
          <Endpoint method="GET" path="/health" purpose="Liveness + active provider" />
        </div>
      </Section>

      <Callout tone="success" title="HITL is end-to-end now">
        <code className="font-mono">/agent/approve</code> rebuilds the LangGraph
        without <code className="font-mono">interrupt_before</code> and calls{" "}
        <code className="font-mono">ainvoke(None, …)</code> with the same{" "}
        <code className="font-mono">thread_id</code> — the MemorySaver picks up
        from the checkpoint and finishes the run. Reject discards the pending
        thread and the agent runs zero tools.
      </Callout>
    </div>
  );
}

function Endpoint({
  method,
  path,
  purpose,
}: {
  method: string;
  path: string;
  purpose: string;
}) {
  const tone =
    method === "GET"
      ? "text-emerald-300 bg-emerald-500/10 border-emerald-500/40"
      : "text-amber-200 bg-amber-500/10 border-amber-500/40";
  return (
    <Card className="flex items-center gap-3 p-3">
      <span
        className={`rounded-md border px-2 py-0.5 text-[11px] font-bold ${tone}`}
      >
        {method}
      </span>
      <code className="font-mono text-sm text-slate-200">{path}</code>
      <span className="ml-auto text-right text-xs text-slate-400">{purpose}</span>
    </Card>
  );
}
