import { useEffect, useState } from "react";
import { api } from "./api";
import { Badge } from "./components/ui";
import {
  BookIcon,
  HitlIcon,
  ListIcon,
  ShieldIcon,
  SparkIcon,
  ZapIcon,
} from "./components/icons";
import { DashboardView } from "./views/DashboardView";
import { AgentRunView } from "./views/AgentRunView";
import { HitlView } from "./views/HitlView";
import { TasksView } from "./views/TasksView";
import { SafetyView } from "./views/SafetyView";
import { RagView } from "./views/RagView";
import { ToolsView } from "./views/ToolsView";
import { CompareView } from "./views/CompareView";

export type Tab =
  | "dashboard"
  | "agent"
  | "hitl"
  | "tasks"
  | "safety"
  | "rag"
  | "tools"
  | "compare";

const TABS: { key: Tab; label: string; icon: React.ReactNode; hint: string }[] = [
  { key: "dashboard", label: "Overview", icon: <BookIcon size={14} />, hint: "What CP3 does" },
  { key: "agent", label: "Agent", icon: <SparkIcon size={14} />, hint: "Run, gate tools, watch the timeline" },
  { key: "hitl", label: "HITL flow", icon: <HitlIcon size={14} />, hint: "End-to-end approval" },
  { key: "tasks", label: "Tasks", icon: <ListIcon size={14} />, hint: "Grid + writes via approval" },
  { key: "safety", label: "Safety", icon: <ShieldIcon size={14} />, hint: "Injection + permissions" },
  { key: "rag", label: "RAG", icon: <BookIcon size={14} />, hint: "Ingest + search" },
  { key: "tools", label: "Tools", icon: <ZapIcon size={14} />, hint: "Schemas + direct call" },
  { key: "compare", label: "Compare", icon: <SparkIcon size={14} />, hint: "LangGraph vs raw loop" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setHealthErr(e.message ?? String(e)));
  }, []);

  return (
    <div className="mx-auto flex min-h-full max-w-7xl flex-col">
      <header className="border-b border-slate-700/60 px-6 py-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-sky-400 to-violet-500 text-slate-950 shadow-lg shadow-sky-500/30">
                <SparkIcon size={18} />
              </span>
              <span className="text-2xl font-bold tracking-tight text-slate-50">
                Checkpoint 3 Workshop
              </span>
              <Badge tone="info">Safety + LangGraph + RAG</Badge>
            </div>
            <p className="mt-1.5 text-sm text-slate-400">
              Agent loop, prompt-injection guard, tool permissions, end-to-end HITL,
              and semantic RAG — every endpoint of the CP3 backend covered.
            </p>
          </div>
          <HealthPill health={health} error={healthErr} />
        </div>
      </header>

      <nav className="sticky top-0 z-10 border-b border-slate-700/60 bg-slate-950/85 backdrop-blur">
        <div className="flex flex-wrap gap-1 px-4 py-2">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-all ${
                tab === t.key
                  ? "bg-sky-500/20 text-sky-200 ring-1 ring-sky-500/40"
                  : "text-slate-300 hover:bg-slate-800/60"
              }`}
              title={t.hint}
            >
              <span className="opacity-80">{t.icon}</span>
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="flex-1 px-6 py-8">
        {tab === "dashboard" && <DashboardView onJump={setTab} />}
        {tab === "agent" && <AgentRunView />}
        {tab === "hitl" && <HitlView />}
        {tab === "tasks" && <TasksView />}
        {tab === "safety" && <SafetyView />}
        {tab === "rag" && <RagView />}
        {tab === "tools" && <ToolsView />}
        {tab === "compare" && <CompareView />}
      </main>

      <footer className="border-t border-slate-700/60 px-6 py-4 text-xs text-slate-500">
        UI proxies <span className="font-mono">/agent</span>,{" "}
        <span className="font-mono">/safety</span>,{" "}
        <span className="font-mono">/rag</span>,{" "}
        <span className="font-mono">/tools</span>,{" "}
        <span className="font-mono">/tasks</span>, and{" "}
        <span className="font-mono">/health</span> to{" "}
        <span className="font-mono">localhost:8000</span>.
      </footer>
    </div>
  );
}

function HealthPill({
  health,
  error,
}: {
  health: Record<string, unknown> | null;
  error: string | null;
}) {
  if (error) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <Badge tone="error">backend offline</Badge>
        <span className="text-xs text-slate-500" title={error}>
          {error.slice(0, 60)}
        </span>
      </div>
    );
  }
  if (!health) return <Badge tone="neutral">checking…</Badge>;
  const provider = (health as { provider?: string }).provider ?? "unknown";
  const model = (health as { model?: string }).model;
  return (
    <div className="flex items-center gap-2 text-sm">
      <Badge tone="ok">backend ok</Badge>
      <span className="text-xs text-slate-400">
        provider: <span className="font-mono text-slate-200">{provider}</span>
        {model && (
          <>
            {" · "}model: <span className="font-mono text-slate-200">{model}</span>
          </>
        )}
      </span>
    </div>
  );
}
