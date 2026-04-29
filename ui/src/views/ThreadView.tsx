import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import {
  ClockIcon,
  ListIcon,
  PauseIcon,
  PlayIcon,
  RefreshIcon,
} from "../components/icons";
import {
  Badge,
  Button,
  Callout,
  Card,
  CodeBlock,
  ErrorBox,
  Field,
  Input,
  JsonView,
  Section,
  Spinner,
  StatPill,
  StatusIcon,
} from "../components/ui";
import { readThreads } from "../storage";
import { SnapshotMessage, ThreadDetail, ThreadSnapshot } from "../types";

const MSG_TONE: Record<string, { label: string; tone: "info" | "ok" | "violet" | "warn" | "neutral" }> = {
  human: { label: "user", tone: "info" },
  ai: { label: "assistant", tone: "violet" },
  tool: { label: "tool", tone: "ok" },
  system: { label: "system", tone: "neutral" },
  unknown: { label: "?", tone: "neutral" },
};

const NODE_TONE: Record<string, "info" | "violet" | "warn" | "neutral"> = {
  agent: "violet",
  tools: "warn",
  __start__: "neutral",
};

export function ThreadView() {
  const [recent, setRecent] = useState<string[]>([]);
  const [threadId, setThreadId] = useState("");
  const [data, setData] = useState<ThreadDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [historyOpen, setHistoryOpen] = useState<Record<string, boolean>>({});
  const [showRaw, setShowRaw] = useState(false);

  function refreshRecent() {
    setRecent(readThreads());
  }
  useEffect(() => {
    refreshRecent();
    window.addEventListener("cp3-threads-changed", refreshRecent);
    return () => window.removeEventListener("cp3-threads-changed", refreshRecent);
  }, []);

  // Auto-load the most recent thread on first mount when no id is set yet.
  useEffect(() => {
    if (!threadId && recent.length > 0) {
      setThreadId(recent[0]);
      void load(recent[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recent.length]);

  async function load(id: string) {
    if (!id.trim()) return;
    setLoading(true);
    setError(null);
    setData(null);
    setHistoryOpen({});
    try {
      const r = await api.agentThread(id.trim());
      setData(r);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <Section
        title="Inspect a thread"
        subtitle="Pull the LangGraph state and full checkpoint history for any thread_id the runner has seen. Useful for explaining how the agent's state evolves call by call."
        icon={<ListIcon size={18} />}
      >
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <Card className="space-y-3 p-5">
            <Field label="thread_id" hint="Returned by every /agent/run response. Recent ones picked up from this browser are listed on the right.">
              <div className="flex items-center gap-2">
                <Input
                  value={threadId}
                  onChange={setThreadId}
                  placeholder="thread_3"
                />
                <Button onClick={() => load(threadId)} disabled={loading || !threadId.trim()}>
                  {loading ? <Spinner /> : <PlayIcon size={14} />}Inspect
                </Button>
              </div>
            </Field>
            {data && (
              <Button size="sm" variant="ghost" onClick={() => load(data.thread_id)}>
                <RefreshIcon size={12} /> Refresh state
              </Button>
            )}
          </Card>

          <Card className="space-y-2 p-5">
            <div className="text-sm font-semibold text-slate-100">Recent threads</div>
            <p className="text-xs text-slate-400">
              Stored in this browser only. Cleared when you reset localStorage.
            </p>
            {recent.length === 0 ? (
              <div className="rounded-md border border-dashed border-slate-700 px-3 py-4 text-center text-xs text-slate-500">
                No threads yet — run the agent first.
              </div>
            ) : (
              <ul className="max-h-44 space-y-1 overflow-auto pr-1">
                {recent.map((id) => (
                  <li key={id}>
                    <button
                      onClick={() => {
                        setThreadId(id);
                        void load(id);
                      }}
                      className={`flex w-full items-center justify-between rounded-md border border-slate-700/60 bg-slate-900/60 px-2.5 py-1.5 text-left text-xs hover:border-sky-500/40 ${
                        data?.thread_id === id ? "ring-1 ring-sky-500/40" : ""
                      }`}
                    >
                      <span className="font-mono text-slate-200">{id}</span>
                      {data?.thread_id === id && (
                        <Badge tone="info">selected</Badge>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </Section>

      <ErrorBox error={error} />

      {data && <ThreadSummary data={data} />}

      {data && (
        <Section
          title="Checkpoint history"
          subtitle={`${data.history.length} snapshot(s). Newest first. Each entry is one node-step; expand to see the messages persisted at that point.`}
          icon={<ClockIcon size={18} />}
        >
          <ol className="space-y-2">
            {data.history.map((snap, i) => {
              const key = snap.checkpoint_id ?? String(i);
              const open = historyOpen[key] ?? false;
              return (
                <li key={key}>
                  <CheckpointRow
                    snap={snap}
                    open={open}
                    isCurrent={snap.checkpoint_id === data.current.checkpoint_id}
                    onToggle={() =>
                      setHistoryOpen((h) => ({ ...h, [key]: !open }))
                    }
                  />
                </li>
              );
            })}
          </ol>
        </Section>
      )}

      {data && (
        <Section title="Current values">
          <Card className="p-5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-100">
                Raw current state
              </span>
              <Button size="sm" variant="ghost" onClick={() => setShowRaw((s) => !s)}>
                {showRaw ? "hide" : "show"}
              </Button>
            </div>
            {showRaw && (
              <div className="mt-3">
                <CodeBlock>{JSON.stringify(data.current, null, 2)}</CodeBlock>
              </div>
            )}
          </Card>
        </Section>
      )}

      {!data && !loading && (
        <Callout tone="info" title="How this works">
          The runner attaches a <code className="font-mono">MemorySaver</code> to every
          LangGraph agent. When you call{" "}
          <code className="font-mono">agent.aget_state(config)</code> with a{" "}
          <code className="font-mono">thread_id</code>, you get the latest checkpoint;{" "}
          <code className="font-mono">aget_state_history(...)</code> yields every
          checkpoint ever written — input, agent step, tool step, post-tool step,
          and so on. This page just renders that.
        </Callout>
      )}
    </div>
  );
}

function ThreadSummary({ data }: { data: ThreadDetail }) {
  const totalMessages = data.current.message_count;
  const toolCalls = useMemo(
    () =>
      data.current.messages.reduce(
        (n, m) => n + (m.tool_calls?.length ?? 0),
        0
      ),
    [data]
  );
  return (
    <Section title="Summary">
      <Card className="space-y-4 p-5">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
          <StatPill
            label="thread_id"
            value={<span className="font-mono text-sm">{data.thread_id}</span>}
            tone="info"
          />
          <StatPill
            label="Status"
            value={data.is_paused ? "paused" : "completed"}
            tone={data.is_paused ? "warn" : "ok"}
          />
          <StatPill label="Checkpoints" value={data.history.length} tone="info" />
          <StatPill label="Messages" value={totalMessages} tone="info" />
          <StatPill label="Tool calls" value={toolCalls} tone="info" />
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span>Next:</span>
          {data.current.next.length === 0 ? (
            <Badge tone="ok">— end —</Badge>
          ) : (
            data.current.next.map((n) => (
              <Badge key={n} tone={NODE_TONE[n] ?? "neutral"}>
                <span className="font-mono">{n}</span>
              </Badge>
            ))
          )}
          {data.is_paused && data.pending_context && (
            <span className="ml-2">
              · paused on goal:{" "}
              <span className="text-slate-200">"{data.pending_context.goal}"</span>
            </span>
          )}
        </div>
      </Card>
    </Section>
  );
}

function CheckpointRow({
  snap,
  open,
  isCurrent,
  onToggle,
}: {
  snap: ThreadSnapshot;
  open: boolean;
  isCurrent: boolean;
  onToggle: () => void;
}) {
  const stepLabel = snap.step ?? "—";
  const writeBadges = snap.writes.length > 0 ? snap.writes : null;
  const isPaused = snap.next.includes("tools") && stepLabel !== "—";

  return (
    <Card
      tone={isCurrent ? "accent" : isPaused ? "warn" : "default"}
      className={`overflow-hidden ${isCurrent ? "ring-1 ring-sky-500/30" : ""}`}
    >
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-slate-800/40"
      >
        <span
          className={`inline-flex h-7 w-7 flex-none items-center justify-center rounded-full border-2 text-[11px] font-bold ${
            isCurrent
              ? "border-sky-400 bg-sky-500/20 text-sky-100"
              : isPaused
              ? "border-amber-400 bg-amber-500/20 text-amber-100"
              : "border-slate-600 bg-slate-800/60 text-slate-200"
          }`}
        >
          {stepLabel}
        </span>

        <div className="flex flex-1 flex-wrap items-center gap-2">
          {isCurrent && <Badge tone="info">current</Badge>}
          {isPaused && (
            <Badge tone="warn">
              <PauseIcon size={11} /> paused
            </Badge>
          )}
          {snap.source && <Badge tone="neutral">{snap.source}</Badge>}
          {writeBadges &&
            writeBadges.map((w) => (
              <Badge key={w} tone={NODE_TONE[w] ?? "neutral"}>
                wrote: <span className="font-mono">{w}</span>
              </Badge>
            ))}
          <span className="text-[11px] text-slate-500">
            {snap.message_count} msg{snap.message_count === 1 ? "" : "s"}
          </span>
          {snap.next.length > 0 && (
            <span className="text-[11px] text-slate-500">
              next →{" "}
              <span className="font-mono text-slate-300">
                {snap.next.join(", ")}
              </span>
            </span>
          )}
          {snap.created_at && (
            <span className="ml-auto text-[11px] text-slate-500">
              {snap.created_at.slice(11, 19)}
            </span>
          )}
        </div>
        <span className="text-xs text-slate-500">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="border-t border-slate-700/60 p-4">
          <div className="mb-3 grid gap-2 text-[11px] text-slate-500 md:grid-cols-2">
            <div>
              checkpoint_id:{" "}
              <span className="font-mono text-slate-300">
                {snap.checkpoint_id?.slice(0, 12) ?? "—"}…
              </span>
            </div>
            <div>
              parent:{" "}
              <span className="font-mono text-slate-300">
                {snap.parent_checkpoint_id
                  ? snap.parent_checkpoint_id.slice(0, 12) + "…"
                  : "(root)"}
              </span>
            </div>
          </div>
          {snap.messages.length === 0 ? (
            <div className="text-xs text-slate-500">No messages persisted yet.</div>
          ) : (
            <ol className="space-y-2">
              {snap.messages.map((m, idx) => (
                <li key={idx}>
                  <MessageRow msg={m} />
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </Card>
  );
}

function MessageRow({ msg }: { msg: SnapshotMessage }) {
  const meta = MSG_TONE[msg.type] ?? MSG_TONE.unknown;
  const truncated =
    msg.content.length > 600 ? msg.content.slice(0, 600) + "…" : msg.content;
  return (
    <div className="rounded-lg border border-slate-700/70 bg-slate-950/60 p-3">
      <div className="mb-1.5 flex flex-wrap items-center gap-2">
        <Badge tone={meta.tone}>{meta.label}</Badge>
        {msg.name && (
          <span className="font-mono text-[11px] text-slate-300">
            {msg.name}
          </span>
        )}
        {msg.status === "error" && (
          <Badge tone="error">
            <StatusIcon status="error" /> error
          </Badge>
        )}
        {msg.tool_call_id && (
          <span className="text-[10px] text-slate-500">
            id: <span className="font-mono">{msg.tool_call_id.slice(0, 10)}…</span>
          </span>
        )}
      </div>
      {msg.content && (
        <pre className="code whitespace-pre-wrap text-slate-200">{truncated}</pre>
      )}
      {msg.tool_calls && msg.tool_calls.length > 0 && (
        <div className="mt-2 space-y-2">
          {msg.tool_calls.map((tc, i) => (
            <div key={i} className="rounded-md border border-violet-500/30 bg-violet-500/5 p-2 text-xs">
              <div className="mb-1 flex items-center gap-2">
                <Badge tone="violet">tool_call</Badge>
                <span className="font-mono text-violet-200">{tc.name}</span>
              </div>
              <JsonView value={tc.args} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
