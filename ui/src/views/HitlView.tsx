import { useEffect, useState } from "react";
import { api } from "../api";
import { AgentTimeline } from "../components/AgentTimeline";
import { ProposedCallCard } from "../components/ProposedCallCard";
import { HitlIcon, PauseIcon, PlayIcon, RefreshIcon, XIcon } from "../components/icons";
import {
  Badge,
  Button,
  Callout,
  Card,
  ErrorBox,
  Field,
  Section,
  Spinner,
  StatPill,
  Textarea,
} from "../components/ui";
import { LangGraphAgentResponse, PendingThread } from "../types";

type Stage = "idle" | "running" | "paused" | "resuming" | "completed" | "rejected";

const PRESETS = [
  {
    label: "Create a task",
    goal: "Create a task to deprecate the legacy invoice service",
    why: "task_manager is a write tool. require_approval=true → the run pauses with the proposed args ready for review.",
  },
  {
    label: "Complete a task",
    goal: "Mark task 1 as done",
    why: "Another write — the agent will pause before flipping the done flag.",
  },
  {
    label: "Read-then-write chain",
    goal: "Look up Alice in engineering, then create a task to schedule a 1:1 with her",
    why: "Two-step run. employee_lookup runs immediately (read), then the agent pauses before task_manager (write).",
  },
];

export function HitlView() {
  const [goal, setGoal] = useState(PRESETS[0].goal);
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<unknown>(null);
  const [pauseRes, setPauseRes] = useState<LangGraphAgentResponse | null>(null);
  const [resumeRes, setResumeRes] = useState<LangGraphAgentResponse | null>(null);
  const [pending, setPending] = useState<PendingThread[]>([]);

  async function refreshPending() {
    try {
      const r = await api.agentPending();
      setPending(r.pending);
    } catch (e) {
      // non-fatal
      console.warn(e);
    }
  }

  useEffect(() => {
    refreshPending();
  }, []);

  async function start() {
    setStage("running");
    setError(null);
    setPauseRes(null);
    setResumeRes(null);
    try {
      const r = await api.agentRun({ goal, require_approval: true });
      setPauseRes(r);
      if (r.status === "paused") setStage("paused");
      else setStage("completed");
      refreshPending();
    } catch (e) {
      setError(e);
      setStage("idle");
    }
  }

  async function decide(approved: boolean) {
    if (!pauseRes?.thread_id) return;
    setStage("resuming");
    setError(null);
    try {
      const r = await api.agentApprove(pauseRes.thread_id, approved);
      setResumeRes(r);
      setStage(approved ? "completed" : "rejected");
      refreshPending();
    } catch (e) {
      setError(e);
      setStage("paused");
    }
  }

  function reset() {
    setStage("idle");
    setError(null);
    setPauseRes(null);
    setResumeRes(null);
  }

  return (
    <div className="space-y-6">
      <Section
        title="Human-in-the-loop, end to end"
        subtitle="A goal that requires a write tool, paused at the approval gate, then resumed in place. The thread_id is the same across both calls — LangGraph's checkpointer is what makes that work."
        icon={<HitlIcon size={18} />}
      >
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <Card className="space-y-4 p-5">
            <Field label="Goal" hint="The agent runs with require_approval=true and pauses before the first tool.">
              <Textarea value={goal} onChange={setGoal} rows={2} disabled={stage !== "idle"} />
            </Field>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <Button
                  key={p.label}
                  size="sm"
                  variant="secondary"
                  onClick={() => setGoal(p.goal)}
                  title={p.why}
                  disabled={stage !== "idle"}
                >
                  {p.label}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={start}
                disabled={stage !== "idle" || !goal.trim()}
              >
                {stage === "running" ? <Spinner /> : <PlayIcon size={14} />}
                {stage === "running" ? "Submitting…" : "Submit for review"}
              </Button>
              {stage !== "idle" && (
                <Button variant="ghost" size="sm" onClick={reset}>
                  Start over
                </Button>
              )}
            </div>
          </Card>

          <Card className="space-y-3 p-5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-100">Inbox</span>
              <button
                onClick={refreshPending}
                className="inline-flex items-center gap-1 text-xs text-sky-300 hover:underline"
              >
                <RefreshIcon size={12} /> refresh
              </button>
            </div>
            <p className="text-xs text-slate-400">
              Threads currently waiting on a human. Backed by{" "}
              <code className="font-mono">GET /agent/pending</code>.
            </p>
            {pending.length === 0 ? (
              <div className="rounded-md border border-dashed border-slate-700 px-3 py-4 text-center text-xs text-slate-500">
                No paused threads.
              </div>
            ) : (
              <ul className="space-y-1.5">
                {pending.map((t) => (
                  <li
                    key={t.thread_id}
                    className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-xs"
                  >
                    <div className="flex items-center gap-1.5">
                      <PauseIcon size={11} className="text-amber-300" />
                      <span className="font-mono text-amber-200">{t.thread_id}</span>
                    </div>
                    <div className="mt-1 truncate text-slate-300" title={t.goal}>
                      {t.goal}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>

        <ErrorBox error={error} />

        {pauseRes && <FlowTracker stage={stage} threadId={pauseRes.thread_id} />}

        {pauseRes && pauseRes.status === "completed" && pauseRes.steps.length > 0 && (
          <Card className="p-5">
            <Callout tone="success" title="Run completed without pausing">
              The agent finished using only read tools (no write tool was reached). Try a
              preset that involves <code className="font-mono">task_manager</code> to see
              the approval gate.
            </Callout>
            <div className="mt-4">
              <AgentTimeline steps={pauseRes.steps} />
            </div>
          </Card>
        )}

        {pauseRes && pauseRes.status === "paused" && pauseRes.proposed_tool_call && (
          <Card className="space-y-4 p-5">
            <ProposedCallCard call={pauseRes.proposed_tool_call} />

            {pauseRes.steps.length > 0 && (
              <div>
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Steps already taken
                </div>
                <AgentTimeline steps={pauseRes.steps} />
              </div>
            )}

            <div className="flex items-center gap-2 border-t border-slate-700/60 pt-4">
              <Button
                variant="approve"
                onClick={() => decide(true)}
                disabled={stage === "resuming"}
              >
                {stage === "resuming" ? <Spinner /> : <PlayIcon size={14} />}Approve & resume
              </Button>
              <Button
                variant="reject"
                onClick={() => decide(false)}
                disabled={stage === "resuming"}
              >
                <XIcon size={14} />Reject
              </Button>
              <span className="ml-auto text-xs text-slate-500">
                Approve calls{" "}
                <code className="font-mono text-slate-300">POST /agent/approve</code>{" "}
                with the same{" "}
                <code className="font-mono text-slate-300">thread_id</code>.
              </span>
            </div>
          </Card>
        )}

        {resumeRes && (
          <Card className="space-y-4 p-5">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                tone={
                  resumeRes.status === "rejected"
                    ? "error"
                    : resumeRes.status === "completed"
                    ? "ok"
                    : "warn"
                }
              >
                {resumeRes.status}
              </Badge>
              <Badge tone="neutral">
                thread_id:{" "}
                <span className="ml-1 font-mono">{resumeRes.thread_id}</span>
              </Badge>
              <Badge tone="info">model: {resumeRes.model}</Badge>
            </div>

            {resumeRes.status === "rejected" ? (
              <Callout tone="danger" title="Rejected">
                The proposed tool call was not executed. The pending thread was
                discarded; <code className="font-mono">/agent/pending</code> no
                longer lists it.
              </Callout>
            ) : (
              <>
                <div>
                  <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Steps executed after approval
                  </div>
                  <AgentTimeline steps={resumeRes.steps} />
                </div>
                {resumeRes.final_answer && (
                  <div>
                    <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Final answer
                    </div>
                    <div className="rounded-lg border border-slate-700/70 bg-slate-950/60 p-4 text-sm leading-relaxed text-slate-100 whitespace-pre-wrap">
                      {resumeRes.final_answer}
                    </div>
                  </div>
                )}
              </>
            )}
          </Card>
        )}
      </Section>

      <Callout tone="info" title="Why this works">
        <ol className="list-decimal space-y-1 pl-5 text-sm">
          <li>
            <code className="font-mono">/agent/run</code> is built with{" "}
            <code className="font-mono">interrupt_before=["tools"]</code>. LangGraph
            halts at the tool node and returns the state.
          </li>
          <li>
            The runner extracts the pending{" "}
            <code className="font-mono">tool_calls[0]</code> from the last AIMessage and
            stores the request context against the{" "}
            <code className="font-mono">thread_id</code> in memory.
          </li>
          <li>
            On approve, <code className="font-mono">/agent/approve</code> rebuilds the
            graph WITHOUT the interrupt and calls{" "}
            <code className="font-mono">ainvoke(None, …)</code> with the same{" "}
            <code className="font-mono">thread_id</code>. The MemorySaver picks up
            from the checkpoint and runs the proposed tool to completion.
          </li>
          <li>
            On reject, the pending entry is dropped and the resume call returns
            a <code className="font-mono">status="rejected"</code> response without
            invoking any tool.
          </li>
        </ol>
      </Callout>
    </div>
  );
}

function FlowTracker({ stage, threadId }: { stage: Stage; threadId: string }) {
  const stages: { key: Stage[]; label: string; tone: "info" | "warn" | "ok" }[] = [
    { key: ["running", "paused", "resuming", "completed", "rejected"], label: "Submitted", tone: "info" },
    { key: ["paused", "resuming", "completed", "rejected"], label: "Paused for approval", tone: "warn" },
    { key: ["resuming", "completed", "rejected"], label: "Decision recorded", tone: "info" },
    { key: ["completed", "rejected"], label: stage === "rejected" ? "Rejected" : "Completed", tone: "ok" },
  ];
  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-center gap-3">
        <Badge tone="neutral">
          thread_id: <span className="ml-1 font-mono">{threadId}</span>
        </Badge>
        <div className="flex flex-1 flex-wrap items-center gap-2">
          {stages.map((s, i) => {
            const reached = s.key.includes(stage);
            return (
              <div key={i} className="flex items-center gap-2">
                <StatPill
                  label={`Stage ${i + 1}`}
                  value={s.label}
                  tone={reached ? s.tone : "info"}
                />
                {i < stages.length - 1 && (
                  <span className="text-slate-600">→</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
