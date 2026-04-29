import { useEffect, useState } from "react";
import { api } from "../api";
import { AgentTimeline } from "../components/AgentTimeline";
import { ProposedCallCard } from "../components/ProposedCallCard";
import { PlayIcon, SparkIcon, XIcon } from "../components/icons";
import {
  Badge,
  Button,
  Callout,
  Card,
  CodeBlock,
  ErrorBox,
  Field,
  Section,
  Spinner,
  StatPill,
  Textarea,
  Toggle,
} from "../components/ui";
import { LangGraphAgentResponse, PermissionsResponse } from "../types";

interface Preset {
  label: string;
  goal: string;
  what: string;
  defaults?: { allowedTools?: string[]; requireApproval?: boolean };
}

const PRESETS: Preset[] = [
  {
    label: "Multi-step: RAG + calculator",
    goal: "What is the per-night hotel cap, and how much is that for a 5 night trip?",
    what:
      "Two visible steps: knowledge_search retrieves the Travel policy chunk, calculator multiplies it. Watch them appear in order in the timeline.",
  },
  {
    label: "Multi-step: lookup + write (HITL)",
    goal: "Look up Bob in engineering, then create a task to schedule a 1:1 with him",
    what:
      "Three steps. Read first (employee_lookup), then a write — agent pauses before task_manager. Approve to see step 2 land.",
    defaults: { requireApproval: true },
  },
  {
    label: "Recovery from tool error",
    goal: "Mark task 999 as done, then list everything",
    what:
      "Task 999 doesn't exist; the tool returns a structured error. The agent reads it as data and recovers by listing the real tasks.",
  },
  {
    label: "RAG: semantic retrieval",
    goal: "how many days off do I get a year?",
    what:
      "Doc title is 'Annual Leave' — zero word overlap. Keyword search would miss this.",
  },
  {
    label: "Permission scope: knowledge_search only",
    goal: "What is our remote work policy?",
    what:
      "The LLM only sees one tool. Forced to use knowledge_search.",
    defaults: { allowedTools: ["knowledge_search"] },
  },
  {
    label: "Permission scope: read-only",
    goal: "Create a task to onboard Alice",
    what:
      "task_manager (write) is hidden. Watch the agent decline politely instead of inventing a task.",
    defaults: {
      allowedTools: ["calculator", "clock", "knowledge_search", "employee_lookup"],
    },
  },
];

export function AgentRunView() {
  const [permissions, setPermissions] = useState<PermissionsResponse | null>(null);
  const [permsErr, setPermsErr] = useState<unknown>(null);

  const [goal, setGoal] = useState(PRESETS[0].goal);
  const [allowedTools, setAllowedTools] = useState<string[] | null>(null);
  const [requireApproval, setRequireApproval] = useState(false);
  const [maxSteps, setMaxSteps] = useState(10);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [response, setResponse] = useState<LangGraphAgentResponse | null>(null);
  const [resumed, setResumed] = useState<LangGraphAgentResponse | null>(null);

  const [decisionBusy, setDecisionBusy] = useState(false);

  useEffect(() => {
    api.permissions().then(setPermissions).catch(setPermsErr);
  }, []);

  const allTools = permissions ? Object.keys(permissions.permissions) : [];

  function applyPreset(p: Preset) {
    setGoal(p.goal);
    setAllowedTools(p.defaults?.allowedTools ?? null);
    setRequireApproval(p.defaults?.requireApproval ?? false);
    setError(null);
    setResponse(null);
    setResumed(null);
  }

  function toggleTool(name: string) {
    if (allowedTools == null) {
      setAllowedTools(allTools.filter((t) => t !== name));
      return;
    }
    if (allowedTools.includes(name)) {
      setAllowedTools(allowedTools.filter((t) => t !== name));
    } else {
      setAllowedTools([...allowedTools, name]);
    }
  }

  async function run() {
    setLoading(true);
    setError(null);
    setResponse(null);
    setResumed(null);
    try {
      const r = await api.agentRun({
        goal,
        allowed_tools: allowedTools,
        require_approval: requireApproval,
        max_steps: maxSteps,
      });
      setResponse(r);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  async function decide(approved: boolean) {
    if (!response?.thread_id) return;
    setDecisionBusy(true);
    try {
      const r = await api.agentApprove(response.thread_id, approved);
      setResumed(r);
    } catch (e) {
      setError(e);
    } finally {
      setDecisionBusy(false);
    }
  }

  const finished = resumed ?? (response?.status === "completed" ? response : null);

  return (
    <div className="space-y-6">
      <Section
        title="Run the agent"
        subtitle="Watch each tool call land in the timeline. Toggle require_approval to pause before tools fire."
        icon={<SparkIcon size={18} />}
      >
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <Card className="space-y-4 p-5">
            <Field label="Goal">
              <Textarea value={goal} onChange={setGoal} rows={3} />
            </Field>

            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Presets
              </div>
              <div className="flex flex-wrap gap-2">
                {PRESETS.map((p) => (
                  <Button
                    key={p.label}
                    size="sm"
                    variant="secondary"
                    onClick={() => applyPreset(p)}
                    title={p.what}
                  >
                    {p.label}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Button onClick={run} disabled={loading || !goal.trim()}>
                {loading ? <Spinner /> : <PlayIcon size={14} />}
                {loading ? "Running…" : "Run agent"}
              </Button>
              <span className="text-xs text-slate-500">
                {requireApproval
                  ? "Will pause before any tool runs."
                  : "Runs end-to-end."}
              </span>
            </div>
          </Card>

          <Card className="space-y-4 p-5">
            <h3 className="text-sm font-semibold text-slate-100">Settings</h3>

            <Toggle
              checked={requireApproval}
              onChange={setRequireApproval}
              label="require_approval"
              hint="Wires interrupt_before=['tools'] so the agent pauses for HITL."
            />

            <Field label="max_steps">
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={1}
                  max={25}
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(Number(e.target.value))}
                  className="flex-1 accent-sky-500"
                />
                <span className="w-8 text-right font-mono text-sm">{maxSteps}</span>
              </div>
            </Field>

            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Allowed tools
                </span>
                <button
                  className="text-xs text-sky-300 hover:underline"
                  onClick={() => setAllowedTools(null)}
                  title="Pass null — expose every registered tool"
                >
                  expose all
                </button>
              </div>
              {permsErr ? <ErrorBox error={permsErr} /> : null}
              {!permissions ? (
                <div className="mt-2 text-xs text-slate-500">Loading…</div>
              ) : (
                <div className="mt-2 space-y-1">
                  {allTools.map((name) => {
                    const allowed = allowedTools == null || allowedTools.includes(name);
                    const perm = permissions.permissions[name];
                    return (
                      <label
                        key={name}
                        className="flex items-center gap-2 rounded-md px-2 py-1 text-sm hover:bg-slate-800/50"
                      >
                        <input
                          type="checkbox"
                          checked={allowed}
                          onChange={() => toggleTool(name)}
                          className="h-3.5 w-3.5 accent-sky-500"
                        />
                        <span className="font-mono text-slate-200">{name}</span>
                        <Badge tone={perm === "write" ? "rw-write" : "rw-read"}>
                          {perm}
                        </Badge>
                      </label>
                    );
                  })}
                  <div className="pt-1 text-[11px] text-slate-500">
                    {allowedTools == null
                      ? "Sending allowed_tools = null"
                      : `Sending allowed_tools = [${allowedTools.length}]`}
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>
      </Section>

      <Section title="Request preview">
        <CodeBlock>
          {JSON.stringify(
            {
              method: "POST",
              path: "/agent/run",
              body: {
                goal,
                allowed_tools: allowedTools,
                require_approval: requireApproval,
                max_steps: maxSteps,
              },
            },
            null,
            2
          )}
        </CodeBlock>
      </Section>

      <ErrorBox error={error} />

      {response && (
        <Section title="Result">
          <Card className="space-y-4 p-5">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <StatPill
                label="Status"
                value={response.status}
                tone={
                  response.status === "paused"
                    ? "warn"
                    : response.status === "completed"
                    ? "ok"
                    : "warn"
                }
              />
              <StatPill
                label="Steps"
                value={(resumed?.steps_completed ?? response.steps_completed) || 0}
                tone="info"
              />
              <StatPill label="Engine" value={response.engine} tone="info" />
              <StatPill label="Model" value={<span className="text-sm">{response.model}</span>} tone="info" />
            </div>

            <div className="text-xs text-slate-500">
              thread_id:{" "}
              <span className="font-mono text-slate-300">{response.thread_id}</span>
            </div>

            {response.status === "paused" && response.proposed_tool_call && !resumed && (
              <>
                <ProposedCallCard call={response.proposed_tool_call} />
                {response.steps.length > 0 && (
                  <div>
                    <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Steps already executed
                    </div>
                    <AgentTimeline steps={response.steps} />
                  </div>
                )}
                <div className="flex items-center gap-2 border-t border-slate-700/60 pt-4">
                  <Button
                    variant="approve"
                    onClick={() => decide(true)}
                    disabled={decisionBusy}
                  >
                    {decisionBusy ? <Spinner /> : <PlayIcon size={14} />}Approve
                  </Button>
                  <Button
                    variant="reject"
                    onClick={() => decide(false)}
                    disabled={decisionBusy}
                  >
                    <XIcon size={14} />Reject
                  </Button>
                </div>
              </>
            )}

            {finished && (
              <>
                <div>
                  <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Step timeline
                  </div>
                  <AgentTimeline steps={finished.steps} />
                </div>
                <div>
                  <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Final answer
                  </div>
                  {finished.final_answer ? (
                    <div className="rounded-lg border border-slate-700/70 bg-slate-950/60 p-4 text-sm leading-relaxed text-slate-100 whitespace-pre-wrap">
                      {finished.final_answer}
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500">No final answer.</div>
                  )}
                </div>
              </>
            )}

            {resumed?.status === "rejected" && (
              <Callout tone="danger" title="Rejected">
                The proposed tool call was not executed. The thread is closed.
              </Callout>
            )}
          </Card>
        </Section>
      )}
    </div>
  );
}
