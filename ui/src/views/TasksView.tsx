import { useEffect, useState } from "react";
import { api } from "../api";
import { AgentTimeline } from "../components/AgentTimeline";
import { ProposedCallCard } from "../components/ProposedCallCard";
import {
  CheckIcon,
  ListIcon,
  PauseIcon,
  PlayIcon,
  RefreshIcon,
  XIcon,
} from "../components/icons";
import {
  Badge,
  Button,
  Callout,
  Card,
  ErrorBox,
  Field,
  Input,
  Section,
  Spinner,
  StatPill,
  StatusIcon,
} from "../components/ui";
import { LangGraphAgentResponse, Task } from "../types";

type RunState = {
  goal: string;
  response: LangGraphAgentResponse | null;
  busy: boolean;
  resumed: LangGraphAgentResponse | null;
};

const initialRun: RunState = {
  goal: "",
  response: null,
  busy: false,
  resumed: null,
};

export function TasksView() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [tasksMeta, setTasksMeta] = useState({ open: 0, done: 0, total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  const [newTitle, setNewTitle] = useState("Onboard the new analytics hire");
  const [run, setRun] = useState<RunState>(initialRun);

  async function refresh() {
    try {
      const r = await api.listTasks();
      setTasks(r.tasks);
      setTasksMeta({ open: r.open, done: r.done, total: r.total });
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function startWrite(goal: string) {
    setRun({ goal, response: null, busy: true, resumed: null });
    setError(null);
    try {
      const response = await api.agentRun({
        goal,
        require_approval: true,
        allowed_tools: ["task_manager"],
      });
      setRun({ goal, response, busy: false, resumed: null });
    } catch (e) {
      setError(e);
      setRun(initialRun);
    }
  }

  async function decide(approved: boolean) {
    if (!run.response?.thread_id) return;
    setRun((r) => ({ ...r, busy: true }));
    try {
      const resumed = await api.agentApprove(run.response.thread_id, approved);
      setRun((r) => ({ ...r, resumed, busy: false }));
      await refresh();
    } catch (e) {
      setError(e);
      setRun((r) => ({ ...r, busy: false }));
    }
  }

  function clearRun() {
    setRun(initialRun);
  }

  return (
    <div className="space-y-6">
      <Section
        title="Tasks"
        subtitle="Live view of the task_manager state. Reads come from /tasks/list directly; writes route through /agent/run with require_approval=true so you can see the HITL gate in action."
        icon={<ListIcon size={18} />}
      >
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatPill label="Total" value={tasksMeta.total} tone="info" />
          <StatPill label="Open" value={tasksMeta.open} tone="warn" />
          <StatPill label="Done" value={tasksMeta.done} tone="ok" />
          <Card className="flex items-center justify-center p-2">
            <Button size="sm" variant="ghost" onClick={refresh}>
              <RefreshIcon size={12} /> Refresh
            </Button>
          </Card>
        </div>

        <ErrorBox error={error} />

        <Card className="overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center p-10">
              <Spinner size={20} />
            </div>
          ) : tasks.length === 0 ? (
            <div className="p-10 text-center text-sm text-slate-400">
              No tasks yet. Use the form below to propose one.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-slate-900/80 text-left text-[11px] uppercase tracking-wide text-slate-400">
                <tr>
                  <th className="px-4 py-2.5 w-16">ID</th>
                  <th className="px-4 py-2.5">Title</th>
                  <th className="px-4 py-2.5 w-32">Status</th>
                  <th className="px-4 py-2.5 w-44 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr
                    key={t.id}
                    className="border-t border-slate-800 transition-colors hover:bg-slate-800/30"
                  >
                    <td className="px-4 py-3 font-mono text-slate-300">#{t.id}</td>
                    <td
                      className={`px-4 py-3 ${
                        t.done ? "text-slate-500 line-through" : "text-slate-100"
                      }`}
                    >
                      {t.title}
                    </td>
                    <td className="px-4 py-3">
                      {t.done ? (
                        <Badge tone="ok">
                          <StatusIcon status="ok" /> done
                        </Badge>
                      ) : (
                        <Badge tone="warn">
                          <PauseIcon size={11} /> open
                        </Badge>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {!t.done && (
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => startWrite(`Mark task ${t.id} as done`)}
                          disabled={run.busy}
                        >
                          <CheckIcon size={12} /> Mark done
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </Section>

      <Section
        title="Propose a new task"
        subtitle="The agent will pause before writing — you approve below."
      >
        <Card className="space-y-3 p-5">
          <Field label="Task title">
            <Input value={newTitle} onChange={setNewTitle} />
          </Field>
          <Button
            onClick={() => startWrite(`Create a task titled '${newTitle.replace(/'/g, "")}'`)}
            disabled={!newTitle.trim() || run.busy}
          >
            <PlayIcon size={14} /> Propose creation
          </Button>
        </Card>
      </Section>

      {run.response && <RunCard run={run} onDecide={decide} onClear={clearRun} />}

      <Callout tone="info" title="Why writes go through the agent">
        Routing writes through <code className="font-mono">/agent/run</code> with{" "}
        <code className="font-mono">allowed_tools=['task_manager']</code> and{" "}
        <code className="font-mono">require_approval=true</code> means every mutation
        gets the same HITL treatment — no special "danger" endpoint, no parallel
        approval system. The agent layer is the gate.
      </Callout>
    </div>
  );
}

function RunCard({
  run,
  onDecide,
  onClear,
}: {
  run: RunState;
  onDecide: (approved: boolean) => void;
  onClear: () => void;
}) {
  const r = run.response!;
  return (
    <Section
      title="Approval flow"
      subtitle={`Goal: "${run.goal}"`}
    >
      <Card className="space-y-4 p-5">
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            tone={
              r.status === "paused"
                ? "warn"
                : r.status === "rejected"
                ? "error"
                : "ok"
            }
          >
            {r.status}
          </Badge>
          <Badge tone="neutral">
            thread_id: <span className="ml-1 font-mono">{r.thread_id}</span>
          </Badge>
          <Button size="sm" variant="ghost" onClick={onClear}>
            Clear
          </Button>
        </div>

        {r.status === "paused" && r.proposed_tool_call && (
          <>
            <ProposedCallCard call={r.proposed_tool_call} />
            <div className="flex items-center gap-2 border-t border-slate-700/60 pt-4">
              <Button
                variant="approve"
                onClick={() => onDecide(true)}
                disabled={run.busy}
              >
                {run.busy ? <Spinner /> : <PlayIcon size={14} />}Approve
              </Button>
              <Button
                variant="reject"
                onClick={() => onDecide(false)}
                disabled={run.busy}
              >
                <XIcon size={14} />Reject
              </Button>
            </div>
          </>
        )}

        {run.resumed && (
          <div className="space-y-3 border-t border-slate-700/60 pt-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={run.resumed.status === "rejected" ? "error" : "ok"}>
                resume: {run.resumed.status}
              </Badge>
              <Badge tone="neutral">
                steps: {run.resumed.steps_completed}
              </Badge>
            </div>
            {run.resumed.steps.length > 0 && (
              <AgentTimeline steps={run.resumed.steps} />
            )}
            {run.resumed.final_answer && (
              <div className="rounded-lg border border-slate-700/70 bg-slate-950/60 p-4 text-sm text-slate-100 whitespace-pre-wrap">
                {run.resumed.final_answer}
              </div>
            )}
          </div>
        )}
      </Card>
    </Section>
  );
}
