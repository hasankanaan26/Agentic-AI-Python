import { AgentStep } from "../types";
import { Badge, Card, JsonView, StatusIcon, useStaggeredReveal } from "./ui";

const TOOL_TONE: Record<string, "info" | "violet" | "ok" | "warn"> = {
  calculator: "info",
  clock: "info",
  knowledge_search: "violet",
  employee_lookup: "ok",
  task_manager: "warn",
};

export function AgentTimeline({ steps }: { steps: AgentStep[] }) {
  const shown = useStaggeredReveal(steps.length, 180);

  if (steps.length === 0) {
    return (
      <Card className="p-6 text-center text-sm text-slate-400">
        No tool calls yet. Run a goal and the timeline will fill in step by step.
      </Card>
    );
  }

  return (
    <ol className="relative ml-3 space-y-3 border-l-2 border-slate-700/60 pl-6">
      {steps.slice(0, shown).map((s) => {
        const tone = TOOL_TONE[s.tool_name] ?? "neutral";
        return (
          <li key={s.step} className="animate-step relative">
            <span
              className={`absolute -left-[35px] top-3 flex h-7 w-7 items-center justify-center rounded-full border-2 text-[12px] font-bold ${
                s.tool_status === "error"
                  ? "border-rose-400 bg-rose-500/20 text-rose-200"
                  : "border-sky-400 bg-sky-500/20 text-sky-200"
              }`}
            >
              {s.step}
            </span>
            <Card className="p-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-sm font-semibold text-slate-50">
                  {s.tool_name}
                </span>
                <Badge tone={tone}>tool</Badge>
                <Badge tone={s.tool_status === "error" ? "error" : "ok"}>
                  <StatusIcon status={s.tool_status === "error" ? "error" : "ok"} />
                  {s.tool_status === "error" ? "error" : "ok"}
                </Badge>
              </div>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Input
                  </div>
                  <div className="mt-1">
                    <JsonView value={s.tool_input} />
                  </div>
                </div>
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Output
                  </div>
                  <pre className="code mt-1 max-h-72 overflow-auto rounded-lg border border-slate-700/70 bg-slate-950/80 p-3 whitespace-pre-wrap text-slate-200">
                    {s.tool_output || "(empty)"}
                  </pre>
                </div>
              </div>
            </Card>
          </li>
        );
      })}
      {shown < steps.length && (
        <li className="text-xs italic text-slate-500">Revealing step {shown + 1}…</li>
      )}
    </ol>
  );
}
