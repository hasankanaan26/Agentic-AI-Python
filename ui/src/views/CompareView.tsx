import { useState } from "react";
import { api } from "../api";
import { AgentTimeline } from "../components/AgentTimeline";
import {
  Badge,
  Button,
  Callout,
  Card,
  ErrorBox,
  Field,
  Section,
  Spinner,
  Textarea,
} from "../components/ui";
import { AgentResponse, LangGraphAgentResponse } from "../types";

export function CompareView() {
  const [goal, setGoal] = useState(
    "Look up Bob in engineering, then create a task to schedule a 1:1 with him"
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [lg, setLg] = useState<LangGraphAgentResponse | null>(null);
  const [raw, setRaw] = useState<AgentResponse | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setLg(null);
    setRaw(null);
    try {
      const [a, b] = await Promise.all([
        api.agentRun({ goal }),
        api.agentRunRaw({ goal }),
      ]);
      setLg(a);
      setRaw(b);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <Section
        title="LangGraph vs the raw loop"
        subtitle="Same goal, both engines. Compare step counts, ordering, and the metadata each engine produces."
      >
        <Card className="space-y-3 p-5">
          <Field label="Goal">
            <Textarea value={goal} onChange={setGoal} rows={2} />
          </Field>
          <Button onClick={run} disabled={busy || !goal.trim()}>
            {busy ? <Spinner /> : null}Run both
          </Button>
          <ErrorBox error={error} />
        </Card>
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <EnginePanel
          title="LangGraph (/agent/run)"
          description="create_react_agent, async ainvoke, checkpointer, interrupt_before hook."
          response={lg}
          extras={
            lg && (
              <Badge tone="info">
                thread_id: <span className="ml-1 font-mono">{lg.thread_id}</span>
              </Badge>
            )
          }
        />
        <EnginePanel
          title="Raw loop (/agent/run-raw)"
          description="The CP2 think-act-observe loop. No checkpointing, no streaming, no interrupts."
          response={raw ? { ...raw, thread_id: "—", engine: "raw" } : null}
        />
      </div>

      <Callout tone="info" title="What to notice">
        <ul className="list-disc space-y-1 pl-5 text-sm">
          <li>
            The LangGraph response carries a <code className="font-mono">thread_id</code>; the raw
            loop has no equivalent because it has no persistent state to resume.
          </li>
          <li>
            Both engines call the same registry tools — the difference is the
            framework around them. The raw loop has to assemble messages and parse
            tool calls itself; LangGraph's prebuilt graph does that for you.
          </li>
          <li>
            Step counts may differ when the LLM phrasing of the prompt shifts —
            LangGraph uses a LangChain chat model with a system prompt; the raw
            loop uses our SDK wrapper.
          </li>
        </ul>
      </Callout>
    </div>
  );
}

function EnginePanel({
  title,
  description,
  response,
  extras,
}: {
  title: string;
  description: string;
  response: (LangGraphAgentResponse | (AgentResponse & { thread_id: string; engine: string })) | null;
  extras?: React.ReactNode;
}) {
  return (
    <Card className="space-y-4 p-5">
      <div>
        <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
        <p className="mt-1 text-xs text-slate-400">{description}</p>
      </div>
      {!response ? (
        <div className="text-xs text-slate-500">
          Run a goal to populate this side.
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="neutral">model: {response.model}</Badge>
            <Badge tone={response.steps_completed > 0 ? "ok" : "warn"}>
              steps: {response.steps_completed}
            </Badge>
            {extras}
          </div>
          <AgentTimeline steps={response.steps} />
          {response.final_answer && (
            <div>
              <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Final answer
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-sm text-slate-100 whitespace-pre-wrap">
                {response.final_answer}
              </div>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
