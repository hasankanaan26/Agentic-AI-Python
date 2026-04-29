import { useEffect, useState } from "react";
import { api } from "../api";
import {
  Badge,
  Button,
  Card,
  ErrorBox,
  Field,
  JsonView,
  Section,
  Spinner,
  Textarea,
} from "../components/ui";
import { ToolCallResponse, ToolDefinition } from "../types";

const SINGLE_CALL_PROMPTS = [
  "What is 147 times 23?",
  "What time is it?",
  "Look up Alice Chen",
  "What is our parental leave policy?",
  "Show me all the open tasks",
];

export function ToolsView() {
  return (
    <div className="space-y-8">
      <ToolListPanel />
      <SingleCallPanel />
    </div>
  );
}

function ToolListPanel() {
  const [tools, setTools] = useState<ToolDefinition[] | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api
      .listTools()
      .then((r) => setTools(r.tools))
      .catch(setError);
  }, []);

  return (
    <Section
      title="Registered tools"
      subtitle="The JSON schemas the LLM sees as function declarations. Same source of truth as the LangChain args_schemas the LangGraph agent uses."
    >
      <ErrorBox error={error} />
      {!tools ? (
        <Spinner />
      ) : (
        <div className="grid gap-3">
          {tools.map((t) => (
            <Card key={t.name} className="p-5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-base font-semibold text-slate-100">
                  {t.name}
                </span>
                <Badge tone="info">
                  {t.parameters.required?.length ?? 0} required arg(s)
                </Badge>
              </div>
              <p className="mt-2 text-sm text-slate-300">{t.description}</p>
              <div className="mt-3 grid gap-2">
                {Object.entries(t.parameters.properties ?? {}).map(([k, v]) => (
                  <div
                    key={k}
                    className="rounded-md border border-slate-800 bg-slate-950/70 px-3 py-2 text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-slate-200">{k}</span>
                      <Badge tone="neutral">{v.type ?? "any"}</Badge>
                      {(t.parameters.required ?? []).includes(k) && (
                        <Badge tone="warn">required</Badge>
                      )}
                      {v.enum && (
                        <span className="font-mono text-[11px] text-slate-400">
                          enum: {v.enum.join(" | ")}
                        </span>
                      )}
                    </div>
                    {v.description && (
                      <div className="mt-1 text-slate-400">{v.description}</div>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </Section>
  );
}

function SingleCallPanel() {
  const [message, setMessage] = useState(SINGLE_CALL_PROMPTS[0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<ToolCallResponse | null>(null);

  async function call() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.callTool(message));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Single-turn tool call"
      subtitle="POST /tools/call — the LLM picks at most one tool and returns its result. This is the CP1 surface; useful for sanity-testing individual tools without running the full agent loop."
    >
      <Card className="space-y-4 p-5">
        <Field label="User message">
          <Textarea value={message} onChange={setMessage} rows={2} />
        </Field>
        <div className="flex flex-wrap gap-2">
          {SINGLE_CALL_PROMPTS.map((p) => (
            <Button key={p} size="sm" variant="secondary" onClick={() => setMessage(p)}>
              {p}
            </Button>
          ))}
        </div>
        <Button onClick={call} disabled={busy || !message.trim()}>
          {busy ? <Spinner /> : null}Call
        </Button>
        <ErrorBox error={error} />

        {result && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={result.tool_called ? "info" : "neutral"}>
                {result.tool_called ? "tool called" : "no tool"}
              </Badge>
              {result.tool_status && (
                <Badge tone={result.tool_status === "error" ? "error" : "ok"}>
                  status: {result.tool_status}
                </Badge>
              )}
              <Badge tone="neutral">model: {result.model}</Badge>
            </div>
            {result.tool_call && (
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Tool call
                </div>
                <JsonView value={result.tool_call} />
              </div>
            )}
            {result.tool_result && (
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Tool result
                </div>
                <pre className="code mt-1 max-h-72 overflow-auto rounded-lg border border-slate-800 bg-slate-950/80 p-3 whitespace-pre-wrap text-slate-200">
                  {result.tool_result}
                </pre>
              </div>
            )}
            {result.llm_response && (
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Direct LLM response (no tool needed)
                </div>
                <pre className="code mt-1 rounded-lg border border-slate-800 bg-slate-950/80 p-3 whitespace-pre-wrap text-slate-200">
                  {result.llm_response}
                </pre>
              </div>
            )}
          </div>
        )}
      </Card>
    </Section>
  );
}
