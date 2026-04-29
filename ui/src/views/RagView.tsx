import { useEffect, useState } from "react";
import { api } from "../api";
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
  Textarea,
  Toggle,
} from "../components/ui";
import { IngestResponse, LangGraphAgentResponse, RagStatus } from "../types";

const RAG_QUERIES = [
  "how many days off do I get?",
  "can I work from my couch?",
  "what's the laptop budget for new hires?",
  "what is our Git workflow?",
];

export function RagView() {
  return (
    <div className="space-y-8">
      <StatusPanel />
      <SearchPanel />
      <Callout tone="info" title="Why this is more than keyword search">
        The query is embedded into the same vector space as the docs, then
        scored by cosine similarity. "How many days off do I get?" matches the
        doc titled "Annual Leave" because the embeddings of those phrases are
        close — even though no word overlaps. Distance scores are shown in the
        tool output below.
      </Callout>
    </div>
  );
}

function StatusPanel() {
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [statusErr, setStatusErr] = useState<unknown>(null);
  const [force, setForce] = useState(false);
  const [ingestRes, setIngestRes] = useState<IngestResponse | null>(null);
  const [ingestErr, setIngestErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setStatusErr(null);
    try {
      setStatus(await api.ragStatus());
    } catch (e) {
      setStatusErr(e);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function ingest() {
    setBusy(true);
    setIngestErr(null);
    setIngestRes(null);
    try {
      setIngestRes(await api.ragIngest(force));
      await refresh();
    } catch (e) {
      setIngestErr(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Index status & ingestion"
      subtitle="The Acme knowledge file is small (one chunk per topic). Ingest is idempotent — pass force=true to re-embed."
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="space-y-3 p-5">
          <div className="text-sm font-semibold text-slate-100">Current state</div>
          <ErrorBox error={statusErr} />
          {status ? (
            <>
              <div className="flex items-center gap-2">
                <Badge tone={status.chunks_indexed > 0 ? "ok" : "warn"}>
                  {status.chunks_indexed} chunks indexed
                </Badge>
              </div>
              <div className="text-xs text-slate-500">
                Chroma path:{" "}
                <span className="font-mono text-slate-300">{status.chroma_path}</span>
              </div>
              <Button size="sm" variant="ghost" onClick={refresh}>
                Refresh
              </Button>
            </>
          ) : (
            <Spinner />
          )}
        </Card>

        <Card className="space-y-4 p-5">
          <div className="text-sm font-semibold text-slate-100">Run ingestion</div>
          <Toggle
            checked={force}
            onChange={setForce}
            label="force=true"
            hint="Re-embed every chunk and upsert. Use after editing the knowledge file."
          />
          <Button onClick={ingest} disabled={busy}>
            {busy ? <Spinner /> : null}POST /rag/ingest{force ? "?force=true" : ""}
          </Button>
          <ErrorBox error={ingestErr} />
          {ingestRes && (
            <div className="text-sm text-slate-300">
              Indexed {ingestRes.chunks_indexed} chunks (
              <span className="font-mono">{ingestRes.embedding_dimensions}</span>-dim)
              from <span className="font-mono">{ingestRes.source}</span>.
            </div>
          )}
        </Card>
      </div>
    </Section>
  );
}

function SearchPanel() {
  const [query, setQuery] = useState(RAG_QUERIES[0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<LangGraphAgentResponse | null>(null);

  async function search() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      // Force the LLM to use only knowledge_search so we exercise the retriever.
      const r = await api.agentRun({
        goal: query,
        allowed_tools: ["knowledge_search"],
      });
      setResult(r);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  const ks = result?.steps.find((s) => s.tool_name === "knowledge_search");

  return (
    <Section
      title="Test semantic retrieval"
      subtitle="Runs /agent/run with allowed_tools=['knowledge_search'] so the LLM's only option is the RAG tool. The raw retrieved chunks (with cosine distance) appear below."
    >
      <Card className="space-y-4 p-5">
        <Field label="Natural-language query">
          <Textarea value={query} onChange={setQuery} rows={2} />
        </Field>
        <div className="flex flex-wrap gap-2">
          {RAG_QUERIES.map((q) => (
            <Button key={q} size="sm" variant="secondary" onClick={() => setQuery(q)}>
              {q}
            </Button>
          ))}
        </div>
        <Button onClick={search} disabled={busy || !query.trim()}>
          {busy ? <Spinner /> : null}Search
        </Button>
        <ErrorBox error={error} />
      </Card>

      {ks && (
        <Card className="mt-4 space-y-3 p-5">
          <div className="text-sm font-semibold text-slate-100">
            Retrieved chunks (knowledge_search output)
          </div>
          <pre className="code max-h-[28rem] overflow-auto rounded-lg border border-slate-800 bg-slate-950/80 p-3 whitespace-pre-wrap text-slate-200">
            {ks.tool_output}
          </pre>
          {result?.final_answer && (
            <div>
              <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Agent's final answer
              </div>
              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4 text-sm text-slate-100 whitespace-pre-wrap">
                {result.final_answer}
              </div>
            </div>
          )}
        </Card>
      )}

      {result && !ks && (
        <CodeBlock>{JSON.stringify(result, null, 2)}</CodeBlock>
      )}
    </Section>
  );
}
