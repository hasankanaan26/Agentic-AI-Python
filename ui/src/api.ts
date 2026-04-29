import {
  AgentResponse,
  ApiError,
  IngestResponse,
  LangGraphAgentResponse,
  PendingThread,
  PermissionsResponse,
  RagStatus,
  SafeAgentRequest,
  SafetyCheckResult,
  TasksList,
  ToolCallResponse,
  ToolDefinition,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    throw new ApiError(res.status, `HTTP ${res.status} ${res.statusText}`, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Record<string, unknown>>("/health"),

  // /agent
  agentRun: (body: SafeAgentRequest) =>
    request<LangGraphAgentResponse>("/agent/run", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  agentRunRaw: (body: { goal: string; max_steps?: number }) =>
    request<AgentResponse>("/agent/run-raw", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  agentApprove: (thread_id: string, approved: boolean) =>
    request<LangGraphAgentResponse>("/agent/approve", {
      method: "POST",
      body: JSON.stringify({ thread_id, approved }),
    }),
  agentPending: () => request<{ pending: PendingThread[] }>("/agent/pending"),

  // /safety
  checkPrompt: (text: string) =>
    request<SafetyCheckResult>("/safety/check-prompt", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  permissions: () => request<PermissionsResponse>("/safety/permissions"),

  // /tools
  listTools: () => request<{ tools: ToolDefinition[] }>("/tools/list"),
  callTool: (message: string) =>
    request<ToolCallResponse>("/tools/call", {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  // /rag
  ragStatus: () => request<RagStatus>("/rag/status"),
  ragIngest: (force = false) =>
    request<IngestResponse>(`/rag/ingest${force ? "?force=true" : ""}`, {
      method: "POST",
    }),

  // /tasks
  listTasks: () => request<TasksList>("/tasks/list"),
};
