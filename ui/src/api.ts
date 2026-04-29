import { recordThread } from "./storage";
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
  ThreadDetail,
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
  agentRun: async (body: SafeAgentRequest) => {
    const r = await request<LangGraphAgentResponse>("/agent/run", {
      method: "POST",
      body: JSON.stringify(body),
    });
    recordThread(r.thread_id);
    return r;
  },
  agentRunRaw: (body: { goal: string; max_steps?: number }) =>
    request<AgentResponse>("/agent/run-raw", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  agentApprove: async (thread_id: string, approved: boolean) => {
    const r = await request<LangGraphAgentResponse>("/agent/approve", {
      method: "POST",
      body: JSON.stringify({ thread_id, approved }),
    });
    recordThread(r.thread_id);
    return r;
  },
  agentPending: () => request<{ pending: PendingThread[] }>("/agent/pending"),
  agentThread: (thread_id: string) =>
    request<ThreadDetail>(`/agent/thread/${encodeURIComponent(thread_id)}`),

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
