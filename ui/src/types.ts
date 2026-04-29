// TypeScript mirrors of the FastAPI Pydantic models in
// checkpoints/checkpoint-3-safety-rag/app/models. Keep in sync if the
// backend schemas change.

export type ToolStatus = "ok" | "error";

export interface AgentStep {
  step: number;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_output: string;
  tool_status: ToolStatus;
}

export interface AgentResponse {
  goal: string;
  steps: AgentStep[];
  final_answer: string | null;
  steps_completed: number;
  model: string;
}

export type AgentRunStatus = "completed" | "paused" | "rejected" | "max_steps";

export interface ProposedToolCall {
  name: string;
  args: Record<string, unknown>;
  id: string | null;
}

export interface LangGraphAgentResponse extends AgentResponse {
  thread_id: string;
  engine: string;
  status: AgentRunStatus;
  proposed_tool_call: ProposedToolCall | null;
}

export interface PendingThread {
  thread_id: string;
  goal: string;
  allowed_tools: string[] | null;
}

export interface Task {
  id: number;
  title: string;
  done: boolean;
}

export interface TasksList {
  tasks: Task[];
  total: number;
  open: number;
  done: number;
}

export interface SafeAgentRequest {
  goal: string;
  allowed_tools?: string[] | null;
  require_approval?: boolean;
  max_steps?: number;
}

export interface InjectionFinding {
  pattern: string;
  description: string;
}

export type RiskLevel = "none" | "medium" | "high";

export interface SafetyCheckResult {
  flagged: boolean;
  findings: InjectionFinding[];
  risk_level: RiskLevel;
}

export interface PermissionsResponse {
  permissions: Record<string, "read" | "write">;
  description: { read: string; write: string };
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: string;
    properties: Record<string, { type?: string; description?: string; enum?: string[] }>;
    required?: string[];
  };
}

export interface ToolCallResponse {
  message: string;
  tool_called: boolean;
  tool_call: { name: string; arguments: Record<string, unknown> } | null;
  tool_result: string | null;
  tool_status: string | null;
  llm_response: string | null;
  model: string;
}

export interface IngestResponse {
  chunks_indexed: number;
  source: string;
  embedding_dimensions: number;
}

export interface RagStatus {
  chunks_indexed: number;
  chroma_path: string;
}

export interface InjectionDetail {
  error: string;
  risk_level: RiskLevel;
  findings: string[];
}

export type MessageType = "human" | "ai" | "tool" | "system" | "unknown";

export interface SnapshotMessage {
  type: MessageType;
  content: string;
  tool_calls?: { name: string; args: Record<string, unknown>; id: string | null }[];
  tool_call_id?: string;
  name?: string;
  status?: "error";
}

export interface ThreadSnapshot {
  checkpoint_id: string | null;
  parent_checkpoint_id: string | null;
  step: number | null;
  source: string | null;
  next: string[];
  writes: string[];
  created_at: string | null;
  messages: SnapshotMessage[];
  message_count: number;
}

export interface ThreadDetail {
  thread_id: string;
  current: ThreadSnapshot;
  history: ThreadSnapshot[];
  is_paused: boolean;
  pending_context: { goal: string; allowed_tools: string[] | null } | null;
}

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}
