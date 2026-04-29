import { ProposedToolCall } from "../types";
import { Badge, Card, JsonView, StatusIcon } from "./ui";

const WRITE_TOOLS = new Set(["task_manager"]);

export function ProposedCallCard({ call }: { call: ProposedToolCall }) {
  const isWrite = WRITE_TOOLS.has(call.name);
  return (
    <Card tone="warn" className="p-4">
      <div className="flex items-start gap-3">
        <span className="pulse-amber inline-flex h-9 w-9 flex-none items-center justify-center rounded-full bg-amber-500/20 text-amber-200">
          <StatusIcon status="paused" size={18} />
        </span>
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold text-slate-50">
              Awaiting approval
            </span>
            <Badge tone={isWrite ? "rw-write" : "rw-read"}>
              {isWrite ? "write" : "read"}
            </Badge>
            <Badge tone="violet">paused</Badge>
          </div>
          <div className="text-sm text-slate-300">
            About to call{" "}
            <span className="font-mono font-semibold text-amber-200">
              {call.name}
            </span>{" "}
            with:
          </div>
          <JsonView value={call.args} />
        </div>
      </div>
    </Card>
  );
}
