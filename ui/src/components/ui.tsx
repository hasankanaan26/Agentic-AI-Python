import { ReactNode, useEffect, useState } from "react";
import { CheckIcon, PauseIcon, XIcon } from "./icons";

export function Card(props: { children: ReactNode; className?: string; tone?: "default" | "accent" | "warn" }) {
  const tone = props.tone ?? "default";
  const tones: Record<string, string> = {
    default: "border-slate-700/70 bg-slate-900/70",
    accent: "border-sky-500/40 bg-sky-500/5",
    warn: "border-amber-500/40 bg-amber-500/5",
  };
  return (
    <div
      className={`rounded-xl border ${tones[tone]} shadow-lg shadow-slate-950/30 backdrop-blur-sm ${
        props.className ?? ""
      }`}
    >
      {props.children}
    </div>
  );
}

export function Section(props: {
  title: string;
  subtitle?: string;
  icon?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="space-y-3">
      <header className="flex items-start gap-3">
        {props.icon && (
          <span className="mt-0.5 inline-flex h-8 w-8 items-center justify-center rounded-lg bg-sky-500/15 text-sky-300">
            {props.icon}
          </span>
        )}
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-slate-50">
            {props.title}
          </h2>
          {props.subtitle && (
            <p className="mt-0.5 text-sm text-slate-400">{props.subtitle}</p>
          )}
        </div>
      </header>
      {props.children}
    </section>
  );
}

export function Button(props: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary" | "danger" | "ghost" | "approve" | "reject";
  size?: "sm" | "md";
  type?: "button" | "submit";
  title?: string;
}) {
  const v = props.variant ?? "primary";
  const s = props.size ?? "md";
  const base =
    "inline-flex items-center gap-2 rounded-lg font-medium transition-all disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-950";
  const sizing = s === "sm" ? "px-3 py-1.5 text-xs" : "px-4 py-2 text-sm";
  const variants: Record<string, string> = {
    primary:
      "bg-sky-500 text-slate-950 hover:bg-sky-400 focus:ring-sky-500/50 shadow-md shadow-sky-500/20",
    secondary:
      "bg-slate-800/80 text-slate-100 hover:bg-slate-700/80 border border-slate-700",
    danger:
      "bg-rose-500 text-slate-950 hover:bg-rose-400 focus:ring-rose-500/50 shadow-md shadow-rose-500/20",
    ghost: "bg-transparent text-slate-300 hover:bg-slate-800/60",
    approve:
      "bg-emerald-500 text-slate-950 hover:bg-emerald-400 focus:ring-emerald-500/50 shadow-md shadow-emerald-500/20",
    reject:
      "bg-rose-500/90 text-slate-950 hover:bg-rose-400 focus:ring-rose-500/50",
  };
  return (
    <button
      type={props.type ?? "button"}
      onClick={props.onClick}
      disabled={props.disabled}
      title={props.title}
      className={`${base} ${sizing} ${variants[v]}`}
    >
      {props.children}
    </button>
  );
}

export function Badge(props: {
  children: ReactNode;
  tone?:
    | "neutral"
    | "ok"
    | "warn"
    | "error"
    | "info"
    | "violet"
    | "rw-read"
    | "rw-write";
}) {
  const tone = props.tone ?? "neutral";
  const tones: Record<string, string> = {
    neutral: "bg-slate-800/80 text-slate-200 border-slate-700",
    ok: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
    warn: "bg-amber-500/15 text-amber-200 border-amber-500/40",
    error: "bg-rose-500/15 text-rose-300 border-rose-500/40",
    info: "bg-sky-500/15 text-sky-300 border-sky-500/40",
    violet: "bg-violet-500/15 text-violet-300 border-violet-500/40",
    "rw-read": "bg-emerald-500/10 text-emerald-300 border-emerald-500/40",
    "rw-write": "bg-amber-500/15 text-amber-200 border-amber-500/40",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {props.children}
    </span>
  );
}

export function StatusIcon({
  status,
  size = 14,
}: {
  status: "ok" | "error" | "paused";
  size?: number;
}) {
  if (status === "ok") return <CheckIcon size={size} className="text-emerald-300" />;
  if (status === "error") return <XIcon size={size} className="text-rose-300" />;
  return <PauseIcon size={size} className="text-amber-300" />;
}

export function Spinner({ size = 16 }: { size?: number }) {
  return (
    <span
      className="inline-block animate-spin rounded-full border-2 border-slate-600 border-t-sky-400"
      style={{ width: size, height: size }}
    />
  );
}

export function Toggle(props: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <label
      className={`flex items-start gap-3 rounded-lg border border-slate-700/70 bg-slate-900/60 p-3 transition-colors ${
        props.disabled
          ? "opacity-50"
          : "cursor-pointer hover:border-sky-500/40 hover:bg-slate-800/60"
      }`}
    >
      <input
        type="checkbox"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(e) => props.onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 rounded border-slate-600 bg-slate-900 text-sky-500"
      />
      <div className="text-sm">
        <div className="font-medium text-slate-100">{props.label}</div>
        {props.hint && <div className="mt-0.5 text-xs text-slate-400">{props.hint}</div>}
      </div>
    </label>
  );
}

export function CodeBlock(props: { children: string; className?: string }) {
  return (
    <pre
      className={`code overflow-x-auto rounded-lg border border-slate-700/70 bg-slate-950/80 p-3 text-slate-200 ${
        props.className ?? ""
      }`}
    >
      {props.children}
    </pre>
  );
}

export function JsonView({ value }: { value: unknown }) {
  let text: string;
  try {
    text = JSON.stringify(value, null, 2);
  } catch {
    text = String(value);
  }
  return <CodeBlock>{text}</CodeBlock>;
}

export function Callout(props: {
  tone?: "info" | "warn" | "danger" | "success";
  title: string;
  children: ReactNode;
}) {
  const tone = props.tone ?? "info";
  const tones: Record<string, string> = {
    info: "border-sky-500/40 bg-sky-500/5",
    warn: "border-amber-500/40 bg-amber-500/5",
    danger: "border-rose-500/40 bg-rose-500/5",
    success: "border-emerald-500/40 bg-emerald-500/5",
  };
  return (
    <div className={`rounded-lg border p-4 ${tones[tone]}`}>
      <div className="text-sm font-semibold text-slate-100">{props.title}</div>
      <div className="mt-1.5 text-sm text-slate-300">{props.children}</div>
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const e = error as { message?: string; status?: number; detail?: unknown };
  return (
    <Callout tone="danger" title={`Request failed${e.status ? ` (HTTP ${e.status})` : ""}`}>
      <div className="space-y-2">
        <div>{e.message ?? String(error)}</div>
        {e.detail != null && <JsonView value={e.detail} />}
      </div>
    </Callout>
  );
}

export function Field(props: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">
        {props.label}
      </label>
      {props.children}
      {props.hint && <div className="text-xs text-slate-500">{props.hint}</div>}
    </div>
  );
}

export function Textarea(props: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
  disabled?: boolean;
}) {
  return (
    <textarea
      value={props.value}
      onChange={(e) => props.onChange(e.target.value)}
      placeholder={props.placeholder}
      rows={props.rows ?? 3}
      disabled={props.disabled}
      className="w-full resize-y rounded-lg border border-slate-700/70 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-500/60 focus:outline-none focus:ring-2 focus:ring-sky-500/30"
    />
  );
}

export function Input(props: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
  type?: "text" | "number";
}) {
  return (
    <input
      type={props.type ?? "text"}
      value={props.value}
      onChange={(e) => props.onChange(e.target.value)}
      placeholder={props.placeholder}
      disabled={props.disabled}
      className="w-full rounded-lg border border-slate-700/70 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-500/60 focus:outline-none focus:ring-2 focus:ring-sky-500/30"
    />
  );
}

export function StatPill(props: { label: string; value: ReactNode; tone?: "info" | "ok" | "warn" }) {
  const tone = props.tone ?? "info";
  const tones: Record<string, string> = {
    info: "from-sky-500/15 to-sky-500/5 border-sky-500/40 text-sky-100",
    ok: "from-emerald-500/15 to-emerald-500/5 border-emerald-500/40 text-emerald-100",
    warn: "from-amber-500/15 to-amber-500/5 border-amber-500/40 text-amber-100",
  };
  return (
    <div
      className={`flex flex-col rounded-lg border bg-gradient-to-br px-4 py-2.5 ${tones[tone]}`}
    >
      <span className="text-[10px] font-semibold uppercase tracking-wide opacity-80">
        {props.label}
      </span>
      <span className="mt-0.5 text-lg font-semibold">{props.value}</span>
    </div>
  );
}

export function useStaggeredReveal(count: number, stepMs = 180): number {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    setShown(0);
    if (count === 0) return;
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      setShown((s) => Math.min(s + 1, count));
      if (i >= count) clearInterval(id);
    }, stepMs);
    return () => clearInterval(id);
  }, [count, stepMs]);
  return shown;
}
