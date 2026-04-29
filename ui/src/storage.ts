// Lightweight localStorage helper for the "recent threads" picker.

const KEY = "cp3.recent_threads";
const MAX = 20;

export function recordThread(thread_id: string): void {
  if (!thread_id) return;
  try {
    const ids = readThreads().filter((t) => t !== thread_id);
    ids.unshift(thread_id);
    localStorage.setItem(KEY, JSON.stringify(ids.slice(0, MAX)));
    // Tell other tabs/components.
    window.dispatchEvent(new Event("cp3-threads-changed"));
  } catch {
    // localStorage may be disabled (private mode); silent fail is fine.
  }
}

export function readThreads(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

export function clearThreads(): void {
  localStorage.removeItem(KEY);
  window.dispatchEvent(new Event("cp3-threads-changed"));
}
