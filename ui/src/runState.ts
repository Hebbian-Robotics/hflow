// The one owner of Airflow's run/task-state vocabulary in the UI: every chip,
// node fill and legend reads its tone from here so the graph and the tables can
// never drift apart. Unknown states render muted rather than failing (Airflow
// gains states across versions).

export type RunStateTone = "ok" | "err" | "warn" | "accent" | "muted";

export function runStateTone(state: string | null | undefined): RunStateTone {
  switch (state?.toLowerCase()) {
    case "success":
      return "ok";
    case "failed":
    case "error":
      return "err";
    case "running":
      return "accent";
    // A deferred task released its worker slot and is WAITING on a trigger —
    // it is healthy, so it reads like work in flight, never like a failure.
    case "deferred":
      return "accent";
    case "upstream_failed":
    case "up_for_retry":
    case "up_for_reschedule":
    case "restarting":
      return "warn";
    default:
      return "muted";
  }
}

export function runStateChipClass(state: string | null | undefined): string {
  return `chip chip-${runStateTone(state)}`;
}

/** States where the honest word is "waiting" rather than "stalled" or "stuck". */
export function isWaitingState(state: string | null | undefined): boolean {
  const normalized = state?.toLowerCase();
  return normalized === "deferred" || normalized === "queued" || normalized === "scheduled";
}
