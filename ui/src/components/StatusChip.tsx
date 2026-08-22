type StatusTone = "ok" | "warn" | "err" | "muted";

// Unknown statuses render gracefully as muted rather than failing (forward compat).
function toneForStatus(status: string): StatusTone {
  switch (status.toLowerCase()) {
    case "ok":
    case "pass":
    case "passed":
    case "success":
      return "ok";
    case "quarantined":
    case "warn":
    case "warning":
      return "warn";
    case "fail":
    case "failed":
    case "error":
      return "err";
    default:
      return "muted";
  }
}

export function StatusChip({ status }: { status: string | null }) {
  // A null state is Airflow's "not scheduled yet", not an error: it reads as
  // the muted em dash the tables use for every other absent value.
  if (status === null) return <span className="cell-null">—</span>;
  return <span className={`chip chip-${toneForStatus(status)}`}>{status}</span>;
}
