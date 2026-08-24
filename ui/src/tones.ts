// The one owner of "what colour does this outcome read as" for the whole UI.
// Two separate vocabularies meet here and must not be confused for each other:
// Airflow's task/run states, and hflow's own recorded check statuses. Every
// node fill and chip reads its tone through one of these two functions, so the
// canvas and the inspector can never drift apart.

/**
 * Six tones, one per thing an outcome can be saying.
 *
 * They name MEANINGS, not colours: styles.css decides what each one looks
 * like. "run" is work in flight; "info" is work that finished and offered no
 * verdict to pass or fail.
 */
export type Tone = "ok" | "err" | "warn" | "run" | "info" | "muted";

/**
 * One Airflow task or run state.
 *
 * An unrecognized state renders muted rather than failing: Airflow gains
 * states across versions, and one this build has not heard of is not an error.
 */
export function airflowStateTone(state: string | null | undefined): Tone {
  switch (state?.toLowerCase()) {
    case "success":
      return "ok";
    case "failed":
      return "err";
    case "running":
    case "queued":
    case "scheduled":
      return "run";
    // A deferred task released its worker slot and is WAITING on a trigger. It
    // is healthy, so it reads like work in flight, never like a failure.
    case "deferred":
      return "run";
    case "upstream_failed":
    case "up_for_retry":
    case "up_for_reschedule":
    case "restarting":
      return "warn";
    // Muted, not a warning: a skipped stage is the NORMAL outcome of a gate
    // whose profile does not enable it (metadata_backfill skips three of the
    // four stages by design), so colouring it as trouble would cry wolf on
    // every backfill.
    case "skipped":
      return "muted";
    default:
      return "muted";
  }
}

/**
 * One recorded check status (``hflow.steps.CheckStatus``).
 *
 * "failed" and "error" share a tone deliberately. They are different facts -- a
 * decided False verdict about the DATA versus a crash in the check itself --
 * but both mean "this check did not pass", and the distinction is carried in
 * the node's own text rather than by inventing a colour for it.
 */
export function checkStatusTone(status: string | null | undefined): Tone {
  switch (status?.toLowerCase()) {
    case "passed":
      return "ok";
    case "failed":
    case "error":
      return "err";
    // Ran and recorded evidence, but offered no verdict: not a pass, not a
    // failure. Neither of those tones would be true.
    case "measured":
      return "info";
    case "skipped":
      return "muted";
    default:
      return "muted";
  }
}
