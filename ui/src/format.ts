// Display formatting for values coming out of the catalog. Pure functions only.

import type { EpisodeMeasurement, EpisodeRow } from "./api";

const ISO_TIMESTAMP_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/;

export function looksLikeIsoTimestamp(text: string): boolean {
  return ISO_TIMESTAMP_PATTERN.test(text);
}

/** "2026-08-21T14:03:22.123456+00:00" -> "2026-08-21 14:03:22" (full value goes in title). */
export function formatTimestamp(isoText: string): string {
  return isoText.replace("T", " ").replace(/(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$/, "");
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (Number.isInteger(value)) return String(value);
  return String(Number.parseFloat(value.toPrecision(6)));
}

export function formatDurationSeconds(seconds: number | null): string {
  if (seconds === null) return "—";
  return `${seconds.toFixed(3)} s`;
}

export function nanosecondsToRelativeSeconds(valueNs: number, originNs: number): string {
  return ((valueNs - originNs) / 1e9).toFixed(3);
}

export function measurementDisplayValue(measurement: EpisodeMeasurement): string {
  if (measurement.value_double !== null && measurement.value_double !== undefined) {
    return formatNumber(measurement.value_double);
  }
  if (measurement.value_bool !== null && measurement.value_bool !== undefined) {
    return measurement.value_bool ? "true" : "false";
  }
  if (measurement.value_text !== null && measurement.value_text !== undefined) {
    return measurement.value_text;
  }
  return "—";
}

export function shortFingerprint(fingerprint: string): string {
  return fingerprint.length > 10 ? fingerprint.slice(0, 10) : fingerprint;
}

/** Content-derived React key for a raw catalog row (no array indexes). */
export function historyRowKey(row: EpisodeRow): string {
  return ["recorded_at", "run_fingerprint", "pipeline_version", "uri"]
    .map((field) => String(row[field] ?? ""))
    .join("|");
}
