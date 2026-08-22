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

/** 0.518 -> "51.8%"; integral percentages drop the decimal. */
export function formatFractionAsPercent(fraction: number): string {
  const percent = fraction * 100;
  const text = Number.isInteger(percent) ? String(percent) : percent.toFixed(1);
  return `${text}%`;
}

/** Display text for one value out of a SUMMARIZE row (open, JSON-safe record). */
export function summarizeValueText(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") {
    return looksLikeIsoTimestamp(value) ? formatTimestamp(value) : value;
  }
  return JSON.stringify(value);
}

export function shortFingerprint(fingerprint: string): string {
  return fingerprint.length > 10 ? fingerprint.slice(0, 10) : fingerprint;
}

/** Wall-clock span between two ISO timestamps, for run timing readouts. */
export function formatDurationBetween(startIso: string | null, endIso: string | null): string {
  if (!startIso || !endIso) return "—";
  const elapsedMs = Date.parse(endIso) - Date.parse(startIso);
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return "—";
  const totalSeconds = elapsedMs / 1000;
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)} s`;
  const wholeMinutes = Math.floor(totalSeconds / 60);
  const remainderSeconds = Math.round(totalSeconds % 60);
  if (wholeMinutes < 60) {
    return `${wholeMinutes}m ${String(remainderSeconds).padStart(2, "0")}s`;
  }
  const wholeHours = Math.floor(wholeMinutes / 60);
  return `${wholeHours}h ${String(wholeMinutes % 60).padStart(2, "0")}m`;
}

/** Tight duration readout for graph badges: "0.4s", "3.2s", "2m 04s", "1h 12m". */
export function formatDurationCompact(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const wholeMinutes = Math.floor(seconds / 60);
  if (wholeMinutes < 60) {
    return `${wholeMinutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  }
  const wholeHours = Math.floor(wholeMinutes / 60);
  return `${wholeHours}h ${String(wholeMinutes % 60).padStart(2, "0")}m`;
}

/** Content-derived React key for a raw catalog row (no array indexes). */
export function historyRowKey(row: EpisodeRow): string {
  return ["recorded_at", "run_fingerprint", "pipeline_version", "uri"]
    .map((field) => String(row[field] ?? ""))
    .join("|");
}
