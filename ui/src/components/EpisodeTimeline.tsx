import { useMemo } from "react";
import type { EpisodeTimeline, TimelineInterval, TimelineMeasurement } from "../api";
import { formatNumber } from "../format";

// Visual evidence for one episode: where in the recording something happened,
// and how big the numbers are. Everything drawn here comes from the server's
// timeline payload (spans, per-interval seconds, numeric measurements) — the
// UI only lays it out.

/** Fixed categorical slots. A kind past the third folds into the neutral
 * "other" slot rather than getting a generated hue, and the legend always
 * names it, so identity never rests on colour alone. */
const CATEGORY_SLOT_COUNT = 3;

const TICK_COUNT = 5;
/** Minimum visible width for a band, in percent of the span. */
const MIN_BAND_PERCENT = 0.6;
/** Band height plus the gap that keeps stacked lanes legible. */
const LANE_PITCH_PX = 22;

/**
 * An interval the server could place on the episode's clock.
 *
 * The served `start_s`/`end_s` are nullable: they are offsets from the
 * episode's own start, and the server sends null when it cannot derive that
 * start. An interval with no position cannot go on a time axis, so this type
 * marks the ones that can, and {@link placeableIntervals} separates them out
 * once rather than every consumer re-checking.
 */
interface PlacedInterval extends Omit<TimelineInterval, "start_s" | "end_s"> {
  start_s: number;
  end_s: number;
}

/** Splits intervals into those that can be drawn and a count of those that
 * cannot, so an unplaceable interval is reported rather than plotted at zero. */
function placeableIntervals(intervals: readonly TimelineInterval[]): {
  placed: PlacedInterval[];
  unplaceableCount: number;
} {
  const placed: PlacedInterval[] = [];
  let unplaceableCount = 0;
  for (const interval of intervals) {
    const { start_s, end_s } = interval;
    if (start_s === null || end_s === null) unplaceableCount += 1;
    else placed.push({ ...interval, start_s, end_s });
  }
  return { placed, unplaceableCount };
}

interface PackedInterval {
  interval: PlacedInterval;
  lane: number;
}

/** Greedy lane packing: overlapping intervals stack instead of hiding each
 * other, and a small gap keeps neighbouring bands visually separate. */
function packIntoLanes(intervals: readonly PlacedInterval[], spanSeconds: number) {
  const minimumGapSeconds = spanSeconds * 0.004;
  const ordered = [...intervals].sort(
    (left, right) => left.start_s - right.start_s || left.end_s - right.end_s,
  );
  const laneEnds: number[] = [];
  const packed: PackedInterval[] = [];
  for (const interval of ordered) {
    let laneIndex = laneEnds.findIndex((end) => end + minimumGapSeconds <= interval.start_s);
    if (laneIndex === -1) {
      laneIndex = laneEnds.length;
      laneEnds.push(interval.end_s);
    } else {
      laneEnds[laneIndex] = Math.max(laneEnds[laneIndex] ?? 0, interval.end_s);
    }
    packed.push({ interval, lane: laneIndex });
  }
  return { packed, laneCount: Math.max(1, laneEnds.length) };
}

/** Stable colour slots keyed by the kind itself (never by rank or count), so
 * a kind keeps its colour as episodes change. */
function colourSlotByKind(intervals: readonly TimelineInterval[]): Map<string, number> {
  const kinds = [...new Set(intervals.map((interval) => interval.kind))].sort();
  return new Map(kinds.map((kind, index) => [kind, index < CATEGORY_SLOT_COUNT ? index + 1 : 0]));
}

function secondsLabel(seconds: number): string {
  return `${formatNumber(Number(seconds.toFixed(seconds < 10 ? 2 : 1)))}s`;
}

export function TimelineStrip({ timeline }: { timeline: EpisodeTimeline }) {
  const { placed: intervals, unplaceableCount } = useMemo(
    () => placeableIntervals(timeline.intervals),
    [timeline.intervals],
  );
  // The server owns the span; when it could not derive one, the intervals
  // themselves are the only honest axis — and with neither, we say so.
  const derivedSpan = intervals.reduce((longest, interval) => Math.max(longest, interval.end_s), 0);
  const spanSeconds = timeline.duration_s ?? (derivedSpan > 0 ? derivedSpan : null);
  const isSpanFromIntervals = timeline.duration_s === null && spanSeconds !== null;

  const layout = useMemo(
    () => (spanSeconds === null ? null : packIntoLanes(intervals, spanSeconds)),
    [intervals, spanSeconds],
  );
  const slotByKind = useMemo(() => colourSlotByKind(intervals), [intervals]);

  if (spanSeconds === null || layout === null) {
    return (
      <p className="empty-note">
        No time span recorded for this episode — nothing in the catalog (no intervals, no duration
        measurement) says how long it is, so there is no axis to draw.
      </p>
    );
  }

  if (intervals.length === 0) {
    return (
      <p className="empty-note">
        The episode spans {secondsLabel(spanSeconds)}, but{" "}
        {unplaceableCount === 0
          ? "no check recorded an interval inside it."
          : `none of its ${unplaceableCount} recorded interval${
              unplaceableCount === 1 ? "" : "s"
            } could be placed on that axis: the catalog has no start time to measure them from.`}
      </p>
    );
  }

  const ticks = Array.from({ length: TICK_COUNT }, (_, index) => ({
    percent: (index / (TICK_COUNT - 1)) * 100,
    seconds: (index / (TICK_COUNT - 1)) * spanSeconds,
  }));
  const kindCounts = new Map<string, number>();
  for (const interval of intervals) {
    kindCounts.set(interval.kind, (kindCounts.get(interval.kind) ?? 0) + 1);
  }

  return (
    <div className="timeline">
      <div
        className="timeline-plot"
        style={{ height: `${layout.laneCount * LANE_PITCH_PX + 6}px` }}
      >
        {ticks.map((tick) => (
          <span
            key={tick.percent}
            className="timeline-gridline"
            style={{ left: `${tick.percent}%` }}
            aria-hidden="true"
          />
        ))}
        {layout.packed.map(({ interval, lane }) => {
          const startPercent = (interval.start_s / spanSeconds) * 100;
          const widthPercent = Math.max(
            MIN_BAND_PERCENT,
            ((interval.end_s - interval.start_s) / spanSeconds) * 100,
          );
          const slot = slotByKind.get(interval.kind) ?? 0;
          return (
            <span
              key={`${interval.label}:${interval.start_ns}:${interval.check_name}`}
              className={`timeline-band is-slot-${slot}`}
              style={{
                left: `${Math.min(startPercent, 100 - MIN_BAND_PERCENT)}%`,
                width: `${Math.min(widthPercent, 100 - startPercent)}%`,
                top: `${lane * LANE_PITCH_PX}px`,
              }}
              title={`${interval.label} · ${secondsLabel(interval.start_s)} → ${secondsLabel(
                interval.end_s,
              )} (${secondsLabel(interval.end_s - interval.start_s)}) · ${interval.check_name}`}
            >
              <span className="timeline-band-label">{interval.label}</span>
            </span>
          );
        })}
      </div>
      <div className="timeline-axis">
        {ticks.map((tick) => (
          <span key={tick.percent} className="timeline-tick" style={{ left: `${tick.percent}%` }}>
            {secondsLabel(tick.seconds)}
          </span>
        ))}
      </div>
      <div className="timeline-legend">
        {[...kindCounts.entries()]
          .sort((left, right) => left[0].localeCompare(right[0]))
          .map(([kind, count]) => (
            <span key={kind} className="timeline-legend-item">
              <span
                className={`timeline-swatch is-slot-${slotByKind.get(kind) ?? 0}`}
                aria-hidden="true"
              />
              <span className="timeline-legend-name">{kind}</span>
              <span className="timeline-legend-count">{count}</span>
            </span>
          ))}
        {isSpanFromIntervals ? (
          <span className="timeline-legend-note">
            axis spans the intervals themselves — the catalog records no episode duration
          </span>
        ) : null}
        {unplaceableCount > 0 ? (
          <span className="timeline-legend-note">
            {unplaceableCount} interval{unplaceableCount === 1 ? "" : "s"} not drawn — the catalog
            has no start time to measure {unplaceableCount === 1 ? "it" : "them"} from
          </span>
        ) : null}
      </div>
    </div>
  );
}

interface MeasurementGroup {
  unit: string | null;
  measurements: TimelineMeasurement[];
  largestMagnitude: number;
}

/** Bars are only comparable within one unit, so each unit gets its own scale
 * and its own heading — one shared axis across mixed units would lie. */
function groupByUnit(measurements: readonly TimelineMeasurement[]): MeasurementGroup[] {
  const byUnit = new Map<string, TimelineMeasurement[]>();
  for (const measurement of measurements) {
    const key = measurement.unit ?? "";
    const bucket = byUnit.get(key);
    if (bucket) bucket.push(measurement);
    else byUnit.set(key, [measurement]);
  }
  return [...byUnit.entries()]
    .map(([unit, entries]) => ({
      unit: unit === "" ? null : unit,
      measurements: [...entries].sort((left, right) => right.value - left.value),
      largestMagnitude: Math.max(...entries.map((entry) => Math.abs(entry.value)), 0),
    }))
    .sort((left, right) => right.measurements.length - left.measurements.length);
}

export function MeasurementBars({
  measurements,
}: {
  measurements: readonly TimelineMeasurement[];
}) {
  const groups = useMemo(() => groupByUnit(measurements), [measurements]);
  if (measurements.length === 0) {
    return (
      <p className="empty-note">
        No numeric measurements recorded for this episode — text and boolean measurements are in the
        table below.
      </p>
    );
  }
  return (
    <div className="measure-groups">
      {groups.map((group) => (
        <div key={group.unit ?? "unitless"} className="measure-group">
          <div className="measure-group-head">
            <span className="measure-group-unit">{group.unit ?? "unitless"}</span>
            <span className="measure-group-note">
              bars scale to the largest value in this unit ({formatNumber(group.largestMagnitude)})
            </span>
          </div>
          <ul className="measure-list">
            {group.measurements.map((measurement) => {
              const fraction =
                group.largestMagnitude === 0
                  ? 0
                  : Math.abs(measurement.value) / group.largestMagnitude;
              return (
                <li key={measurement.key} className="measure-row">
                  <span className="measure-key" title={measurement.key}>
                    {measurement.key}
                  </span>
                  <span className="measure-track">
                    <span
                      className={
                        measurement.value < 0 ? "measure-fill is-negative" : "measure-fill"
                      }
                      // Exactly zero draws nothing — a minimum-width sliver
                      // would read as "small but non-zero".
                      style={{
                        width: measurement.value === 0 ? "0%" : `${Math.max(fraction * 100, 1)}%`,
                      }}
                    />
                  </span>
                  <span className="measure-value">
                    {formatNumber(measurement.value)}
                    {measurement.unit ? (
                      <span className="measure-unit"> {measurement.unit}</span>
                    ) : null}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}
