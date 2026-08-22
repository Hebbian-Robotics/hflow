import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { formatDurationCompact, formatOffsetCompact } from "../format";
import {
  ChevronDownIcon,
  InfoIcon,
  PauseIcon,
  PlayIcon,
  SkipEndIcon,
  SkipStartIcon,
} from "../icons";
import { runStateTone } from "../runState";
import {
  axisTickOffsetsSeconds,
  formatInstant,
  formatOffsetLabel,
  instanceSpans,
  MAPPED_COLLAPSE_THRESHOLD,
  MASTER_SCOPE,
  type ReplayBatchRow,
  type ReplayGroup,
  type ReplayRow,
  type ReplaySpan,
  type RunNodeSelection,
  type RunReplay,
  replayedRunState,
  type TimedInstance,
} from "../runTimeline";
import { usePrefersReducedMotion } from "../usePrefersReducedMotion";
import { RunStateChip } from "./RunStateChip";

// The run replay: a Gantt of every task instance on one shared axis, with a
// draggable playhead that re-paints the DAG above at the instant it names.
//
// Everything drawn here is derived from the timestamps the run-graph payload
// already carries (see runTimeline.ts). The reconstruction's limits are stated
// in the panel itself — a replay assembled from start/end stamps is not an
// event log, and this UI never pretends otherwise.

/** Playback stretches (or compresses) the run into roughly this long. */
const PLAYBACK_FIT_SECONDS = 12;

const SPEED_CHOICES: readonly { id: string; label: string }[] = [
  { id: "fit", label: `fit ≈ ${PLAYBACK_FIT_SECONDS}s` },
  { id: "1", label: "1×" },
  { id: "5", label: "5×" },
  { id: "25", label: "25×" },
  { id: "100", label: "100×" },
];

/** An arrow key steps this fraction of the run; Shift multiplies it by ten. */
const KEY_STEP_FRACTION = 1 / 100;

/** So a zero-length span (a skip with no end) is still a visible mark. */
const MIN_BAR_PERCENT = 0.4;

/** In a collapsed fan-out every batch shares one lane, so the states that
 * matter are painted last rather than hidden under the successes. */
const FAN_OUT_PAINT_RANK: Record<string, number> = {
  success: 0,
  removed: 1,
  skipped: 1,
  upstream_failed: 2,
  up_for_retry: 3,
  up_for_reschedule: 3,
  queued: 4,
  scheduled: 4,
  deferred: 5,
  running: 6,
  failed: 7,
};

/** The pointer contract every drag surface (ruler, row track, grip) shares. */
interface ScrubHandlers {
  onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void;
  onPointerMove: (event: ReactPointerEvent<HTMLElement>) => void;
  onPointerUp: (event: ReactPointerEvent<HTMLElement>) => void;
  onPointerCancel: (event: ReactPointerEvent<HTMLElement>) => void;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

interface AxisGeometry {
  startMs: number;
  spanMs: number;
  endMs: number;
  offsetMinutes: number;
}

function percentOf(instantMs: number, axis: AxisGeometry): number {
  return clamp(((instantMs - axis.startMs) / axis.spanMs) * 100, 0, 100);
}

function spanTitle(span: ReplaySpan, label: string, axis: AxisGeometry): string {
  const fromSeconds = (span.fromMs - axis.startMs) / 1000;
  const toSeconds = (span.toMs - axis.startMs) / 1000;
  const offsetLabel = formatOffsetLabel(axis.offsetMinutes);
  const what = span.kind === "queued" ? "queued, waiting for a worker" : "running";
  const openNote = span.open ? " · still open (no end recorded yet)" : "";
  return (
    `${label} — ${what}\n` +
    `+${formatOffsetCompact(fromSeconds)} → +${formatOffsetCompact(toSeconds)} ` +
    `(${formatDurationCompact((span.toMs - span.fromMs) / 1000)})\n` +
    `${formatInstant(span.fromMs, axis.offsetMinutes)} → ` +
    `${formatInstant(span.toMs, axis.offsetMinutes)} ${offsetLabel}${openNote}`
  );
}

function SpanBar({
  span,
  timed,
  label,
  axis,
}: {
  span: ReplaySpan;
  timed: TimedInstance;
  label: string;
  axis: AxisGeometry;
}) {
  const left = percentOf(span.fromMs, axis);
  const rawWidth = ((span.toMs - span.fromMs) / axis.spanMs) * 100;
  const width = Math.max(MIN_BAR_PERCENT, Math.min(rawWidth, 100 - left));
  const classes = [
    "replay-bar",
    span.kind === "queued" ? "is-queued" : `is-${runStateTone(timed.instance.state)}`,
    span.open ? "is-open" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <span
      className={classes}
      style={{ left: `${left}%`, width: `${width}%` }}
      title={spanTitle(span, label, axis)}
    />
  );
}

/** Every bar one instance contributes, in one row's track. */
function InstanceBars({
  timed,
  label,
  axis,
}: {
  timed: TimedInstance;
  label: string;
  axis: AxisGeometry;
}) {
  const spans = instanceSpans(timed, axis.endMs);
  return (
    <>
      {spans.map((span) => (
        <SpanBar
          key={`${span.kind}:${span.fromMs}:${span.toMs}`}
          span={span}
          timed={timed}
          label={label}
          axis={axis}
        />
      ))}
    </>
  );
}

function fanOutPaintRank(timed: TimedInstance): number {
  const state = timed.instance.state?.toLowerCase();
  return state === undefined ? 4 : (FAN_OUT_PAINT_RANK[state] ?? 4);
}

function rowDurationText(timed: readonly TimedInstance[]): string {
  if (timed.length === 1) {
    const only = timed[0] as TimedInstance;
    if (only.open && only.startedAtMs !== null) return "open";
    return formatDurationCompact(only.instance.duration_s);
  }
  const total = timed.reduce((sum, entry) => sum + (entry.instance.duration_s ?? 0), 0);
  return total > 0 ? formatDurationCompact(total) : "—";
}

// ---- the panel -------------------------------------------------------------------

export interface RunTimelineProps {
  replay: RunReplay;
  /** The instant the graph above is painted at. */
  playheadMs: number;
  /** null re-pins the playhead to the axis end (which is "now" while live). */
  onPlayheadChange: (playheadMs: number | null) => void;
  /** True while the playhead is pinned to the end rather than parked. */
  isPinnedToEnd: boolean;
  selection: RunNodeSelection | null;
  onSelect: (selection: RunNodeSelection) => void;
  /** The page's 10s poll — live mode is only fed while it is on. */
  isAutoRefreshOn: boolean;
}

export function RunTimeline({
  replay,
  playheadMs,
  onPlayheadChange,
  isPinnedToEnd,
  selection,
  onSelect,
  isAutoRefreshOn,
}: RunTimelineProps) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedChoice, setSpeedChoice] = useState<string>("fit");
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [toggledFanOutKeys, setToggledFanOutKeys] = useState<ReadonlySet<string>>(new Set());
  const trackRef = useRef<HTMLDivElement | null>(null);

  const axis: AxisGeometry = useMemo(
    () => ({
      startMs: replay.startMs,
      spanMs: replay.spanMs,
      endMs: replay.endMs,
      offsetMinutes: replay.offsetMinutes,
    }),
    [replay.startMs, replay.spanMs, replay.endMs, replay.offsetMinutes],
  );

  const spanSeconds = replay.spanMs / 1000;
  const offsetSeconds = (playheadMs - replay.startMs) / 1000;
  const playheadPercent = percentOf(playheadMs, axis);
  const isLive = replay.unfinished && isPinnedToEnd;
  const speedMultiplier =
    speedChoice === "fit"
      ? Math.max(1, replay.spanMs / (PLAYBACK_FIT_SECONDS * 1000))
      : Number.parseFloat(speedChoice);

  // The playback loop reads these through refs so a frame never re-subscribes
  // the effect (and so live mode's one-second axis growth cannot restart it).
  const playheadRef = useRef(playheadMs);
  playheadRef.current = playheadMs;
  const speedRef = useRef(speedMultiplier);
  speedRef.current = speedMultiplier;
  const onPlayheadChangeRef = useRef(onPlayheadChange);
  onPlayheadChangeRef.current = onPlayheadChange;

  const endMs = replay.endMs;
  useEffect(() => {
    if (!isPlaying) return;
    let frame = 0;
    let previousTimestamp = performance.now();
    const step = (timestamp: number) => {
      const advancedMs = (timestamp - previousTimestamp) * speedRef.current;
      previousTimestamp = timestamp;
      const next = playheadRef.current + advancedMs;
      if (next >= endMs) {
        // Running off the end re-pins to the end, which for an unfinished run
        // is live: playback ends by handing the reader back to the present.
        onPlayheadChangeRef.current(null);
        setIsPlaying(false);
        return;
      }
      onPlayheadChangeRef.current(next);
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [isPlaying, endMs]);

  // A reader who asked for reduced motion never gets an animation, so a
  // playback session cannot survive the preference flipping mid-run either.
  useEffect(() => {
    if (prefersReducedMotion) setIsPlaying(false);
  }, [prefersReducedMotion]);

  const seekToMs = (instantMs: number) => {
    setIsPlaying(false);
    const clamped = clamp(instantMs, replay.startMs, replay.endMs);
    // Landing exactly on the end re-pins (and re-attaches live) rather than
    // parking a millisecond behind the present.
    onPlayheadChange(clamped >= replay.endMs ? null : clamped);
  };

  const seekToClientX = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const fraction = clamp((clientX - rect.left) / rect.width, 0, 1);
    seekToMs(replay.startMs + fraction * replay.spanMs);
  };

  // A ref, not the state, decides whether a move is a drag: the first
  // pointermove can arrive before React has re-rendered with the new state.
  const isScrubbingRef = useRef(false);
  const beginScrub = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    isScrubbingRef.current = true;
    setIsScrubbing(true);
    seekToClientX(event.clientX);
  };
  const continueScrub = (event: ReactPointerEvent<HTMLElement>) => {
    if (!isScrubbingRef.current) return;
    seekToClientX(event.clientX);
  };
  const endScrub = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    isScrubbingRef.current = false;
    setIsScrubbing(false);
  };
  const scrubHandlers: ScrubHandlers = {
    onPointerDown: beginScrub,
    onPointerMove: continueScrub,
    onPointerUp: endScrub,
    onPointerCancel: endScrub,
  };

  const stepBy = (direction: -1 | 1, isCoarse: boolean) => {
    const stepMs = replay.spanMs * KEY_STEP_FRACTION * (isCoarse ? 10 : 1);
    seekToMs(playheadMs + direction * stepMs);
  };

  const togglePlay = () => {
    if (prefersReducedMotion) return;
    setIsPlaying((playing) => {
      if (playing) return false;
      // Playing from the very end would finish instantly; rewind first.
      if (playheadMs >= replay.endMs) onPlayheadChange(replay.startMs);
      return true;
    });
  };

  const handleSliderKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    switch (event.key) {
      case "ArrowLeft":
      case "ArrowDown":
        event.preventDefault();
        stepBy(-1, event.shiftKey);
        return;
      case "ArrowRight":
      case "ArrowUp":
        event.preventDefault();
        stepBy(1, event.shiftKey);
        return;
      case "Home":
        event.preventDefault();
        seekToMs(replay.startMs);
        return;
      case "End":
        event.preventDefault();
        setIsPlaying(false);
        onPlayheadChange(null);
        return;
      case " ":
      case "Spacebar":
        event.preventDefault();
        togglePlay();
        return;
      default:
    }
  };

  const tickOffsets = useMemo(() => axisTickOffsetsSeconds(spanSeconds), [spanSeconds]);
  const isFanOutExpanded = (row: ReplayRow): boolean => {
    const expandedByDefault = row.batches.length <= MAPPED_COLLAPSE_THRESHOLD;
    return toggledFanOutKeys.has(row.key) ? !expandedByDefault : expandedByDefault;
  };
  const toggleFanOut = (rowKey: string) => {
    setToggledFanOutKeys((previous) => {
      const next = new Set(previous);
      if (next.has(rowKey)) next.delete(rowKey);
      else next.add(rowKey);
      return next;
    });
  };

  const absoluteReadout = `${formatInstant(playheadMs, replay.offsetMinutes)} ${formatOffsetLabel(
    replay.offsetMinutes,
  )}`;

  return (
    <figure className={isScrubbing ? "replay is-scrubbing" : "replay"} aria-label="Run replay">
      <div className="replay-head">
        <span className="replay-title">Replay</span>
        <div className="replay-transport">
          <button
            type="button"
            className="btn btn-tiny"
            onClick={() => seekToMs(replay.startMs)}
            disabled={!replay.hasTimedInstances}
            title="Jump to the run's start (Home)"
          >
            <SkipStartIcon />
          </button>
          <button
            type="button"
            className={isPlaying ? "btn btn-tiny is-playing" : "btn btn-tiny"}
            onClick={togglePlay}
            disabled={prefersReducedMotion || !replay.hasTimedInstances}
            title={
              prefersReducedMotion
                ? "Playback is off because this system asks for reduced motion — scrub or step with the arrow keys instead."
                : isPlaying
                  ? "Pause (Space)"
                  : "Play the run back (Space)"
            }
          >
            {isPlaying ? <PauseIcon /> : <PlayIcon />}
            <span>{isPlaying ? "Pause" : "Play"}</span>
          </button>
          <button
            type="button"
            className="btn btn-tiny"
            onClick={() => {
              setIsPlaying(false);
              onPlayheadChange(null);
            }}
            disabled={!replay.hasTimedInstances}
            title="Jump to the end of the axis (End)"
          >
            <SkipEndIcon />
          </button>
          <label className="replay-speed">
            <span className="replay-speed-label">speed</span>
            <select
              className="input"
              value={speedChoice}
              onChange={(event) => setSpeedChoice(event.target.value)}
              disabled={prefersReducedMotion}
              title={`Playback speed. "fit" stretches or compresses the whole run into about ${PLAYBACK_FIT_SECONDS} seconds.`}
            >
              {SPEED_CHOICES.map((choice) => (
                <option key={choice.id} value={choice.id}>
                  {choice.label}
                </option>
              ))}
            </select>
          </label>
          {prefersReducedMotion ? (
            <span
              className="replay-reduced-note"
              title="Nothing on this page animates itself while your system asks for reduced motion; the playhead still moves when you drag it or step it with the arrow keys."
            >
              auto-play off · reduced motion
            </span>
          ) : null}
        </div>
        <span className="toolbar-spacer" />
        {replay.hasTimedInstances ? (
          <span className="replay-readout">
            <span className="replay-readout-offset">+{formatOffsetCompact(offsetSeconds)}</span>
            <span className="replay-readout-absolute">{absoluteReadout}</span>
          </span>
        ) : null}
        {replay.unfinished ? (
          <button
            type="button"
            className={isLive ? "replay-live is-live" : "replay-live"}
            onClick={() => {
              setIsPlaying(false);
              onPlayheadChange(null);
            }}
            aria-pressed={isLive}
            title={
              isLive
                ? "The playhead is pinned to now; scrubbing back detaches it."
                : "Re-attach the playhead to now."
            }
          >
            <span className="replay-live-dot" aria-hidden="true" />
            <span>{isLive ? "Live" : "Go live"}</span>
          </button>
        ) : null}
      </div>

      {replay.unfinished && isLive && !isAutoRefreshOn ? (
        <p className="replay-note">
          Live, but auto-refresh is off: the playhead follows the clock while the task data stays as
          last fetched. Turn on auto-refresh above (or press Refresh) to keep it fed.
        </p>
      ) : null}

      {!replay.hasTimedInstances ? (
        <p className="empty-note replay-empty">
          No task instance of this run carries a timestamp yet, so there is nothing to place on an
          axis. The replay appears as soon as the first task is queued or starts.
        </p>
      ) : (
        <div className="replay-chart">
          <div className="replay-scroll">
            <div className="replay-lanes">
              <div className="replay-ruler">
                <span className="replay-ruler-gutter">from run start</span>
                {/* The ruler is the playhead's drag surface; the focusable
                    slider below carries the keyboard contract for it. */}
                <div
                  className="replay-ruler-track"
                  ref={trackRef}
                  {...scrubHandlers}
                  title="Drag to move the playhead"
                >
                  {tickOffsets.map((offset) => (
                    <span
                      key={offset}
                      className="replay-tick"
                      style={{ left: `${(offset / Math.max(spanSeconds, 1e-9)) * 100}%` }}
                      title={`${formatInstant(
                        replay.startMs + offset * 1000,
                        replay.offsetMinutes,
                      )} ${formatOffsetLabel(replay.offsetMinutes)}`}
                    >
                      {formatOffsetCompact(offset)}
                    </span>
                  ))}
                </div>
              </div>

              {replay.groups.map((group) => (
                <ReplayGroupRows
                  key={group.scope}
                  group={group}
                  axis={axis}
                  playheadMs={playheadMs}
                  selection={selection}
                  onSelect={onSelect}
                  isFanOutExpanded={isFanOutExpanded}
                  onToggleFanOut={toggleFanOut}
                  scrubHandlers={scrubHandlers}
                />
              ))}

              <div className="replay-overlay">
                {tickOffsets.map((offset) => (
                  <span
                    key={offset}
                    className="replay-gridline"
                    style={{ left: `${(offset / Math.max(spanSeconds, 1e-9)) * 100}%` }}
                    aria-hidden="true"
                  />
                ))}
                <div className="replay-playhead" style={{ left: `${playheadPercent}%` }}>
                  {/* A slider over a time axis has no native element:
                      role=slider plus the arrow/Home/End contract below is the
                      accessible equivalent, and it reports both readouts. */}
                  <div
                    className="replay-grip"
                    role="slider"
                    tabIndex={0}
                    aria-label="Run replay playhead"
                    aria-valuemin={0}
                    aria-valuemax={Number(spanSeconds.toFixed(3))}
                    aria-valuenow={Number(offsetSeconds.toFixed(3))}
                    aria-valuetext={`+${formatOffsetCompact(offsetSeconds)}, ${absoluteReadout}`}
                    onKeyDown={handleSliderKeyDown}
                    {...scrubHandlers}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <ReplayLimits replay={replay} />

      <figcaption className="dag-hint replay-hint">
        Drag the axis (or any row) to move the playhead; the graphs take the state each task was in
        at that instant. Click a task name for its details. With the playhead focused, Left and
        Right step it (hold Shift for a bigger step), Home and End jump to the ends, and Space
        plays. Hatched lead-ins are queued time; a bar that fades out at the edge has no end
        recorded yet.
      </figcaption>
    </figure>
  );
}

function ReplayGroupRows({
  group,
  axis,
  playheadMs,
  selection,
  onSelect,
  isFanOutExpanded,
  onToggleFanOut,
  scrubHandlers,
}: {
  group: ReplayGroup;
  axis: AxisGeometry;
  playheadMs: number;
  selection: RunNodeSelection | null;
  onSelect: (selection: RunNodeSelection) => void;
  isFanOutExpanded: (row: ReplayRow) => boolean;
  onToggleFanOut: (rowKey: string) => void;
  scrubHandlers: ScrubHandlers;
}) {
  const replayedState = replayedRunState(group.instances, group.state, playheadMs, axis.endMs);
  return (
    <div className="replay-group">
      <div className="replay-group-head">
        <span className="replay-group-title">
          {group.scope === MASTER_SCOPE ? "master DAG" : `${group.title} sub-DAG`}
        </span>
        <RunStateChip state={replayedState ?? "not run"} />
        {group.dagId ? (
          <code className="cell-mono replay-group-dag" title={group.dagId}>
            {group.dagId}
          </code>
        ) : null}
      </div>
      {group.note ? <p className="replay-group-note">{group.note}</p> : null}
      {group.rows.map((row) => (
        <ReplayRowView
          key={row.key}
          row={row}
          axis={axis}
          isSelected={selection?.scope === row.scope && selection.taskId === row.taskId}
          onSelect={onSelect}
          isExpanded={isFanOutExpanded(row)}
          onToggleFanOut={onToggleFanOut}
          scrubHandlers={scrubHandlers}
        />
      ))}
    </div>
  );
}

function ReplayRowView({
  row,
  axis,
  isSelected,
  onSelect,
  isExpanded,
  onToggleFanOut,
  scrubHandlers,
}: {
  row: ReplayRow;
  axis: AxisGeometry;
  isSelected: boolean;
  onSelect: (selection: RunNodeSelection) => void;
  isExpanded: boolean;
  onToggleFanOut: (rowKey: string) => void;
  scrubHandlers: ScrubHandlers;
}) {
  const isFanOut = row.kind === "fanout";
  const paintOrdered = useMemo(
    () => (isFanOut ? [...row.timed].sort((a, b) => fanOutPaintRank(a) - fanOutPaintRank(b)) : []),
    [isFanOut, row.timed],
  );
  const batchesShown = isFanOut && isExpanded;
  return (
    <>
      <div className={isSelected ? "replay-row is-selected" : "replay-row"}>
        <div className="replay-row-label">
          <button
            type="button"
            className="replay-row-name"
            onClick={() => onSelect({ scope: row.scope, taskId: row.taskId })}
            aria-current={isSelected ? "true" : undefined}
            title={`${row.taskId} — ${row.summary}`}
          >
            {row.taskId}
          </button>
          {row.deferred ? (
            <span
              className="replay-tag is-waits"
              title="This task defers its worker slot: most of its span was spent waiting on a trigger, not running. The payload cannot separate the two."
            >
              waits
            </span>
          ) : null}
          {row.retriedTryNumber !== null ? (
            <span
              className="replay-tag is-retry"
              title={`Attempt ${row.retriedTryNumber}. Only the latest attempt is in this payload — earlier attempts are not shown, so the span below is that last try alone.`}
            >
              try {row.retriedTryNumber}
            </span>
          ) : null}
          {isFanOut ? (
            <button
              type="button"
              className="replay-fanout-toggle"
              onClick={() => onToggleFanOut(row.key)}
              aria-expanded={isExpanded}
              title={
                isExpanded
                  ? `Collapse these ${row.batches.length} batches into one summary row`
                  : `Show all ${row.batches.length} batches (fan-outs over ${MAPPED_COLLAPSE_THRESHOLD} collapse by default)`
              }
            >
              <ChevronDownIcon className={isExpanded ? "chevron is-open" : "chevron"} />
              <span>×{row.batches.length}</span>
            </button>
          ) : (
            <span className="replay-row-duration">{rowDurationText(row.timed)}</span>
          )}
        </div>
        {/* A track is a drag surface for the playhead; selection lives on the
            row's button, and the playhead itself is the focusable control. */}
        <div
          className={isFanOut ? "replay-row-track is-summary" : "replay-row-track"}
          {...scrubHandlers}
        >
          {row.timed.every((timed) => timed.untimed) ? (
            <span
              className="replay-untimed"
              title="Airflow reported this instance with no queue time and no start time (a skip it never scheduled), so the replay cannot place it on the axis: it stays unstarted and only takes its reported state at the axis end."
            >
              no timestamps — cannot be placed on the axis
            </span>
          ) : (
            (isFanOut ? paintOrdered : row.timed).map((timed) => (
              <InstanceBars
                key={`${timed.instance.map_index}:${timed.instance.start_date ?? "unstarted"}`}
                timed={timed}
                label={isFanOut ? `${row.taskId}[${timed.instance.map_index}]` : row.taskId}
                axis={axis}
              />
            ))
          )}
        </div>
      </div>
      {batchesShown
        ? row.batches.map((batch) => (
            <ReplayBatchRowView
              key={batch.key}
              row={row}
              batch={batch}
              axis={axis}
              isSelected={isSelected}
              onSelect={onSelect}
              scrubHandlers={scrubHandlers}
            />
          ))
        : null}
    </>
  );
}

function ReplayBatchRowView({
  row,
  batch,
  axis,
  isSelected,
  onSelect,
  scrubHandlers,
}: {
  row: ReplayRow;
  batch: ReplayBatchRow;
  axis: AxisGeometry;
  isSelected: boolean;
  onSelect: (selection: RunNodeSelection) => void;
  scrubHandlers: ScrubHandlers;
}) {
  const tryNumber = batch.timed.instance.try_number;
  return (
    <div className={isSelected ? "replay-row is-batch is-selected" : "replay-row is-batch"}>
      <div className="replay-row-label">
        <button
          type="button"
          className="replay-row-name is-batch"
          onClick={() => onSelect({ scope: row.scope, taskId: row.taskId })}
          title={`${row.taskId}[${batch.mapIndex}] — one mapped batch of this task`}
        >
          [{batch.mapIndex}]
        </button>
        {tryNumber !== null && tryNumber > 1 ? (
          <span
            className="replay-tag is-retry"
            title={`Attempt ${tryNumber}. Earlier attempts of this batch are not in the payload.`}
          >
            try {tryNumber}
          </span>
        ) : null}
        <span className="replay-row-duration">{rowDurationText([batch.timed])}</span>
      </div>
      {/* A drag surface, like the task row's track above. */}
      <div className="replay-row-track" {...scrubHandlers}>
        <InstanceBars timed={batch.timed} label={`${row.taskId}[${batch.mapIndex}]`} axis={axis} />
      </div>
    </div>
  );
}

/** What this reconstruction can and cannot say, stated where it is drawn. */
function ReplayLimits({ replay }: { replay: RunReplay }) {
  return (
    <details className="replay-limits">
      <summary className="replay-limits-summary">
        <InfoIcon />
        <span>
          Reconstructed from each task's start and end timestamps — not an event log.
          {replay.retriedTaskCount === 0
            ? null
            : replay.retriedTaskCount === 1
              ? " One task here shows only its latest attempt."
              : ` ${replay.retriedTaskCount} tasks here show only their latest attempt.`}
        </span>
      </summary>
      <ul className="replay-limits-list">
        <li>
          <strong>Latest attempt only.</strong> The payload carries one row per task instance, so a
          retried task shows its final try. Earlier attempts — and the gaps between them — are not
          in this data and are not drawn.
        </li>
        <li>
          <strong>No progress inside a task.</strong> A bar is all-or-nothing between its start and
          its end; nothing here knows how far through its work a task was.
        </li>
        <li>
          <strong>Deferred tasks look busy.</strong> A <code>trigger_*</code> task releases its
          worker and waits on a trigger, but the payload timestamps only its whole span, so it is
          one continuous bar.
        </li>
        <li>
          <strong>Queued time needs the stamp.</strong> The hatched lead-in is drawn only where
          Airflow reported <code>queued_at</code>; without it a task appears to start the instant it
          was scheduled.
        </li>
        {replay.untimedInstanceCount > 0 ? (
          <li>
            <strong>
              {replay.untimedInstanceCount} instance
              {replay.untimedInstanceCount === 1 ? "" : "s"} carry no timestamps.
            </strong>{" "}
            Nothing places them on the axis, so they stay unstarted through the replay and take
            their reported state only at the axis end.
          </li>
        ) : null}
        <li>
          <strong>Fan-outs over {MAPPED_COLLAPSE_THRESHOLD} batches</strong> collapse to one summary
          row where every batch shares a lane; expand it (the ×N control) for a row per batch.
        </li>
        <li>
          Stage sub-DAG runs are matched to this master run by start time, as the stage headers say
          — the replay inherits that heuristic.
        </li>
      </ul>
    </details>
  );
}
