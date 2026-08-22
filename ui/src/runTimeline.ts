// The run replay: how one master run progressed, reconstructed CLIENT-SIDE
// from the timestamps /runtime/runs/{id}/graph already serves. No new endpoint
// and no event log — which is exactly why the honest limits below are stated
// in the UI, not only here:
//
//   - the payload carries ONE row per task instance, so a retried task shows
//     its LATEST attempt only; earlier attempts are not in this data at all;
//   - a span is all-or-nothing: nothing here knows a task's internal progress;
//   - a deferred task (every `trigger_*`) is one continuous span even though
//     it spent most of it parked on a trigger rather than running;
//   - an instance with no timestamps cannot be placed on the axis at all.
//
// Everything in this module is a pure function of the payload plus one "now",
// so the same instant can be asked for the bars, the graph colours and the
// details panel and they can never disagree.

import type {
  MappedFanOutSummary,
  PipelineGraphResponse,
  RunGraphResponse,
  RunTaskInstance,
} from "./api";

/** The scope name the master DAG's rows and selections use. */
export const MASTER_SCOPE = "master";

/** One selected task: MASTER_SCOPE or a stage name, plus the task id. The
 * graph, the timeline rows and the details panel all speak this one identity,
 * so selecting in any of them lights up the others. */
export interface RunNodeSelection {
  scope: string;
  taskId: string;
}

/** Fan-outs wider than this collapse to one summary row; the UI says so. */
export const MAPPED_COLLAPSE_THRESHOLD = 8;

/** Painted (and counted) for an instance that has not started at this instant. */
export const NOT_STARTED_STATE = "not started";

/** Minimum axis span, so a sub-second run still has a divisible width. */
const MIN_SPAN_MS = 1000;

/** Airflow states that mean the instance will not change again. */
const TERMINAL_STATES: ReadonlySet<string> = new Set([
  "success",
  "failed",
  "skipped",
  "upstream_failed",
  "removed",
]);

export function isTerminalRunState(state: string | null | undefined): boolean {
  return state !== null && state !== undefined && TERMINAL_STATES.has(state.toLowerCase());
}

function parseInstantMs(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

const UTC_OFFSET_PATTERN = /(Z|[+-]\d{2}:?\d{2})$/;

/** The UTC offset an Airflow timestamp carries, in minutes, or null. */
function parseOffsetMinutes(value: string | null | undefined): number | null {
  if (!value) return null;
  const match = UTC_OFFSET_PATTERN.exec(value);
  if (!match) return null;
  const token = match[1] as string;
  if (token === "Z") return 0;
  const digits = token.slice(1).replace(":", "");
  const hours = Number.parseInt(digits.slice(0, 2), 10);
  const minutes = Number.parseInt(digits.slice(2, 4), 10);
  if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return null;
  return (token.startsWith("-") ? -1 : 1) * (hours * 60 + minutes);
}

/**
 * One instant rendered in the offset the run's own timestamps carry, so an
 * axis hover reads exactly like the run table's `started` column rather than
 * silently switching the reader into UTC or into their browser's zone.
 */
export function formatInstant(instantMs: number, offsetMinutes: number): string {
  const shifted = new Date(instantMs + offsetMinutes * 60_000);
  if (!Number.isFinite(shifted.getTime())) return "—";
  return shifted.toISOString().replace("T", " ").slice(0, 19);
}

/** "+00:00" / "-05:30" — the suffix that says which clock formatInstant used. */
export function formatOffsetLabel(offsetMinutes: number): string {
  const sign = offsetMinutes < 0 ? "-" : "+";
  const magnitude = Math.abs(offsetMinutes);
  const hours = String(Math.floor(magnitude / 60)).padStart(2, "0");
  const minutes = String(magnitude % 60).padStart(2, "0");
  return `${sign}${hours}:${minutes}`;
}

// ---- one instance on the axis ---------------------------------------------------

export interface TimedInstance {
  instance: RunTaskInstance;
  queuedAtMs: number | null;
  startedAtMs: number | null;
  endedAtMs: number | null;
  /** Started, no end recorded, and the reported state is still in flight. */
  open: boolean;
  /** Neither queued nor started: nothing places this instance on the axis. */
  untimed: boolean;
}

export function timeInstance(instance: RunTaskInstance): TimedInstance {
  const queuedAtMs = parseInstantMs(instance.queued_at);
  const startedAtMs = parseInstantMs(instance.start_date);
  const endedAtMs = parseInstantMs(instance.end_date);
  return {
    instance,
    queuedAtMs,
    startedAtMs,
    endedAtMs,
    open: startedAtMs !== null && endedAtMs === null && !isTerminalRunState(instance.state),
    untimed: queuedAtMs === null && startedAtMs === null,
  };
}

export type ReplaySpanKind = "queued" | "run";

export interface ReplaySpan {
  kind: ReplaySpanKind;
  fromMs: number;
  toMs: number;
  /** No end recorded: the span runs to the axis edge and is drawn unclosed. */
  open: boolean;
}

/**
 * The bars one instance contributes: the queued wait (only when the payload
 * carries `queued_at`) and the run itself.
 *
 * A terminal instance with no `end_date` gets a ZERO-LENGTH run span at its
 * start rather than a bar stretching to the axis edge: "skipped" did not run
 * for the rest of the window, and inventing that span would be the one lie
 * this whole module exists to avoid.
 */
export function instanceSpans(timed: TimedInstance, windowEndMs: number): ReplaySpan[] {
  const spans: ReplaySpan[] = [];
  const { queuedAtMs, startedAtMs, endedAtMs, open } = timed;
  if (queuedAtMs !== null) {
    const queuedUntilMs = startedAtMs ?? windowEndMs;
    if (queuedUntilMs > queuedAtMs) {
      spans.push({
        kind: "queued",
        fromMs: queuedAtMs,
        toMs: queuedUntilMs,
        open: startedAtMs === null,
      });
    }
  }
  if (startedAtMs !== null) {
    spans.push({
      kind: "run",
      fromMs: startedAtMs,
      toMs: endedAtMs ?? (open ? windowEndMs : startedAtMs),
      open: endedAtMs === null && open,
    });
  }
  return spans;
}

export type ReplayPhase = "pending" | "queued" | "running" | "final";

export interface ReplayedState {
  phase: ReplayPhase;
  /** State name to paint with; null before the instance exists on the axis. */
  state: string | null;
  /** Seconds elapsed inside the run span at this instant, while running. */
  elapsedS: number | null;
}

/**
 * What one instance was doing at `atMs`: not started before its start, running
 * between start and end, its reported state after the end.
 *
 * An untimed instance (no queue, no start — a skip Airflow never scheduled)
 * cannot be placed, so it stays "pending" for the whole replay and only takes
 * its reported state at the axis edge, where the replay meets the live graph.
 */
export function replayStateAt(
  timed: TimedInstance,
  atMs: number,
  windowEndMs: number,
): ReplayedState {
  const { instance, queuedAtMs, startedAtMs, endedAtMs, open, untimed } = timed;
  if (untimed) {
    return atMs >= windowEndMs
      ? { phase: "final", state: instance.state, elapsedS: null }
      : { phase: "pending", state: null, elapsedS: null };
  }
  const enteredMs = queuedAtMs ?? (startedAtMs as number);
  if (atMs < enteredMs) return { phase: "pending", state: null, elapsedS: null };
  if (startedAtMs === null || atMs < startedAtMs) {
    return { phase: "queued", state: "queued", elapsedS: null };
  }
  // A terminal state with no end_date transitions at its start (see
  // instanceSpans); an open one has not transitioned at all yet.
  const finishedMs = endedAtMs ?? (open ? null : startedAtMs);
  if (finishedMs !== null && atMs >= finishedMs) {
    return { phase: "final", state: instance.state, elapsedS: null };
  }
  return {
    // An open instance reports what it is doing (running, deferred, …); a
    // closed one is simply "running" — its span carries no finer truth.
    phase: "running",
    state: open ? (instance.state ?? "running") : "running",
    elapsedS: (atMs - startedAtMs) / 1000,
  };
}

/**
 * The same instances as the payload, restated at one instant: the state each
 * was in, and the duration elapsed SO FAR rather than the final one. Feeding
 * these to the existing graph code is what recolours the DAG.
 */
export function replayTaskInstances(
  instances: readonly RunTaskInstance[],
  atMs: number,
  windowEndMs: number,
): RunTaskInstance[] {
  return instances.map((instance) => {
    const replayed = replayStateAt(timeInstance(instance), atMs, windowEndMs);
    return {
      ...instance,
      state: replayed.state,
      duration_s:
        replayed.phase === "running"
          ? replayed.elapsedS
          : replayed.phase === "final"
            ? instance.duration_s
            : null,
    };
  });
}

/** The fan-out split at one instant; the total stays the run's real total. */
export function replayedFanOutSummary(
  summary: MappedFanOutSummary | null,
  replayedInstances: readonly RunTaskInstance[],
): MappedFanOutSummary | null {
  if (summary === null) return null;
  const byState: Record<string, number> = {};
  for (const instance of replayedInstances) {
    if (instance.task_id !== summary.task_id || instance.map_index < 0) continue;
    const key = instance.state ?? NOT_STARTED_STATE;
    byState[key] = (byState[key] ?? 0) + 1;
  }
  return { task_id: summary.task_id, total: summary.total, by_state: byState };
}

/**
 * A DAG run's own state at one instant. Only three answers are derivable
 * without inventing a verdict: nothing started yet, everything reached its
 * end (so the reported state stands), or the run is still in flight.
 */
export function replayedRunState(
  instances: readonly RunTaskInstance[],
  reportedState: string | null,
  atMs: number,
  windowEndMs: number,
): string | null {
  if (instances.length === 0) return reportedState;
  let allPending = true;
  let allFinal = true;
  for (const instance of instances) {
    const { phase } = replayStateAt(timeInstance(instance), atMs, windowEndMs);
    if (phase !== "pending") allPending = false;
    if (phase !== "final") allFinal = false;
  }
  if (allFinal) return reportedState;
  if (allPending) return NOT_STARTED_STATE;
  return "running";
}

// ---- rows ------------------------------------------------------------------------

export interface ReplayBatchRow {
  key: string;
  mapIndex: number;
  timed: TimedInstance;
}

export interface ReplayRow {
  key: string;
  /** MASTER_SCOPE, or the stage name — the same scope the selection speaks. */
  scope: string;
  taskId: string;
  kind: "task" | "fanout";
  /** The topology's description of the task, for the row's hover text. */
  summary: string;
  mapped: boolean;
  deferred: boolean;
  /** Every instance behind this row (one task instance, or every batch). */
  timed: TimedInstance[];
  /** Fan-out rows only: one entry per map_index, in map order. */
  batches: ReplayBatchRow[];
  /** Above 1 when the payload is showing a retried task's latest attempt. */
  retriedTryNumber: number | null;
}

export interface ReplayGroup {
  scope: string;
  title: string;
  dagId: string | null;
  dagRunId: string | null;
  /** The run state the server reported for this DAG run. */
  state: string | null;
  /** Every instance in the group, for the group's replayed state. */
  instances: RunTaskInstance[];
  rows: ReplayRow[];
  /** Why this group has no rows, when it has none. */
  note: string | null;
}

export interface RunReplay {
  groups: ReplayGroup[];
  /** Earliest queue/start across the whole run — the axis origin. */
  startMs: number;
  /** Latest end, or now while anything is still open. */
  endMs: number;
  spanMs: number;
  /** False when no instance carries a timestamp: there is nothing to replay. */
  hasTimedInstances: boolean;
  /** Something is still open, or the master run itself has not finished. */
  unfinished: boolean;
  untimedInstanceCount: number;
  retriedTaskCount: number;
  deferredTaskCount: number;
  /** UTC offset the run's timestamps carry, for absolute-time readouts. */
  offsetMinutes: number;
}

function retriedTryNumberOf(instances: readonly RunTaskInstance[]): number | null {
  let highest = 1;
  for (const instance of instances) {
    if (instance.try_number !== null && instance.try_number > highest)
      highest = instance.try_number;
  }
  return highest > 1 ? highest : null;
}

function buildGroup(
  scope: string,
  title: string,
  dagId: string | null,
  dagRunId: string | null,
  state: string | null,
  topologyTasks: readonly {
    task_id: string;
    summary: string;
    mapped: boolean;
    deferred: boolean;
  }[],
  instances: readonly RunTaskInstance[],
  note: string | null,
): ReplayGroup {
  const instancesByTaskId = new Map<string, RunTaskInstance[]>();
  for (const instance of instances) {
    const bucket = instancesByTaskId.get(instance.task_id);
    if (bucket) bucket.push(instance);
    else instancesByTaskId.set(instance.task_id, [instance]);
  }

  // Topology order first (the payload is already sorted that way, but the
  // topology is the one owner of the order AND of the mapped/deferred flags),
  // then any task id the topology does not describe rather than dropping it.
  const orderedTaskIds = topologyTasks.map((task) => task.task_id);
  for (const taskId of instancesByTaskId.keys()) {
    if (!orderedTaskIds.includes(taskId)) orderedTaskIds.push(taskId);
  }

  const rows: ReplayRow[] = [];
  for (const taskId of orderedTaskIds) {
    const taskInstances = instancesByTaskId.get(taskId) ?? [];
    if (taskInstances.length === 0) continue;
    const node = topologyTasks.find((task) => task.task_id === taskId);
    const mappedInstances = taskInstances
      .filter((instance) => instance.map_index >= 0)
      .sort((left, right) => left.map_index - right.map_index);
    const isFanOut = mappedInstances.length > 0;
    const rowInstances = isFanOut ? mappedInstances : taskInstances;
    rows.push({
      key: `${scope}:${taskId}`,
      scope,
      taskId,
      kind: isFanOut ? "fanout" : "task",
      summary: node?.summary ?? "This task is not part of the topology this server describes.",
      mapped: node?.mapped ?? isFanOut,
      deferred: node?.deferred ?? false,
      timed: rowInstances.map(timeInstance),
      batches: isFanOut
        ? mappedInstances.map((instance) => ({
            key: `${scope}:${taskId}:${instance.map_index}`,
            mapIndex: instance.map_index,
            timed: timeInstance(instance),
          }))
        : [],
      retriedTryNumber: retriedTryNumberOf(rowInstances),
    });
  }

  return {
    scope,
    title,
    dagId,
    dagRunId,
    state,
    instances: [...instances],
    rows,
    note: rows.length === 0 ? note : null,
  };
}

/**
 * Every row of the replay plus the shared axis. `nowMs` only matters while the
 * run is unfinished: it is the right-hand edge an open span runs to.
 */
export function buildRunReplay(
  runGraph: RunGraphResponse,
  graph: PipelineGraphResponse,
  nowMs: number,
): RunReplay {
  const groups: ReplayGroup[] = [
    buildGroup(
      MASTER_SCOPE,
      MASTER_SCOPE,
      graph.master.dag_id,
      runGraph.master.dag_run_id,
      runGraph.master.state,
      graph.master.tasks,
      runGraph.master.tasks,
      "No task of the master DAG has an instance yet.",
    ),
  ];
  for (const stage of runGraph.stages) {
    const topology = graph.stages.find((entry) => entry.stage === stage.stage)?.dag;
    groups.push(
      buildGroup(
        stage.stage,
        stage.stage,
        stage.dag_id,
        stage.dag_run_id,
        stage.state,
        topology?.tasks ?? [],
        stage.tasks,
        stage.dag_run_id === null
          ? "This stage did not run for this master run."
          : "This stage's run has no task instances.",
      ),
    );
  }

  let earliestMs = Number.POSITIVE_INFINITY;
  let latestMs = Number.NEGATIVE_INFINITY;
  let unfinished = !isTerminalRunState(runGraph.master.state);
  let untimedInstanceCount = 0;
  let retriedTaskCount = 0;
  let deferredTaskCount = 0;
  for (const group of groups) {
    for (const row of group.rows) {
      if (row.retriedTryNumber !== null) retriedTaskCount += 1;
      if (row.deferred) deferredTaskCount += 1;
      for (const timed of row.timed) {
        if (timed.untimed) untimedInstanceCount += 1;
        if (timed.open) unfinished = true;
        for (const candidate of [timed.queuedAtMs, timed.startedAtMs]) {
          if (candidate !== null && candidate < earliestMs) earliestMs = candidate;
        }
        for (const candidate of [timed.startedAtMs, timed.endedAtMs]) {
          if (candidate !== null && candidate > latestMs) latestMs = candidate;
        }
      }
    }
  }

  const hasTimedInstances = Number.isFinite(earliestMs);
  const startMs = hasTimedInstances ? earliestMs : nowMs;
  const observedEndMs = hasTimedInstances ? Math.max(latestMs, startMs) : nowMs;
  const endMs = Math.max(
    unfinished ? Math.max(observedEndMs, nowMs) : observedEndMs,
    startMs + MIN_SPAN_MS,
  );

  return {
    groups,
    startMs,
    endMs,
    spanMs: endMs - startMs,
    hasTimedInstances,
    unfinished,
    untimedInstanceCount,
    retriedTaskCount,
    deferredTaskCount,
    offsetMinutes: runTimestampOffsetMinutes(runGraph),
  };
}

function runTimestampOffsetMinutes(runGraph: RunGraphResponse): number {
  const everyInstance = [runGraph.master.tasks, ...runGraph.stages.map((stage) => stage.tasks)];
  for (const instances of everyInstance) {
    for (const instance of instances) {
      for (const value of [instance.start_date, instance.queued_at, instance.end_date]) {
        const offsetMinutes = parseOffsetMinutes(value);
        if (offsetMinutes !== null) return offsetMinutes;
      }
    }
  }
  return 0;
}

/** True while the run can still change — the only reason to keep a clock. */
export function isRunUnfinished(runGraph: RunGraphResponse): boolean {
  if (!isTerminalRunState(runGraph.master.state)) return true;
  const everyInstance = [runGraph.master.tasks, ...runGraph.stages.map((stage) => stage.tasks)];
  return everyInstance.some((instances) =>
    instances.some((instance) => timeInstance(instance).open),
  );
}

// ---- axis ------------------------------------------------------------------------

/** Tick steps in seconds: seconds, then the minute/hour steps a clock uses. */
const TICK_STEPS_SECONDS: readonly number[] = [
  0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 21600, 43200,
  86400,
];
const MAX_TICKS = 40;

/** Offsets (seconds from the run's start) for the axis ticks. */
export function axisTickOffsetsSeconds(spanSeconds: number, targetCount = 6): number[] {
  const roughStep = spanSeconds / Math.max(1, targetCount);
  const step =
    TICK_STEPS_SECONDS.find((candidate) => candidate >= roughStep) ??
    Math.max(roughStep, TICK_STEPS_SECONDS[TICK_STEPS_SECONDS.length - 1] ?? 86400);
  const offsets: number[] = [];
  for (let index = 0; index < MAX_TICKS; index++) {
    const offset = index * step;
    if (offset > spanSeconds + 1e-6) break;
    offsets.push(Number(offset.toFixed(3)));
  }
  return offsets;
}
