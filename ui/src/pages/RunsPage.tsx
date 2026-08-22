import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ExternalLink, Play, RefreshCw } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  type DagTopology,
  describeApiError,
  fetchPipelineGraph,
  fetchRunGraph,
  fetchRuntimeRuns,
  fetchRuntimeStatus,
  fetchWorkspaceConfig,
  type MappedFanOutSummary,
  type PipelineGraphResponse,
  type RunGraphResponse,
  type RunGraphStage,
  type RunTaskInstance,
  type RuntimeHealth,
  type RuntimeRun,
  type RuntimeStatus,
  STAGE_ORDER,
  type StageRecentRuns,
  triggerIngest,
  type WorkspaceConfig,
} from "../api";
import { DagGraph, type DagGraphNode } from "../components/DagGraph";
import { DetailBlock, DetailRow, DetailsPanel } from "../components/DetailsPanel";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "../components/QueryStates";
import { RunStateChip } from "../components/RunStateChip";
import { RunTimeline } from "../components/RunTimeline";
import {
  formatDurationBetween,
  formatDurationCompact,
  formatOffsetCompact,
  formatTimestamp,
} from "../format";
import { runStateTone } from "../runState";
import {
  buildRunReplay,
  formatInstant,
  formatOffsetLabel,
  isRunUnfinished,
  MASTER_SCOPE,
  NOT_STARTED_STATE,
  type RunNodeSelection,
  replayedFanOutSummary,
  replayedRunState,
  replayTaskInstances,
} from "../runTimeline";
import { useNowTick } from "../useNowTick";
import { usePrefersReducedMotion } from "../usePrefersReducedMotion";

const RUNS_PAGE_LIMIT = 25;
const AUTO_REFRESH_INTERVAL_MS = 10_000;

// How often the replay's "now" edge advances while a run is unfinished. The
// axis only has to keep up with the reader's eye, and a reader who asked for
// reduced motion gets the slow clock rather than a creeping edge.
const LIVE_TICK_INTERVAL_MS = 1_000;
const LIVE_TICK_INTERVAL_REDUCED_MOTION_MS = 5_000;

// The runs monitor renders exactly what the server proxied from Airflow —
// the browser never sees Airflow credentials, only these JSON summaries.

type HealthTone = "ok" | "err" | "warn" | "muted";

function healthTone(status: string | null): HealthTone {
  if (status === null) return "muted";
  if (status.toLowerCase() === "healthy") return "ok";
  if (status.toLowerCase() === "unhealthy") return "err";
  return "warn";
}

const HEALTH_COMPONENT_NAMES: readonly (keyof RuntimeHealth)[] = [
  "metadatabase",
  "scheduler",
  "triggerer",
  "dag_processor",
];

function StatusTiles({ status }: { status: RuntimeStatus }) {
  return (
    <div className="status-tiles">
      <div className="status-tile">
        <span className="status-tile-label">source</span>
        <span className="status-tile-value">{status.source ?? "—"}</span>
      </div>
      <div className="status-tile">
        <span className="status-tile-label">dag</span>
        <span className="status-tile-value status-tile-mono" title={status.dag_id ?? undefined}>
          {status.dag_id ?? "—"}
        </span>
        {status.registered === null ? null : (
          <span className={status.registered ? "chip chip-ok" : "chip chip-warn"}>
            {status.registered ? "registered" : "not registered"}
          </span>
        )}
      </div>
      {HEALTH_COMPONENT_NAMES.map((componentName) => {
        const componentStatus = status.health ? status.health[componentName] : null;
        const tone = healthTone(componentStatus);
        return (
          <div key={componentName} className={`status-tile health-tile is-${tone}`}>
            <span className="status-tile-label">{componentName.replace("_", " ")}</span>
            <span className="status-tile-value">
              <span className={`health-dot is-${tone}`} aria-hidden="true" />
              {componentStatus ?? "absent"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function confSummaryText(conf: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(conf)) {
    if (parts.length === 4) {
      parts.push("…");
      break;
    }
    if (Array.isArray(value)) parts.push(`${key} ×${value.length}`);
    else if (value !== null && typeof value === "object") parts.push(`${key}={…}`);
    else parts.push(`${key}=${String(value)}`);
  }
  return parts.length > 0 ? parts.join(" · ") : "—";
}

/** Airflow 3 web route for one dag run, or null when the base/dag is unknown. */
function airflowRunUrl(
  webUrlBase: string | null,
  dagId: string | null,
  dagRunId: string | null,
): string | null {
  // No base, no dag, or no run id means there is nothing to deep-link to; the
  // caller renders no link rather than a broken one.
  if (!webUrlBase || !dagId || !dagRunId) return null;
  const base = webUrlBase.replace(/\/$/, "");
  return `${base}/dags/${encodeURIComponent(dagId)}/runs/${encodeURIComponent(dagRunId)}`;
}

function airflowTaskUrl(
  webUrlBase: string | null,
  dagId: string | null,
  dagRunId: string | null,
  taskId: string,
): string | null {
  if (!dagRunId) return null;
  const runUrl = airflowRunUrl(webUrlBase, dagId, dagRunId);
  return runUrl ? `${runUrl}/tasks/${encodeURIComponent(taskId)}` : null;
}

// ---- run graph ----------------------------------------------------------------

/** One node's live picture: the instances Airflow reported for that task id. */
interface TaskInstanceGroup {
  instances: RunTaskInstance[];
  /** The state to paint the node with; the badge carries the exact counts. */
  state: string | null;
  badge: string | null;
}

function summarizeInstances(
  instances: RunTaskInstance[],
  mappedSummary: MappedFanOutSummary | null,
): TaskInstanceGroup {
  if (instances.length === 0) return { instances, state: null, badge: null };
  const isMapped = instances.some((instance) => instance.map_index >= 0);
  if (!isMapped) {
    const single = instances[0] as RunTaskInstance;
    return {
      instances,
      state: single.state,
      badge: single.duration_s === null ? null : formatDurationCompact(single.duration_s),
    };
  }
  const byState = new Map<string, number>();
  for (const instance of instances) {
    const stateName = instance.state ?? "no state";
    byState.set(stateName, (byState.get(stateName) ?? 0) + 1);
  }
  const total = mappedSummary?.total ?? instances.length;
  const succeeded = mappedSummary?.by_state.success ?? byState.get("success") ?? 0;
  // The painted state names the most urgent thing happening in the fan-out;
  // the badge below it always shows the real split, so nothing is hidden.
  const dominantState =
    (byState.has("running") && "running") ||
    (byState.has("failed") && "failed") ||
    (byState.has("upstream_failed") && "upstream_failed") ||
    (byState.has("deferred") && "deferred") ||
    (succeeded === total && "success") ||
    instances[0]?.state ||
    null;
  return { instances, state: dominantState, badge: `${succeeded}/${total} ok` };
}

function instanceGroupsByTaskId(
  tasks: RunTaskInstance[],
  mappedSummary: MappedFanOutSummary | null,
): Map<string, TaskInstanceGroup> {
  const byTaskId = new Map<string, RunTaskInstance[]>();
  for (const instance of tasks) {
    // task_id is nullable in the served payload; an instance without one
    // belongs to no node on the graph, so it is left out of the grouping
    // rather than bucketed under a fabricated key.
    const taskId = instance.task_id;
    if (taskId === null) continue;
    const bucket = byTaskId.get(taskId);
    if (bucket) bucket.push(instance);
    else byTaskId.set(taskId, [instance]);
  }
  const groups = new Map<string, TaskInstanceGroup>();
  for (const [taskId, instances] of byTaskId) {
    groups.set(
      taskId,
      summarizeInstances(
        instances,
        mappedSummary && mappedSummary.task_id === taskId ? mappedSummary : null,
      ),
    );
  }
  return groups;
}

/**
 * Nodes for one DAG. `unstartedState` is what a task with nothing to paint
 * reads as: live that means Airflow has no instance for it at all, but at a
 * replay instant it means the instance exists and had not started yet — two
 * different truths that must not share one word.
 */
function liveNodes(
  topology: DagTopology,
  groups: Map<string, TaskInstanceGroup>,
  unstartedState: string,
): DagGraphNode[] {
  return topology.tasks.map((task) => {
    const group = groups.get(task.task_id);
    return {
      id: task.task_id,
      label: task.task_id,
      summary: task.summary,
      mapped: task.mapped,
      deferred: task.deferred,
      state: group?.state ?? unstartedState,
      badge: group?.badge ?? (task.mapped ? "×0" : null),
    };
  });
}

function TaskInstanceDetails({
  selection,
  group,
  summary,
  dagId,
  dagRunId,
  airflowWebUrl,
  replayAtMs,
  replayEndMs,
  replayOffsetMinutes,
  replayStartMs,
}: {
  selection: RunNodeSelection;
  group: TaskInstanceGroup | undefined;
  summary: string;
  dagId: string;
  dagRunId: string | null;
  airflowWebUrl: string | null;
  /** The replay instant the panel reports state at; null = the live payload. */
  replayAtMs: number | null;
  replayEndMs: number;
  replayOffsetMinutes: number;
  replayStartMs: number;
}) {
  const taskUrl = airflowTaskUrl(airflowWebUrl, dagId, dagRunId, selection.taskId);
  const rawInstances = group?.instances ?? [];
  // The panel keeps the facts (timestamps, tries, real durations) whatever the
  // playhead says; only the STATE is restated at the replay instant.
  const replayedGroup =
    replayAtMs === null || group === undefined
      ? undefined
      : summarizeInstances(replayTaskInstances(rawInstances, replayAtMs, replayEndMs), null);
  // A replayed null state means "had not started", which is a real answer —
  // it must not fall through to the run's final state. The fallback word
  // matches the graph node's for the same instant.
  const unstartedState = replayAtMs === null ? "no instance" : NOT_STARTED_STATE;
  const shownState =
    replayedGroup !== undefined
      ? (replayedGroup.state ?? unstartedState)
      : (group?.state ?? unstartedState);
  const mappedInstances = (
    replayAtMs === null ? rawInstances : replayTaskInstances(rawInstances, replayAtMs, replayEndMs)
  ).filter((instance) => instance.map_index >= 0);
  const single = rawInstances.length === 1 ? rawInstances[0] : undefined;
  return (
    <DetailsPanel
      title={selection.taskId}
      kicker={selection.scope === MASTER_SCOPE ? "master task" : `${selection.scope} task`}
    >
      <p className="details-prose">{summary}</p>
      <DetailRow label="state">
        <RunStateChip state={shownState} />
      </DetailRow>
      {replayAtMs === null ? null : (
        <DetailRow label="at">
          <span className="detail-replay-at">
            +{formatOffsetCompact((replayAtMs - replayStartMs) / 1000)} ·{" "}
            {formatInstant(replayAtMs, replayOffsetMinutes)}
          </span>
        </DetailRow>
      )}
      {single ? (
        <>
          <DetailRow label="started">
            {single.start_date ? formatTimestamp(single.start_date) : "—"}
          </DetailRow>
          <DetailRow label="ended">
            {single.end_date ? formatTimestamp(single.end_date) : "—"}
          </DetailRow>
          <DetailRow label="duration">{formatDurationCompact(single.duration_s)}</DetailRow>
          <DetailRow label="tries">{single.try_number ?? "—"}</DetailRow>
        </>
      ) : null}
      {mappedInstances.length > 0 ? (
        <DetailBlock
          label={`mapped instances · ${mappedInstances.length}${
            replayAtMs === null ? "" : " · at the playhead"
          }`}
        >
          <ul className="instance-list">
            {mappedInstances.map((instance) => (
              <li key={instance.map_index} className="instance-row">
                <span className="instance-index">[{instance.map_index}]</span>
                <RunStateChip state={instance.state} />
                <span className="instance-duration">
                  {formatDurationCompact(instance.duration_s)}
                </span>
                {instance.try_number !== null && instance.try_number > 1 ? (
                  <span
                    className="chip chip-warn"
                    title="Only this last attempt is in the payload; earlier ones are not."
                  >
                    try {instance.try_number}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </DetailBlock>
      ) : null}
      {replayAtMs === null ? null : (
        <p className="details-hint">
          States are reconstructed at the playhead; the timings above are the run's own.
        </p>
      )}
      {taskUrl ? (
        <a className="btn btn-tiny" href={taskUrl} target="_blank" rel="noreferrer">
          <ExternalLink />
          <span>Open in Airflow</span>
        </a>
      ) : null}
    </DetailsPanel>
  );
}

/** The batch fan-out at a glance: one proportional bar in the state palette,
 * with the counts spelled out beside it so the split never rests on colour. */
function FanOutSummary({ summary }: { summary: MappedFanOutSummary }) {
  const succeeded = summary.by_state.success ?? 0;
  const otherStates = Object.entries(summary.by_state)
    .filter(([stateName]) => stateName !== "success")
    .sort((left, right) => right[1] - left[1]);
  return (
    <span className="fanout">
      <span className="fanout-bar" aria-hidden="true">
        {Object.entries(summary.by_state).map(([stateName, count]) => (
          <span
            key={stateName}
            className={`fanout-seg is-${runStateTone(stateName)}`}
            style={{ flexGrow: count }}
            title={`${count} ${stateName}`}
          />
        ))}
      </span>
      <span className="fanout-readout">
        {succeeded} of {summary.total} batches succeeded
        {otherStates.map(([stateName, count]) => ` · ${count} ${stateName}`).join("")}
      </span>
    </span>
  );
}

function StageRunGraph({
  stage,
  topology,
  selection,
  onSelect,
  airflowWebUrl,
  replayAtMs,
  replayEndMs,
}: {
  stage: RunGraphStage;
  topology: DagTopology | undefined;
  selection: RunNodeSelection | null;
  onSelect: (selection: RunNodeSelection) => void;
  airflowWebUrl: string | null;
  /** Non-null while the playhead is parked: paint this instant, not the live state. */
  replayAtMs: number | null;
  replayEndMs: number;
}) {
  // Replaying re-states the instances at the playhead and recounts the
  // fan-out from them; the server's summary is the run's final split, which
  // would contradict the bars mid-scrub.
  const tasks = useMemo(
    () =>
      replayAtMs === null ? stage.tasks : replayTaskInstances(stage.tasks, replayAtMs, replayEndMs),
    [stage.tasks, replayAtMs, replayEndMs],
  );
  const mappedSummary = useMemo(
    () =>
      replayAtMs === null
        ? stage.mapped_summary
        : replayedFanOutSummary(stage.mapped_summary, tasks),
    [stage.mapped_summary, tasks, replayAtMs],
  );
  const groups = useMemo(
    () => instanceGroupsByTaskId(tasks, mappedSummary),
    [tasks, mappedSummary],
  );
  const stageState =
    replayAtMs === null
      ? stage.state
      : replayedRunState(stage.tasks, stage.state, replayAtMs, replayEndMs);
  const runUrl = stage.dag_run_id
    ? airflowRunUrl(airflowWebUrl, stage.dag_id, stage.dag_run_id)
    : null;
  return (
    <div className="stage-lane-card is-expanded">
      <div className="stage-run-head">
        <span className="lane-stage">{stage.stage}</span>
        <RunStateChip state={stageState ?? "not run"} />
        <code className="cell-mono stage-run-dag" title={stage.dag_id}>
          {stage.dag_id}
        </code>
        {mappedSummary ? <FanOutSummary summary={mappedSummary} /> : null}
        <span className="toolbar-spacer" />
        {stage.match === "heuristic" ? (
          <span
            className="chip chip-muted"
            title="Matched to this master run by start time — Airflow stores no parent-run link for triggered sub-DAGs."
          >
            matched by time
          </span>
        ) : null}
        {runUrl ? (
          <a className="btn btn-ghost btn-tiny" href={runUrl} target="_blank" rel="noreferrer">
            <ExternalLink />
          </a>
        ) : null}
      </div>
      {stage.dag_run_id === null ? (
        <p className="empty-note stage-run-empty">
          This stage did not run for this master run (the run profile skipped it, or the chain
          stopped before it).
        </p>
      ) : !topology ? (
        <p className="empty-note stage-run-empty">
          No topology for this stage — the server does not describe it.
        </p>
      ) : (
        <DagGraph
          nodes={liveNodes(
            topology,
            groups,
            replayAtMs === null ? "no instance" : NOT_STARTED_STATE,
          )}
          edges={topology.edges.map(([from, to]) => ({ from, to }))}
          label={`${stage.stage} sub-DAG for this run`}
          selectedNodeId={selection?.scope === stage.stage ? selection.taskId : null}
          onSelectNode={(taskId) => onSelect({ scope: stage.stage, taskId })}
        />
      )}
    </div>
  );
}

function RunGraphSection({
  runGraph,
  graph,
  airflowWebUrl,
  isAutoRefreshOn,
}: {
  runGraph: RunGraphResponse;
  graph: PipelineGraphResponse;
  airflowWebUrl: string | null;
  isAutoRefreshOn: boolean;
}) {
  const [selection, setSelection] = useState<RunNodeSelection | null>(null);
  // null = pinned to the axis end, which for an unfinished run is live; any
  // number is a parked playhead, and parking is what detaches from live.
  const [playheadChoice, setPlayheadChoice] = useState<number | null>(null);

  const prefersReducedMotion = usePrefersReducedMotion();
  const unfinished = isRunUnfinished(runGraph);
  const nowMs = useNowTick(
    unfinished
      ? prefersReducedMotion
        ? LIVE_TICK_INTERVAL_REDUCED_MOTION_MS
        : LIVE_TICK_INTERVAL_MS
      : null,
  );
  const replay = useMemo(() => buildRunReplay(runGraph, graph, nowMs), [runGraph, graph, nowMs]);
  const playheadMs = playheadChoice ?? replay.endMs;
  // Only a parked playhead repaints the graphs; pinned to the end, the page
  // shows the payload exactly as the server sent it.
  const replayAtMs = playheadChoice === null ? null : playheadMs;

  const masterTasks = useMemo(
    () =>
      replayAtMs === null
        ? runGraph.master.tasks
        : replayTaskInstances(runGraph.master.tasks, replayAtMs, replay.endMs),
    [runGraph.master.tasks, replayAtMs, replay.endMs],
  );
  const masterGroups = useMemo(() => instanceGroupsByTaskId(masterTasks, null), [masterTasks]);
  // The details panel reads the RAW instances: its timestamps and tries are
  // facts about the run, not about the instant being replayed.
  const rawMasterGroups = useMemo(
    () => instanceGroupsByTaskId(runGraph.master.tasks, null),
    [runGraph.master.tasks],
  );

  const selectedStage =
    selection && selection.scope !== MASTER_SCOPE
      ? runGraph.stages.find((stage) => stage.stage === selection.scope)
      : undefined;
  const selectedGroup =
    selection === null
      ? undefined
      : selection.scope === MASTER_SCOPE
        ? rawMasterGroups.get(selection.taskId)
        : selectedStage
          ? instanceGroupsByTaskId(selectedStage.tasks, selectedStage.mapped_summary).get(
              selection.taskId,
            )
          : undefined;
  const selectedTopology =
    selection === null
      ? undefined
      : selection.scope === MASTER_SCOPE
        ? graph.master
        : graph.stages.find((stage) => stage.stage === selection.scope)?.dag;
  const selectedSummary =
    selectedTopology?.tasks.find((task) => task.task_id === selection?.taskId)?.summary ??
    "This task is not part of the topology this server describes.";

  const masterState =
    replayAtMs === null
      ? runGraph.master.state
      : replayedRunState(runGraph.master.tasks, runGraph.master.state, replayAtMs, replay.endMs);

  return (
    <section className="section">
      <h2 className="section-title">Run graph</h2>
      <div className="run-graph-head">
        <RunStateChip state={masterState} />
        <code className="cell-mono run-graph-id" title={runGraph.master.dag_run_id}>
          {runGraph.master.dag_run_id}
        </code>
        {replayAtMs === null ? null : (
          <span
            className="chip chip-accent"
            title={`The graphs are painted as of ${formatInstant(
              replayAtMs,
              replay.offsetMinutes,
            )} ${formatOffsetLabel(replay.offsetMinutes)}, not at the live state.`}
          >
            replay +{formatOffsetCompact((replayAtMs - replay.startMs) / 1000)}
          </span>
        )}
      </div>
      <div className={replayAtMs === null ? "graph-layout" : "graph-layout is-replaying"}>
        <div className="graph-column">
          <DagGraph
            nodes={liveNodes(
              graph.master,
              masterGroups,
              replayAtMs === null ? "no instance" : NOT_STARTED_STATE,
            )}
            edges={graph.master.edges.map(([from, to]) => ({ from, to }))}
            label="Master DAG for this run"
            selectedNodeId={selection?.scope === MASTER_SCOPE ? selection.taskId : null}
            onSelectNode={(taskId) => setSelection({ scope: MASTER_SCOPE, taskId })}
          />
          <RunTimeline
            replay={replay}
            playheadMs={playheadMs}
            onPlayheadChange={setPlayheadChoice}
            isPinnedToEnd={playheadChoice === null}
            selection={selection}
            onSelect={setSelection}
            isAutoRefreshOn={isAutoRefreshOn}
          />
          <div className="stage-lane-list">
            {runGraph.stages.map((stage) => (
              <StageRunGraph
                key={stage.stage}
                stage={stage}
                topology={graph.stages.find((entry) => entry.stage === stage.stage)?.dag}
                selection={selection}
                onSelect={setSelection}
                airflowWebUrl={airflowWebUrl}
                replayAtMs={replayAtMs}
                replayEndMs={replay.endMs}
              />
            ))}
          </div>
        </div>
        <div className="details-column">
          {selection === null ? (
            <DetailsPanel title="No task selected" kicker="details">
              <p className="details-hint">
                Select a node in the master DAG, a row in the replay, or a node in a stage sub-DAG
                to see its state, timings and tries. Stacked nodes are mapped fan-outs: their badge
                is the batch split, and the panel lists every instance.
              </p>
            </DetailsPanel>
          ) : (
            <TaskInstanceDetails
              selection={selection}
              group={selectedGroup}
              summary={selectedSummary}
              dagId={
                selection.scope === MASTER_SCOPE
                  ? graph.master.dag_id
                  : (selectedStage?.dag_id ?? "")
              }
              dagRunId={
                selection.scope === MASTER_SCOPE
                  ? runGraph.master.dag_run_id
                  : (selectedStage?.dag_run_id ?? null)
              }
              airflowWebUrl={airflowWebUrl}
              replayAtMs={replayAtMs}
              replayEndMs={replay.endMs}
              replayStartMs={replay.startMs}
              replayOffsetMinutes={replay.offsetMinutes}
            />
          )}
        </div>
      </div>
    </section>
  );
}

// ---- tables -------------------------------------------------------------------

/**
 * A run the server could identify.
 *
 * `dag_run_id` is nullable in the served payload, and a run without one cannot
 * be selected, deep-linked, or used as a React key. {@link identifiedRuns}
 * separates those out once instead of every use site re-checking.
 */
type IdentifiedRun = Omit<RuntimeRun, "dag_run_id"> & { dag_run_id: string };

function identifiedRuns(runs: readonly RuntimeRun[]): IdentifiedRun[] {
  return runs.flatMap((run) => {
    const dagRunId = run.dag_run_id;
    return dagRunId === null ? [] : [{ ...run, dag_run_id: dagRunId }];
  });
}

function MasterRunsTable({
  runs,
  airflowWebUrl,
  dagId,
  selectedRunId,
  onSelectRun,
}: {
  runs: RuntimeRun[];
  airflowWebUrl: string | null;
  dagId: string | null;
  selectedRunId: string | null;
  onSelectRun: (dagRunId: string) => void;
}) {
  const [expandedRunIds, setExpandedRunIds] = useState<ReadonlySet<string>>(new Set());
  const hasAirflowLinks = airflowWebUrl !== null && dagId !== null;

  const toggleExpanded = (dagRunId: string) => {
    setExpandedRunIds((previous) => {
      const next = new Set(previous);
      if (next.has(dagRunId)) next.delete(dagRunId);
      else next.add(dagRunId);
      return next;
    });
  };

  if (runs.length === 0) {
    return (
      <p className="empty-note">
        No runs yet. Trigger an ingest below, or with <code>hflow ingest</code>.
      </p>
    );
  }

  const columnCount = hasAirflowLinks ? 7 : 6;

  return (
    <div className="table-overflow">
      <table className="evidence-table">
        <thead>
          <tr>
            <th aria-label="Expand" />
            <th>state</th>
            <th>run id</th>
            <th>started</th>
            <th>duration</th>
            <th>conf</th>
            {hasAirflowLinks ? <th aria-label="Airflow link" /> : null}
          </tr>
        </thead>
        <tbody>
          {identifiedRuns(runs).map((run) => {
            const isExpanded = expandedRunIds.has(run.dag_run_id);
            const isSelected = run.dag_run_id === selectedRunId;
            const runUrl = airflowRunUrl(airflowWebUrl, dagId, run.dag_run_id);
            return (
              <Fragment key={run.dag_run_id}>
                <tr className={isSelected ? "is-selected-run" : undefined}>
                  <td>
                    <button
                      type="button"
                      className="btn btn-ghost btn-tiny"
                      onClick={() => toggleExpanded(run.dag_run_id)}
                      aria-expanded={isExpanded}
                      title={isExpanded ? "Hide the full conf" : "Show the full conf"}
                    >
                      <ChevronDown className={isExpanded ? "chevron is-open" : "chevron"} />
                    </button>
                  </td>
                  <td>
                    <RunStateChip state={run.state} />
                  </td>
                  <td>
                    <button
                      type="button"
                      className={isSelected ? "run-id-button is-selected" : "run-id-button"}
                      onClick={() => onSelectRun(run.dag_run_id)}
                      title={
                        isSelected ? "Shown in the run graph above" : "Show this run in the graph"
                      }
                      aria-current={isSelected}
                    >
                      {run.dag_run_id}
                    </button>
                  </td>
                  <td className="cell-mono">
                    {run.start_date ? formatTimestamp(run.start_date) : "—"}
                  </td>
                  <td className="cell-num">
                    {formatDurationBetween(run.start_date, run.end_date)}
                  </td>
                  <td className="conf-summary" title={confSummaryText(run.conf)}>
                    {confSummaryText(run.conf)}
                  </td>
                  {hasAirflowLinks ? (
                    <td>
                      {runUrl ? (
                        <a
                          className="btn btn-ghost btn-tiny"
                          href={runUrl}
                          target="_blank"
                          rel="noreferrer"
                          title="Open this run in Airflow"
                        >
                          <ExternalLink />
                        </a>
                      ) : null}
                    </td>
                  ) : null}
                </tr>
                {isExpanded ? (
                  <tr className="conf-detail-row">
                    <td colSpan={columnCount}>
                      <pre className="conf-json">{JSON.stringify(run.conf, null, 2)}</pre>
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function StageRunsStrip({
  stages,
  airflowWebUrl,
}: {
  stages: StageRecentRuns[];
  airflowWebUrl: string | null;
}) {
  // Render in the canonical stage order; unknown extras (forward compat) go
  // last — indexOf answers -1 for them, which would sort them FIRST.
  const stageRank = (stage: string): number => {
    const knownIndex = STAGE_ORDER.indexOf(stage);
    return knownIndex === -1 ? STAGE_ORDER.length : knownIndex;
  };
  const orderedStages = [...stages].sort(
    (left, right) => stageRank(left.stage) - stageRank(right.stage),
  );
  return (
    <section className="section">
      <h2 className="section-title">Recent stage runs</h2>
      <p className="stage-strip-note">
        Latest runs per stage sub-DAG — recency only, not correlated with specific master runs
        above.
      </p>
      <div className="stage-strip">
        {orderedStages.map((stageEntry) => (
          <div key={stageEntry.stage} className="stage-cell">
            <div className="stage-cell-header">
              <span className="stage-cell-name">{stageEntry.stage}</span>
              <span className="stage-cell-dag" title={stageEntry.dag_id}>
                {stageEntry.dag_id}
              </span>
            </div>
            {stageEntry.recent.length === 0 ? (
              <p className="empty-note">no runs</p>
            ) : (
              <ul className="stage-run-list">
                {stageEntry.recent.map((stageRun) => {
                  const runUrl = airflowRunUrl(
                    airflowWebUrl,
                    stageEntry.dag_id,
                    stageRun.dag_run_id,
                  );
                  return (
                    <li key={stageRun.dag_run_id} className="stage-run">
                      <RunStateChip state={stageRun.state} />
                      <span className="cell-mono stage-run-time">
                        {stageRun.start_date ? formatTimestamp(stageRun.start_date) : "—"}
                      </span>
                      <span className="cell-num stage-run-duration">
                        {formatDurationBetween(stageRun.start_date, stageRun.end_date)}
                      </span>
                      {runUrl ? (
                        <a
                          className="btn btn-ghost btn-tiny"
                          href={runUrl}
                          target="_blank"
                          rel="noreferrer"
                          title={`Open ${stageRun.dag_run_id} in Airflow`}
                        >
                          <ExternalLink />
                        </a>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function TriggerIngestForm({ config }: { config: WorkspaceConfig | undefined }) {
  const queryClient = useQueryClient();
  const [urisText, setUrisText] = useState("");
  const [profileChoice, setProfileChoice] = useState<string | null>(null);
  const [modeChoice, setModeChoice] = useState<string | null>(null);
  const [batchCountText, setBatchCountText] = useState("");

  // The selects offer the LIVE vocabularies /api/v1/config serves, so
  // hflow.steps stays the one owner of both. Empty only before config loads;
  // until the user picks, the first option holds.
  const profileNames = config?.run_profiles ?? [];
  const ingestModes = config?.ingest_modes ?? [];
  const profile = profileChoice ?? profileNames[0] ?? "full";
  const mode = modeChoice ?? ingestModes[0] ?? "batch";

  const ingestMutation = useMutation({
    mutationFn: triggerIngest,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["runtime-runs"] });
    },
  });

  const uris = urisText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const submitIngest = () => {
    const batchCountParsed = Number.parseInt(batchCountText, 10);
    ingestMutation.mutate({
      uris,
      profile,
      mode,
      ...(Number.isFinite(batchCountParsed) && batchCountParsed > 0
        ? { batch_count: batchCountParsed }
        : {}),
    });
  };

  return (
    <section className="section">
      <h2 className="section-title">Trigger ingest</h2>
      <div className="ingest-form">
        <label className="ingest-field ingest-field-uris">
          <span className="toolbar-field-label">URIs — one per line</span>
          <textarea
            className="input ingest-uris"
            rows={3}
            placeholder={"file:///data/incoming/run_0001.mcap\nfile:///data/incoming/run_0002.mcap"}
            value={urisText}
            onChange={(event) => setUrisText(event.target.value)}
          />
        </label>
        <div className="ingest-controls">
          <label className="toolbar-field">
            <span className="toolbar-field-label">Profile</span>
            <select
              className="input"
              value={profile}
              onChange={(event) => setProfileChoice(event.target.value)}
            >
              {profileNames.map((profileName) => (
                <option key={profileName} value={profileName}>
                  {profileName}
                </option>
              ))}
            </select>
          </label>
          <label className="toolbar-field">
            <span className="toolbar-field-label">Mode</span>
            <select
              className="input"
              value={mode}
              onChange={(event) => setModeChoice(event.target.value)}
            >
              {ingestModes.map((modeName) => (
                <option key={modeName} value={modeName}>
                  {modeName}
                </option>
              ))}
            </select>
          </label>
          <label className="toolbar-field">
            <span className="toolbar-field-label">Batch count</span>
            <input
              className="input ingest-batch-count"
              type="number"
              min={1}
              placeholder="auto"
              value={batchCountText}
              onChange={(event) => setBatchCountText(event.target.value)}
            />
          </label>
          <button
            type="button"
            className="btn btn-primary"
            disabled={uris.length === 0 || ingestMutation.isPending}
            onClick={submitIngest}
          >
            <Play />
            <span>{ingestMutation.isPending ? "Triggering…" : "Trigger run"}</span>
          </button>
        </div>
        {ingestMutation.isError ? (
          <p className="form-error">{describeApiError(ingestMutation.error)}</p>
        ) : null}
        {ingestMutation.data ? (
          <p className="ingest-success" role="status">
            Triggered <code className="cell-mono">{ingestMutation.data.dag_run_id}</code> —{" "}
            <RunStateChip state={ingestMutation.data.state} />
          </p>
        ) : null}
      </div>
    </section>
  );
}

export function RunsPage() {
  const [isAutoRefreshOn, setIsAutoRefreshOn] = useState(false);
  const [selectedRunIdChoice, setSelectedRunIdChoice] = useState<string | null>(null);

  useEffect(() => {
    document.title = "Runs · HFlow";
    return () => {
      document.title = "HFlow";
    };
  }, []);

  const configQuery = useQuery({ queryKey: ["config"], queryFn: fetchWorkspaceConfig });
  // Until the config arrives (or if it fails), keep write affordances hidden.
  const readOnly = configQuery.data ? configQuery.data.read_only : true;

  const statusQuery = useQuery({
    queryKey: ["runtime-status"],
    queryFn: fetchRuntimeStatus,
    refetchInterval: isAutoRefreshOn ? AUTO_REFRESH_INTERVAL_MS : false,
  });
  const runtimeStatus = statusQuery.data;

  const runsQuery = useQuery({
    queryKey: ["runtime-runs", RUNS_PAGE_LIMIT],
    queryFn: () => fetchRuntimeRuns(RUNS_PAGE_LIMIT),
    enabled: runtimeStatus?.available === true,
    refetchInterval: isAutoRefreshOn ? AUTO_REFRESH_INTERVAL_MS : false,
  });

  // Without an explicit pick, the graph follows the newest run.
  const selectedRunId = selectedRunIdChoice ?? runsQuery.data?.runs[0]?.dag_run_id ?? null;

  // The topology is the same payload the Pipeline page draws (shared cache);
  // the run graph only adds live state over it.
  const topologyQuery = useQuery({
    queryKey: ["pipeline-graph"],
    queryFn: fetchPipelineGraph,
    enabled: runtimeStatus?.available === true,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && (error.status === 404 || error.status === 409)) &&
      failureCount < 1,
  });

  const runGraphQuery = useQuery({
    queryKey: ["run-graph", selectedRunId],
    queryFn: () => fetchRunGraph(selectedRunId ?? ""),
    enabled: selectedRunId !== null && runtimeStatus?.available === true,
    refetchInterval: isAutoRefreshOn ? AUTO_REFRESH_INTERVAL_MS : false,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && (error.status === 404 || error.status === 409)) &&
      failureCount < 1,
  });

  const refreshNow = () => {
    void statusQuery.refetch();
    if (runtimeStatus?.available) {
      void runsQuery.refetch();
      void runGraphQuery.refetch();
    }
  };

  const isRefreshing =
    (statusQuery.isFetching && !statusQuery.isPending) ||
    (runsQuery.isFetching && !runsQuery.isPending) ||
    (runGraphQuery.isFetching && !runGraphQuery.isPending);

  return (
    <div className="runs-page">
      <div className="page-title-row">
        <h1 className="page-title">Runs</h1>
        {isRefreshing ? <span className="refresh-note">updating…</span> : null}
        <div className="toolbar-spacer" />
        <label className="toolbar-field auto-refresh-toggle">
          <input
            type="checkbox"
            checked={isAutoRefreshOn}
            onChange={(event) => setIsAutoRefreshOn(event.target.checked)}
          />
          <span className="toolbar-field-label">auto-refresh 10s</span>
        </label>
        <button type="button" className="btn" onClick={refreshNow} title="Refetch status and runs">
          <RefreshCw />
          <span>Refresh</span>
        </button>
        {runtimeStatus?.airflow_web_url ? (
          <a
            className="btn"
            href={runtimeStatus.airflow_web_url}
            target="_blank"
            rel="noreferrer"
            title="Open the Airflow web UI"
          >
            <ExternalLink />
            <span>Open in Airflow</span>
          </a>
        ) : null}
      </div>

      {statusQuery.isPending ? (
        <LoadingPanel label="Checking the runtime…" />
      ) : statusQuery.isError ? (
        <ErrorPanel error={statusQuery.error} onRetry={refreshNow} />
      ) : !runtimeStatus?.available ? (
        <EmptyPanel title="Runtime not available." hint={runtimeStatus?.detail ?? undefined}>
          <p className="state-detail">
            Start the local stack with <code>hflow up</code>, then refresh.
          </p>
        </EmptyPanel>
      ) : (
        <>
          <StatusTiles status={runtimeStatus} />

          {runGraphQuery.isPending && selectedRunId !== null ? (
            <LoadingPanel label="Loading the run graph…" />
          ) : runGraphQuery.data && topologyQuery.data ? (
            <RunGraphSection
              // Keyed by run: picking another run starts with a clean
              // selection instead of pointing at the previous run's task.
              key={runGraphQuery.data.master.dag_run_id}
              runGraph={runGraphQuery.data}
              graph={topologyQuery.data}
              airflowWebUrl={runtimeStatus.airflow_web_url ?? null}
              isAutoRefreshOn={isAutoRefreshOn}
            />
          ) : runGraphQuery.isError || topologyQuery.isError ? (
            <section className="section">
              <h2 className="section-title">Run graph</h2>
              <p className="empty-note">
                {describeApiError(runGraphQuery.error ?? topologyQuery.error)} — the runs table
                below still works.
              </p>
            </section>
          ) : null}

          {readOnly ? null : <TriggerIngestForm config={configQuery.data} />}

          <section className="section">
            <h2 className="section-title">Master runs</h2>
            {runsQuery.isPending ? (
              <LoadingPanel label="Loading runs…" />
            ) : runsQuery.isError ? (
              <ErrorPanel
                error={runsQuery.error}
                onRetry={() => {
                  void runsQuery.refetch();
                }}
              />
            ) : (
              <MasterRunsTable
                runs={runsQuery.data.runs}
                airflowWebUrl={runtimeStatus.airflow_web_url ?? null}
                dagId={runtimeStatus.dag_id ?? null}
                selectedRunId={selectedRunId}
                onSelectRun={setSelectedRunIdChoice}
              />
            )}
          </section>

          {runsQuery.data?.stages && runsQuery.data.stages.length > 0 ? (
            <StageRunsStrip
              stages={runsQuery.data.stages}
              airflowWebUrl={runtimeStatus.airflow_web_url ?? null}
            />
          ) : null}
        </>
      )}
    </div>
  );
}
