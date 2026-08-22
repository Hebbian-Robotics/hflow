import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useEffect, useState } from "react";
import {
  describeApiError,
  fetchRuntimeRuns,
  fetchRuntimeStatus,
  fetchWorkspaceConfig,
  INGEST_MODES,
  RUN_PROFILE_NAMES,
  type RuntimeHealth,
  type RuntimeRun,
  type RuntimeStatus,
  STAGE_ORDER,
  type StageRecentRuns,
  triggerIngest,
  type WorkspaceConfig,
} from "../api";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "../components/QueryStates";
import { formatDurationBetween, formatTimestamp } from "../format";
import { ChevronDownIcon, ExternalLinkIcon, PlayIcon, RefreshIcon } from "../icons";

const RUNS_PAGE_LIMIT = 25;
const AUTO_REFRESH_INTERVAL_MS = 10_000;

// The runs monitor renders exactly what the server proxied from Airflow —
// the browser never sees Airflow credentials, only these JSON summaries.

function runStateChipClass(state: string): string {
  switch (state.toLowerCase()) {
    case "success":
      return "chip chip-ok";
    case "failed":
    case "error":
      return "chip chip-err";
    case "running":
      return "chip chip-accent";
    case "up_for_retry":
    case "up_for_reschedule":
    case "restarting":
      return "chip chip-warn";
    default:
      return "chip chip-muted";
  }
}

function RunStateChip({ state }: { state: string }) {
  return <span className={runStateChipClass(state)}>{state}</span>;
}

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
  dagRunId: string,
): string | null {
  if (!webUrlBase || !dagId) return null;
  const base = webUrlBase.replace(/\/$/, "");
  return `${base}/dags/${encodeURIComponent(dagId)}/runs/${encodeURIComponent(dagRunId)}`;
}

function MasterRunsTable({
  runs,
  airflowWebUrl,
  dagId,
}: {
  runs: RuntimeRun[];
  airflowWebUrl: string | null;
  dagId: string | null;
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
          {runs.map((run) => {
            const isExpanded = expandedRunIds.has(run.dag_run_id);
            const runUrl = airflowRunUrl(airflowWebUrl, dagId, run.dag_run_id);
            return (
              <Fragment key={run.dag_run_id}>
                <tr>
                  <td>
                    <button
                      type="button"
                      className="btn btn-ghost btn-tiny"
                      onClick={() => toggleExpanded(run.dag_run_id)}
                      aria-expanded={isExpanded}
                      title={isExpanded ? "Hide the full conf" : "Show the full conf"}
                    >
                      <ChevronDownIcon className={isExpanded ? "chevron is-open" : "chevron"} />
                    </button>
                  </td>
                  <td>
                    <RunStateChip state={run.state} />
                  </td>
                  <td className="cell-mono" title={run.dag_run_id}>
                    {run.dag_run_id}
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
                          <ExternalLinkIcon />
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
                          <ExternalLinkIcon />
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

  // The selects offer the LIVE vocabularies /api/v1/config serves (hflow.steps
  // stays the one owner); the constants are only the fallback for servers that
  // predate those config fields. Until the user picks, the first option holds.
  const profileNames = config?.run_profiles ?? RUN_PROFILE_NAMES;
  const ingestModes = config?.ingest_modes ?? INGEST_MODES;
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
            <PlayIcon />
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

  const refreshNow = () => {
    void statusQuery.refetch();
    if (runtimeStatus?.available) void runsQuery.refetch();
  };

  const isRefreshing =
    (statusQuery.isFetching && !statusQuery.isPending) ||
    (runsQuery.isFetching && !runsQuery.isPending);

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
          <RefreshIcon />
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
            <ExternalLinkIcon />
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
                airflowWebUrl={runtimeStatus.airflow_web_url}
                dagId={runtimeStatus.dag_id}
              />
            )}
          </section>

          {runsQuery.data?.stages && runsQuery.data.stages.length > 0 ? (
            <StageRunsStrip
              stages={runsQuery.data.stages}
              airflowWebUrl={runtimeStatus.airflow_web_url}
            />
          ) : null}
        </>
      )}
    </div>
  );
}
