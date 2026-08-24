// The only place this app talks to the server. Every payload type is an alias
// into src/apiSchema.ts, which `pnpm gen:api` regenerates from the server's own
// OpenAPI declaration -- so nothing here hand-copies a field name, and a
// contract change surfaces as a TypeScript error rather than as an undefined at
// runtime.

import { useQuery } from "@tanstack/react-query";
import type { components } from "./apiSchema";

type Schemas = components["schemas"];

export type Stage = Schemas["Stage"];
export type DagTaskNode = Schemas["DagTaskNodePayload"];
export type EpisodeCheckRun = Schemas["EpisodeCheckRunRecord"];
export type EpisodeDossier = Schemas["EpisodeDossierResponse"];
export type EpisodePage = Schemas["EpisodePageResponse"];
export type PipelineEngineStep = Schemas["PipelineEngineStep"];
export type PipelineGate = Schemas["PipelineGate"];
export type PipelineGraph = Schemas["PipelineGraphResponse"];
export type PipelineGraphStage = Schemas["PipelineGraphStage"];
export type PipelineUserStep = Schemas["PipelineUserStep"];
export type QuarantineGate = Schemas["QuarantineGate"];
export type RunGraph = Schemas["RunGraphResponse"];
export type RunGraphStage = Schemas["RunGraphStage"];
export type RunTaskInstance = Schemas["RunTaskInstance"];
export type RuntimeRunSummary = Schemas["RuntimeRunSummary"];
export type RuntimeRuns = Schemas["RuntimeRunsResponse"];
export type RuntimeStatus = Schemas["RuntimeStatusResponse"];
export type WorkspaceConfig = Schemas["WorkspaceConfigResponse"];

/** A refused request, carrying the server's own detail string. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

/** FastAPI answers a refusal with `detail`, either a string or a validation list. */
function refusalDetail(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const { detail } = body as { detail: unknown };
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((entry) => JSON.stringify(entry)).join("; ");
  }
  return `request failed with status ${status}`;
}

type QueryValue = string | number | readonly string[] | undefined;

async function getJson<T>(path: string, query: Record<string, QueryValue> = {}): Promise<T> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined) continue;
    // An array becomes the same key repeated, which is how FastAPI's
    // `list[str] | None = Query()` filters read a multi-value filter.
    if (Array.isArray(value)) for (const entry of value) search.append(key, entry);
    else search.set(key, String(value));
  }
  const suffix = search.size > 0 ? `?${search}` : "";
  // Relative on purpose: the server serves this bundle and the API from the
  // same origin, and Vite's dev proxy forwards /api to it.
  const response = await fetch(`/api/v1${path}${suffix}`, {
    headers: { accept: "application/json" },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(response.status, refusalDetail(body, response.status));
  }
  return (await response.json()) as T;
}

// Airflow states that mean "this run is finished". Anything else -- running,
// queued, a state a newer Airflow invented -- keeps the poll alive, so an
// unrecognized state errs toward refreshing rather than toward going stale.
const TERMINAL_RUN_STATES = new Set(["success", "failed", "skipped", "upstream_failed"]);

export function isTerminalRunState(state: string | null | undefined): boolean {
  return state !== null && state !== undefined && TERMINAL_RUN_STATES.has(state.toLowerCase());
}

const LIVE_POLL_MS = 4000;

export function useWorkspaceConfig() {
  return useQuery({
    queryKey: ["config"],
    queryFn: () => getJson<WorkspaceConfig>("/config"),
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useRuntimeStatus() {
  return useQuery({
    queryKey: ["runtime", "status"],
    queryFn: () => getJson<RuntimeStatus>("/runtime/status"),
    refetchInterval: LIVE_POLL_MS,
  });
}

/** The master runs the canvas can be pointed at, newest first. */
export function useRuntimeRuns(enabled: boolean) {
  return useQuery({
    queryKey: ["runtime", "runs"],
    queryFn: () => getJson<RuntimeRuns>("/runtime/runs", { limit: 25 }),
    enabled,
    refetchInterval: LIVE_POLL_MS,
  });
}

/** The topology: what the DAGs and the pipeline's steps ARE, run or no run. */
export function usePipelineGraph() {
  return useQuery({
    queryKey: ["pipeline", "graph"],
    queryFn: () => getJson<PipelineGraph>("/pipeline/graph"),
    // The bundle is re-rendered by `hflow up`, so the shape can change under a
    // long-lived tab -- just far less often than a run's state does.
    staleTime: 60_000,
  });
}

/** One master run's live state over that topology. */
export function useRunGraph(dagRunId: string | null) {
  return useQuery({
    queryKey: ["runtime", "runs", dagRunId, "graph"],
    queryFn: () => getJson<RunGraph>(`/runtime/runs/${encodeURIComponent(dagRunId ?? "")}/graph`),
    enabled: dagRunId !== null,
    // Stop polling once the master run is finished: its stages are finished
    // too, so there is nothing left to refresh.
    refetchInterval: (query) =>
      isTerminalRunState(query.state.data?.master.state) ? false : LIVE_POLL_MS,
  });
}

/**
 * The episodes one ingest run produced, asked for by ALL of its stage run ids.
 *
 * This is the join the canvas drills through: every stage's `process_batch`
 * stamps its own Airflow run id onto the catalog rows it appends
 * (`episodes.orchestrator_run_id`).
 *
 * Every stage run id at once, not one: the catalog's `episodes` view is one row
 * per episode -- the most recent append wins -- so in a full ingest the media
 * stage's rows supersede sync's, meta's and labels'. Asking with a single
 * stage's id therefore answers 0 for every stage but the last one to record,
 * which was measured, not assumed. The union is the honest question: which
 * episodes' current catalog row came out of this run.
 */
export function useRunEpisodes(orchestratorRunIds: readonly string[], limit = 200) {
  // Sorted so the cache key does not depend on the order the stages arrived in.
  const runIds = [...orchestratorRunIds].sort();
  return useQuery({
    queryKey: ["episodes", "byRun", runIds, limit],
    queryFn: () => getJson<EpisodePage>("/episodes", { orchestrator_run_id: runIds, limit }),
    enabled: runIds.length > 0,
  });
}

export function useEpisodeDossier(episodeId: string | null) {
  return useQuery({
    queryKey: ["episodes", episodeId],
    queryFn: () => getJson<EpisodeDossier>(`/episodes/${encodeURIComponent(episodeId ?? "")}`),
    enabled: episodeId !== null,
  });
}
