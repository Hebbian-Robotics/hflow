// Typed client for the hflow-server JSON API (/api/v1).
// Interfaces mirror the served contract shapes exactly; this module is
// the only place that talks to the network.

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export interface WorkspaceCapabilities {
  catalog: boolean;
  media: boolean;
  runtime: boolean;
  /** True when the server imported a --pipeline app at startup. */
  pipeline: boolean;
}

export interface WorkspaceConfig {
  mode: string;
  read_only: boolean;
  hflow_version: string;
  hflow_ui_version: string;
  data_root: string;
  workspace_id: string | null;
  capabilities: WorkspaceCapabilities;
  // These two carry the live vocabularies from hflow.steps so the frontend
  // never hardcodes them. Required, not optional: this bundle ships inside the
  // same wheel as the server that serves it, so the two cannot be different
  // versions and "an older server omits the field" is not a reachable state.
  run_profiles: string[];
  ingest_modes: string[];
}

export interface EpisodeColumn {
  name: string;
  type: string;
}

/** One row of the wide `episodes` view; columns are described by EpisodeColumn. */
export type EpisodeRow = Record<string, unknown>;

export interface EpisodesResponse {
  rows: EpisodeRow[];
  total: number;
  columns: EpisodeColumn[];
  /** The SELECT the server compiled for exactly these filters (with LIMIT/OFFSET). */
  sql: string;
}

export interface FacetEntry {
  value: string;
  count: number;
}

export type FacetName = "task" | "operator" | "embodiment" | "status" | "pipeline_version";

export type EpisodeFacets = Record<FacetName, FacetEntry[]>;

export type EpisodeStatus = "ok" | "quarantined";

export interface EpisodeMeasurement {
  key: string;
  value_double: number | null;
  value_text: string | null;
  value_bool: boolean | null;
  check_name: string;
  check_version: string;
  recorded_at: string;
}

export interface EpisodeCheckRun {
  check_name: string;
  check_version: string;
  critical: boolean;
  status: string;
  duration_s: number | null;
  error: string | null;
  recorded_at: string;
  run_fingerprint: string;
}

export interface EpisodeInterval {
  label: string;
  start_ns: number;
  end_ns: number;
  check_name: string;
  /** Joined from check_runs (LEFT JOIN — the intervals table carries no
   * version), so it is null when no matching check_runs row exists. */
  check_version: string | null;
}

export interface EpisodeTagRecord {
  tag: string;
  check_name: string;
  recorded_at: string;
}

export interface EpisodeMediaItem {
  name: string;
  uri: string;
  /** Same-origin byte-serving URL, or null when the artifact is not servable. */
  url: string | null;
}

export interface EpisodeDossier {
  episode: EpisodeRow & { status: EpisodeStatus; quarantine_tags: string[] };
  measurements: EpisodeMeasurement[];
  check_runs: EpisodeCheckRun[];
  intervals: EpisodeInterval[];
  tags: EpisodeTagRecord[];
  history: EpisodeRow[];
  media: EpisodeMediaItem[];
  canonical_url: string | null;
}

/** The structured filter params /episodes and /episodes/stats share. */
export interface EpisodesFilter {
  task: string[];
  operator: string[];
  embodiment: string[];
  status: EpisodeStatus | null;
  success: "true" | "false" | null;
  search: string;
}

export interface EpisodesQuery extends EpisodesFilter {
  orderBy: string | null;
  order: "asc" | "desc";
  limit: number;
  offset: number;
}

export const DEFAULT_PAGE_SIZE = 50;
export const MAX_PAGE_SIZE = 500;
/** The server's default sort column; mirrored so the header indicator is honest. */
export const DEFAULT_ORDER_BY = "recorded_at";

// --- fetch helpers ------------------------------------------------------------
// The server carries no authentication of its own (see docs/UI.md, "Trust
// posture"), so every call here is a plain same-origin request: no credential
// is captured from the landing URL, stored, or attached. Server-side paths
// (media, canonical downloads, pinned Parquet) are used verbatim in
// <img src> / <a href>.

type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

async function requestJson<ResponseBody>(
  path: string,
  method: HttpMethod,
  requestBody?: unknown,
): Promise<ResponseBody> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (requestBody !== undefined) headers["Content-Type"] = "application/json";
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: requestBody === undefined ? undefined : JSON.stringify(requestBody),
    });
  } catch (networkError) {
    const reason = networkError instanceof Error ? networkError.message : "network error";
    throw new ApiError(0, reason);
  }
  if (!response.ok) {
    let detail = `request failed with status ${response.status}`;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null) {
        const detailValue = (body as { detail?: unknown }).detail;
        if (typeof detailValue === "string") detail = detailValue;
      }
    } catch {
      // Non-JSON error body; keep the status-based detail.
    }
    throw new ApiError(response.status, detail);
  }
  // DELETE endpoints answer 204 with no body.
  if (response.status === 204) return undefined as unknown as ResponseBody;
  return (await response.json()) as ResponseBody;
}

async function fetchJson<ResponseBody>(path: string): Promise<ResponseBody> {
  return requestJson<ResponseBody>(path, "GET");
}

/** Only the filter params (no sort/pagination) — shared with /episodes/stats,
 * whose distributions must reflect the same filtered set regardless of paging. */
function buildEpisodesFilterParams(filter: EpisodesFilter): URLSearchParams {
  const params = new URLSearchParams();
  for (const value of filter.task) params.append("task", value);
  for (const value of filter.operator) params.append("operator", value);
  for (const value of filter.embodiment) params.append("embodiment", value);
  if (filter.status) params.set("status", filter.status);
  if (filter.success) params.set("success", filter.success);
  if (filter.search) params.set("search", filter.search);
  return params;
}

function buildEpisodesSearchParams(query: EpisodesQuery): URLSearchParams {
  const params = buildEpisodesFilterParams(query);
  if (query.orderBy) params.set("order_by", query.orderBy);
  params.set("order", query.order);
  params.set("limit", String(query.limit));
  params.set("offset", String(query.offset));
  return params;
}

/**
 * Parse browser URL search params (which intentionally use the same names as
 * the API: task, operator, embodiment, status, success, search, order_by,
 * order, limit, offset) into a typed query.
 */
export function parseEpisodesQuery(params: URLSearchParams): EpisodesQuery {
  const statusParam = params.get("status");
  const successParam = params.get("success");
  const orderParam = params.get("order");
  const limitParsed = Number.parseInt(params.get("limit") ?? "", 10);
  const offsetParsed = Number.parseInt(params.get("offset") ?? "", 10);
  const limit = Number.isFinite(limitParsed)
    ? Math.min(Math.max(limitParsed, 1), MAX_PAGE_SIZE)
    : DEFAULT_PAGE_SIZE;
  const offset = Number.isFinite(offsetParsed) ? Math.max(offsetParsed, 0) : 0;
  return {
    task: params.getAll("task"),
    operator: params.getAll("operator"),
    embodiment: params.getAll("embodiment"),
    status: statusParam === "ok" || statusParam === "quarantined" ? statusParam : null,
    success: successParam === "true" || successParam === "false" ? successParam : null,
    search: params.get("search") ?? "",
    orderBy: params.get("order_by"),
    order: orderParam === "asc" ? "asc" : "desc",
    limit,
    offset,
  };
}

// --- API calls ----------------------------------------------------------------

export function fetchWorkspaceConfig(): Promise<WorkspaceConfig> {
  return fetchJson<WorkspaceConfig>("/api/v1/config");
}

export function fetchEpisodesPage(query: EpisodesQuery): Promise<EpisodesResponse> {
  return fetchJson<EpisodesResponse>(`/api/v1/episodes?${buildEpisodesSearchParams(query)}`);
}

export function fetchEpisodeFacets(): Promise<EpisodeFacets> {
  return fetchJson<EpisodeFacets>("/api/v1/episodes/facets");
}

export function fetchEpisodeDossier(episodeId: string): Promise<EpisodeDossier> {
  return fetchJson<EpisodeDossier>(`/api/v1/episodes/${encodeURIComponent(episodeId)}`);
}

// --- curation studio --------------------------------------------------------

/** {name, type} column descriptor, as returned by the server's DESCRIBE. */
export type ColumnDescriptor = EpisodeColumn;

/** One generic result row; the server guarantees JSON-safe values. */
export type ResultRow = Record<string, unknown>;

/**
 * One row of a DuckDB `SUMMARIZE` result (column_name, column_type, min, max,
 * approx_unique, null_percentage, …). The exact key set varies with the DuckDB
 * version, so it stays an open record and the UI renders whatever arrives.
 */
export type SummarizeRow = Record<string, unknown>;

export interface CurationPreviewRequest {
  sql: string;
  /** Server default 100, max 1000. */
  limit?: number;
  /** When true the server also runs SUMMARIZE over the result. */
  stats?: boolean;
}

export interface CurationPreviewResult {
  columns: ColumnDescriptor[];
  rows: ResultRow[];
  /** Full count over the user's SELECT, independent of the preview limit. */
  row_count: number;
  truncated: boolean;
  column_stats: SummarizeRow[] | null;
  /** The wrapped SELECT the server actually ran. */
  sql: string;
}

export interface CoverageEntry {
  check_name: string;
  episodes_ran: number;
  total_episodes: number;
  fraction: number;
}

export interface CurationReport {
  row_count: number;
  total_episodes: number;
  coverage: CoverageEntry[];
}

export interface PinManifestRequest {
  sql: string;
  name: string;
  description?: string;
}

export interface ManifestRegistryEntry {
  id: string;
  name: string;
  description: string | null;
  sql: string;
  /** Data-root-relative path of the pinned Parquet file. */
  manifest_path: string;
  row_count: number;
  total_episodes: number;
  coverage: CoverageEntry[];
  created_at: string;
}

export interface SavedQuery {
  id: string;
  name: string;
  sql: string;
  updated_at: string;
}

export interface CatalogTable {
  name: string;
  kind: "view" | "table";
  columns: ColumnDescriptor[];
}

export interface CatalogTableSummary {
  row_count: number;
  columns: SummarizeRow[];
}

export function runCurationPreview(
  request: CurationPreviewRequest,
): Promise<CurationPreviewResult> {
  return requestJson<CurationPreviewResult>("/api/v1/curation/preview", "POST", request);
}

export function runCurationReport(sql: string): Promise<CurationReport> {
  return requestJson<CurationReport>("/api/v1/curation/report", "POST", { sql });
}

export function pinManifest(request: PinManifestRequest): Promise<ManifestRegistryEntry> {
  return requestJson<ManifestRegistryEntry>("/api/v1/curation/pin", "POST", request);
}

export function fetchManifests(): Promise<ManifestRegistryEntry[]> {
  return fetchJson<{ manifests: ManifestRegistryEntry[] }>("/api/v1/manifests").then(
    (body) => body.manifests,
  );
}

/** Same-origin Parquet download path, ready to use as an href. */
export function manifestDownloadPath(manifestId: string): string {
  return `/api/v1/manifests/${encodeURIComponent(manifestId)}/download`;
}

export function fetchSavedQueries(): Promise<SavedQuery[]> {
  return fetchJson<{ queries: SavedQuery[] }>("/api/v1/queries").then((body) => body.queries);
}

export function createSavedQuery(input: { name: string; sql: string }): Promise<SavedQuery> {
  return requestJson<SavedQuery>("/api/v1/queries", "POST", input);
}

export function updateSavedQuery(
  queryId: string,
  changes: { name?: string; sql?: string },
): Promise<SavedQuery> {
  return requestJson<SavedQuery>(`/api/v1/queries/${encodeURIComponent(queryId)}`, "PUT", changes);
}

export function deleteSavedQuery(queryId: string): Promise<void> {
  return requestJson<void>(`/api/v1/queries/${encodeURIComponent(queryId)}`, "DELETE");
}

export function fetchCatalogTables(): Promise<CatalogTable[]> {
  return fetchJson<{ tables: CatalogTable[] }>("/api/v1/catalog/tables").then(
    (body) => body.tables,
  );
}

export function fetchCatalogTableSummary(tableName: string): Promise<CatalogTableSummary> {
  return fetchJson<CatalogTableSummary>(
    `/api/v1/catalog/tables/${encodeURIComponent(tableName)}/summary`,
  );
}

// --- runs monitor -------------------------------------------------------------

/** Per-component health from Airflow's monitor endpoint; null = component absent
 * (triggerer/dag_processor may legitimately be missing in minimal deployments). */
export interface RuntimeHealth {
  metadatabase: string | null;
  scheduler: string | null;
  triggerer: string | null;
  dag_processor: string | null;
}

export interface RuntimeStatus {
  available: boolean;
  /** Why the runtime is unavailable (only meaningful when available is false). */
  detail: string | null;
  source: "bundle" | "remote" | null;
  /** Deep-link base for the Airflow web UI, when the server knows one. */
  airflow_web_url: string | null;
  dag_id: string | null;
  registered: boolean | null;
  health: RuntimeHealth | null;
}

export interface RuntimeRun {
  dag_run_id: string;
  state: string;
  logical_date: string | null;
  start_date: string | null;
  end_date: string | null;
  conf: Record<string, unknown>;
}

/** Canonical hflow.steps.Stage order, for display sorting only — the server
 * may emit stage names this build does not know (forward compat). */
export const STAGE_ORDER: readonly string[] = ["sync", "meta", "labels", "media"];

export interface StageRun {
  dag_run_id: string;
  state: string;
  start_date: string | null;
  end_date: string | null;
}

export interface StageRecentRuns {
  /** A hflow.steps.Stage value; unknown names can appear under version skew. */
  stage: string;
  dag_id: string;
  recent: StageRun[];
}

export interface RuntimeRunsResponse {
  runs: RuntimeRun[];
  /** Recent-per-stage runs (bundle only); NOT correlated with specific master runs. */
  stages: StageRecentRuns[] | null;
}

export interface IngestRequest {
  uris: string[];
  profile: string;
  /** A hflow.steps.IngestMode value; the server validates against the live set. */
  mode: string;
  batch_count?: number;
}

export interface IngestResponse {
  dag_run_id: string;
  state: string;
}

export function fetchRuntimeStatus(): Promise<RuntimeStatus> {
  return fetchJson<RuntimeStatus>("/api/v1/runtime/status");
}

export function fetchRuntimeRuns(limit: number): Promise<RuntimeRunsResponse> {
  return fetchJson<RuntimeRunsResponse>(`/api/v1/runtime/runs?limit=${limit}`);
}

export function triggerIngest(request: IngestRequest): Promise<IngestResponse> {
  return requestJson<IngestResponse>("/api/v1/runtime/ingest", "POST", request);
}

// --- pipeline page --------------------------------------------------------------

/** One registered step out of App.manifest() (hflow.manifest.StepManifest). */
export interface PipelineStepManifest {
  name: string;
  kind: string;
  /** Content hash of the live function — long; display truncated with copy. */
  version: string;
  critical: boolean;
  requires: string[];
  /** Endpoint alias the step declares, or null. */
  uses: string | null;
}

export interface DerivedChannelManifest {
  topic: string;
  version: string;
}

export interface PipelineManifest {
  manifest_version: number;
  pipeline_name: string;
  hflow_version: string;
  schema_version: string;
  pipeline_version: string;
  checks: PipelineStepManifest[];
  enrichments: PipelineStepManifest[];
  derived_channels: DerivedChannelManifest[];
  endpoint_aliases: string[];
  has_transform_override: boolean;
}

export interface ObservedVersion {
  check_name: string;
  check_version: string;
  first_seen: string;
  last_seen: string;
  run_count: number;
}

export interface StaleSummary {
  pipeline_version: string;
  count: number;
}

export interface PipelineResponse {
  manifest: PipelineManifest;
  observed: ObservedVersion[];
  stale: StaleSummary | null;
}

/** 409 (ApiError with the server's detail) when no --pipeline is configured. */
export function fetchPipeline(): Promise<PipelineResponse> {
  return fetchJson<PipelineResponse>("/api/v1/pipeline");
}

// --- episode column distributions --------------------------------------------------

export interface StatsBucket {
  lo: number;
  hi: number;
  count: number;
}

export interface StatsValue {
  value: string;
  count: number;
}

export interface NumericColumnStats {
  name: string;
  kind: "numeric";
  /** Bucket bounds carry the range; top-level min/max are optional extras. */
  min?: number | null;
  max?: number | null;
  buckets: StatsBucket[];
}

export interface CategoricalColumnStats {
  name: string;
  kind: "categorical";
  /** Top values with counts (server-capped). */
  values: StatsValue[];
  /** Rows beyond the cap, rolled up; 0 or absent when the cap covered everything. */
  other_count?: number | null;
}

export type EpisodeColumnStats = NumericColumnStats | CategoricalColumnStats;

export interface EpisodesStatsResponse {
  columns: EpisodeColumnStats[];
}

/** Distributions over the SAME filtered set as /episodes (sort/paging excluded). */
export function fetchEpisodeStats(filter: EpisodesFilter): Promise<EpisodesStatsResponse> {
  const params = buildEpisodesFilterParams(filter);
  const suffix = params.size > 0 ? `?${params}` : "";
  return fetchJson<EpisodesStatsResponse>(`/api/v1/episodes/stats${suffix}`);
}

// --- Visualization wave: DAG topology, live run graph, episode timeline ----------

/** One task of a generated DAG (mirrors hflow.runtime.DagTaskNode). */
export interface DagTaskNode {
  task_id: string;
  summary: string;
  /** Dynamically mapped: one instance per planned batch — draw it as a fan-out. */
  mapped: boolean;
  /** Defers instead of holding a worker slot — say "waiting", never "stalled". */
  deferred: boolean;
}

export interface DagTopology {
  dag_id: string;
  tasks: DagTaskNode[];
  /** [upstream, downstream] task-id pairs, declaration order. */
  edges: [string, string][];
}

/** Engine work inside one stage that the manifest's step lists do not carry. */
export interface PipelineEngineStep {
  name: string;
  summary: string;
}

/**
 * One registered step, as the graph endpoint serves it. `tier` mirrors
 * App._ordered_checks: tier 2 iff the step declares requires or uses, so it
 * runs after the cheap ones. Steps within a tier have NO ordering.
 */
export interface PipelineUserStep {
  name: string;
  kind: string;
  version: string;
  critical: boolean;
  uses: string | null;
  requires: string[];
  tier: 1 | 2;
}

export interface PipelineGraphStage {
  /** A hflow.steps.Stage value; unknown names can appear under version skew. */
  stage: string;
  title: string;
  description: string;
  gate_task_id: string;
  trigger_task_id: string;
  enabling_profiles: string[];
  dag: DagTopology;
  engine_steps: PipelineEngineStep[];
  user_steps: PipelineUserStep[];
}

/** The one real cross-step edge: meta's critical checks gate later enrichment. */
export interface QuarantineGate {
  from_stage: string;
  to_stages: string[];
  critical_step_names: string[];
  explanation: string;
}

export interface PipelineGraphResponse {
  /** False when no runtime is addressed: the master id is display-only. */
  dag_ids_known: boolean;
  /** False without --pipeline: the steps inside process_batch are unknown. */
  steps_known: boolean;
  master: DagTopology;
  stages: PipelineGraphStage[];
  /** Null exactly when steps_known is false. */
  quarantine_gate: QuarantineGate | null;
}

export function fetchPipelineGraph(): Promise<PipelineGraphResponse> {
  return fetchJson<PipelineGraphResponse>("/api/v1/pipeline/graph");
}

/** One Airflow task instance; map_index -1 means the task is not mapped. */
export interface RunTaskInstance {
  task_id: string;
  state: string | null;
  start_date: string | null;
  end_date: string | null;
  /** When the scheduler queued the task, so the replay can tell "waiting for a
   * worker" from "running". Airflow may omit it, and servers that predate the
   * field leave the key absent entirely — always treat it as optional. */
  queued_at: string | null;
  try_number: number | null;
  map_index: number;
  duration_s: number | null;
}

export interface RunGraphMaster {
  dag_run_id: string;
  state: string | null;
  tasks: RunTaskInstance[];
}

export interface MappedFanOutSummary {
  task_id: string;
  total: number;
  by_state: Record<string, number>;
}

export interface RunGraphStage {
  stage: string;
  dag_id: string;
  /** Null when this stage never ran for this master run. */
  dag_run_id: string | null;
  state: string | null;
  /** "heuristic": matched by start time, not by a stored parent-run link. */
  match: "heuristic" | null;
  tasks: RunTaskInstance[];
  mapped_summary: MappedFanOutSummary | null;
}

export interface RunGraphResponse {
  master: RunGraphMaster;
  stages: RunGraphStage[];
}

export function fetchRunGraph(dagRunId: string): Promise<RunGraphResponse> {
  return fetchJson<RunGraphResponse>(`/api/v1/runtime/runs/${encodeURIComponent(dagRunId)}/graph`);
}

/** One recorded interval, with seconds relative to the episode span. */
export interface TimelineInterval {
  label: string;
  start_ns: number;
  end_ns: number;
  start_s: number;
  end_s: number;
  check_name: string;
  /** Label prefix ("gap", "joint_discontinuity", …) — the colour grouping. */
  kind: string;
}

/** A numeric measurement, ready to draw as a bar. */
export interface TimelineMeasurement {
  key: string;
  value: number;
  unit: string | null;
}

export interface EpisodeTimeline {
  start_ns: number | null;
  end_ns: number | null;
  /** Null when nothing in the episode gives a span — the UI must say so. */
  duration_s: number | null;
  intervals: TimelineInterval[];
  measurements: TimelineMeasurement[];
}

export function fetchEpisodeTimeline(episodeId: string): Promise<EpisodeTimeline> {
  return fetchJson<EpisodeTimeline>(`/api/v1/episodes/${encodeURIComponent(episodeId)}/timeline`);
}

/** Human-readable message for any error a query can surface. */
export function describeApiError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 0) {
      return `Could not reach the hflow server (${error.detail}). Is \`hflow serve\` running?`;
    }
    return error.detail;
  }
  if (error instanceof Error) return error.message;
  return "Unknown error";
}
