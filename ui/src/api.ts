// Typed client for the hflow-server JSON API (/api/v1). The only module here
// that talks to the network.
//
// Payload types are ALIASES of `apiSchema.ts`, which `pnpm gen:api` generates
// from the server's own /api/openapi.json. They were hand-written once and
// drifted: the server declared `kind: StepKind` while the copy said
// `kind: string`, it grew a `curation` capability the copy never learned
// about, and a renamed config field went unnoticed for a release. Nothing
// caught any of it, because nothing compared them. Now there is one owner --
// the server -- and regenerating is a diff rather than an audit.
//
// What stays hand-written below is the part the server does not declare:
// request shapes this client composes, and view-model types (`EpisodesQuery`,
// `FacetName`) that exist only in the browser.

import type { components } from "./apiSchema";

type Served = components["schemas"];

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

export type WorkspaceCapabilities = Served["WorkspaceCapabilities"];

export type WorkspaceConfig = Served["WorkspaceConfigResponse"];

export type EpisodeColumn = Served["ColumnDescriptor"];

/** One row of the wide `episodes` view; columns are described by EpisodeColumn. */
export type EpisodeRow = Record<string, unknown>;

export type EpisodesResponse = Served["EpisodePageResponse"];

export type FacetEntry = Served["ValueCount"];

export type FacetName = "task" | "operator" | "embodiment" | "status" | "pipeline_version";

export type EpisodeFacets = Served["EpisodeFacetsResponse"];

export type EpisodeStatus = "ok" | "quarantined";

export type EpisodeMeasurement = Served["EpisodeMeasurementRecord"];

export type EpisodeCheckRun = Served["EpisodeCheckRunRecord"];

export type EpisodeInterval = Served["EpisodeIntervalRecord"];

export type EpisodeTagRecord = Served["EpisodeTagRecord"];

export type EpisodeMediaItem = Served["EpisodeMediaArtifact"];

export type EpisodeDossier = Served["EpisodeDossierResponse"];

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

export type CurationPreviewResult = Served["CurationPreviewResponse"];

export type CoverageEntry = Served["CheckCoverageEntry"];

export type CurationReport = Served["CurationReportResponse"];

export interface PinManifestRequest {
  sql: string;
  name: string;
  description?: string;
}

export type ManifestRegistryEntry = Served["PinnedManifestEntry"];

export type SavedQuery = Served["SavedQueryEntry"];

export type CatalogTable = Served["CatalogTableDescription"];

export type CatalogTableSummary = Served["CatalogTableSummaryResponse"];

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

export type RuntimeHealth = Served["RuntimeHealthComponents"];

export type RuntimeStatus = Served["RuntimeStatusResponse"];

export type RuntimeRun = Served["RuntimeRunSummary"];

/** Canonical hflow.steps.Stage order, for display sorting only — the server
 * may emit stage names this build does not know (forward compat). */
export const STAGE_ORDER: readonly string[] = ["sync", "meta", "labels", "media"];

export type StageRun = Served["StageRunSummary"];

export type StageRecentRuns = Served["StageRecentRuns"];

export type RuntimeRunsResponse = Served["RuntimeRunsResponse"];

export interface IngestRequest {
  uris: string[];
  profile: string;
  /** A hflow.steps.IngestMode value; the server validates against the live set. */
  mode: string;
  batch_count?: number;
}

export type IngestResponse = Served["IngestTriggerResponse"];

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

export type ObservedVersion = Served["ObservedCheckVersion"];

export type StaleSummary = Served["StaleSummary"];

/**
 * The served pipeline payload, with `manifest` narrowed.
 *
 * This is the one payload the schema cannot type for us: the server declares
 * `manifest` as an open object because `hflow.manifest` owns that shape and it
 * is forwarded verbatim rather than mirrored. Everything else here comes from
 * the generated type; only the hole is filled in, and `PipelineManifest` above
 * is the local restatement that fills it. Typing it server-side would delete
 * this narrowing.
 */
export type PipelineResponse = Omit<Served["PipelineResponse"], "manifest"> & {
  manifest: PipelineManifest;
};

/** 409 (ApiError with the server's detail) when no --pipeline is configured. */
export function fetchPipeline(): Promise<PipelineResponse> {
  return fetchJson<PipelineResponse>("/api/v1/pipeline");
}

// --- episode column distributions --------------------------------------------------

export type StatsBucket = Served["NumericHistogramBucket"];

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

export type EpisodesStatsResponse = Served["EpisodeStatsResponse"];

/** Distributions over the SAME filtered set as /episodes (sort/paging excluded). */
export function fetchEpisodeStats(filter: EpisodesFilter): Promise<EpisodesStatsResponse> {
  const params = buildEpisodesFilterParams(filter);
  const suffix = params.size > 0 ? `?${params}` : "";
  return fetchJson<EpisodesStatsResponse>(`/api/v1/episodes/stats${suffix}`);
}

// --- Visualization wave: DAG topology, live run graph, episode timeline ----------

export type DagTaskNode = Served["DagTaskNodePayload"];

export type DagTopology = Served["DagTopologyPayload"];

export type PipelineEngineStep = Served["PipelineEngineStep"];

export type PipelineUserStep = Served["PipelineUserStep"];

export type PipelineGraphStage = Served["PipelineGraphStage"];

export type QuarantineGate = Served["QuarantineGate"];

export type PipelineGraphResponse = Served["PipelineGraphResponse"];

export function fetchPipelineGraph(): Promise<PipelineGraphResponse> {
  return fetchJson<PipelineGraphResponse>("/api/v1/pipeline/graph");
}

export type RunTaskInstance = Served["RunTaskInstance"];

export type RunGraphMaster = Served["RunGraphMaster"];

export type MappedFanOutSummary = Served["MappedFanOutSummary"];

export type RunGraphStage = Served["RunGraphStage"];

export type RunGraphResponse = Served["RunGraphResponse"];

export function fetchRunGraph(dagRunId: string): Promise<RunGraphResponse> {
  return fetchJson<RunGraphResponse>(`/api/v1/runtime/runs/${encodeURIComponent(dagRunId)}/graph`);
}

export type TimelineInterval = Served["TimelineInterval"];

export type TimelineMeasurement = Served["TimelineMeasurement"];

export type EpisodeTimeline = Served["EpisodeTimelineResponse"];

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
