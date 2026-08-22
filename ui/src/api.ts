// Typed client for the hflow-ui JSON API (/api/v1).
// Interfaces mirror the M0 contract shapes exactly; this module is the only
// place that talks to the network.

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
}

export interface WorkspaceConfig {
  mode: string;
  read_only: boolean;
  hflow_version: string;
  hflow_ui_version: string;
  data_root: string;
  workspace_id: string | null;
  capabilities: WorkspaceCapabilities;
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
  check_version: string;
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

export interface EpisodesQuery {
  task: string[];
  operator: string[];
  embodiment: string[];
  status: EpisodeStatus | null;
  success: "true" | "false" | null;
  search: string;
  orderBy: string | null;
  order: "asc" | "desc";
  limit: number;
  offset: number;
}

export const DEFAULT_PAGE_SIZE = 50;
export const MAX_PAGE_SIZE = 500;
/** The server's default sort column; mirrored so the header indicator is honest. */
export const DEFAULT_ORDER_BY = "recorded_at";

// --- session token -----------------------------------------------------------
// `hflow ui` prints a tokened URL; the server also sets a cookie on first use.
// We keep the token from the landing URL so API calls work even before the
// cookie exists (e.g. behind the Vite dev proxy) and across client-side routes.

const SESSION_TOKEN_STORAGE_KEY = "hflow-ui-session-token";

function readSessionToken(): string | null {
  let tokenFromUrl: string | null = null;
  try {
    tokenFromUrl = new URLSearchParams(window.location.search).get("token");
  } catch {
    tokenFromUrl = null;
  }
  try {
    if (tokenFromUrl) {
      sessionStorage.setItem(SESSION_TOKEN_STORAGE_KEY, tokenFromUrl);
      return tokenFromUrl;
    }
    return sessionStorage.getItem(SESSION_TOKEN_STORAGE_KEY);
  } catch {
    return tokenFromUrl;
  }
}

const sessionToken = readSessionToken();

/** Append the session token to a same-origin API URL (media, canonical download). */
export function withSessionToken(url: string): string {
  if (!sessionToken) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(sessionToken)}`;
}

// --- fetch helpers ------------------------------------------------------------

async function fetchJson<ResponseBody>(path: string): Promise<ResponseBody> {
  let response: Response;
  try {
    response = await fetch(withSessionToken(path), { headers: { Accept: "application/json" } });
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
  return (await response.json()) as ResponseBody;
}

function buildEpisodesSearchParams(query: EpisodesQuery): URLSearchParams {
  const params = new URLSearchParams();
  for (const value of query.task) params.append("task", value);
  for (const value of query.operator) params.append("operator", value);
  for (const value of query.embodiment) params.append("embodiment", value);
  if (query.status) params.set("status", query.status);
  if (query.success) params.set("success", query.success);
  if (query.search) params.set("search", query.search);
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

/** Human-readable message for any error a query can surface. */
export function describeApiError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 0) {
      return `Could not reach the hflow ui server (${error.detail}). Is \`hflow ui\` running?`;
    }
    if (error.status === 401) {
      return "Not authorized. Reopen the tokened URL that `hflow ui` printed.";
    }
    return error.detail;
  }
  if (error instanceof Error) return error.message;
  return "Unknown error";
}
