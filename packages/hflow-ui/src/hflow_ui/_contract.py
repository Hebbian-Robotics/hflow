"""The published JSON contract: one model per payload this API serves.

The API is the product surface -- the shipped SPA is only its reference
client, and third parties (increasingly coding agents) build against the
schema at ``/api/openapi.json`` (the schema JSON is the whole published docs
surface -- FastAPI's Swagger page is disabled because it loads from a CDN).
So every route declares a model from this module as its response type instead
of hand-building a dict: the model is the ONE owner of that payload's field
names and types, and the generated OpenAPI describes what actually goes over
the wire.

Four shapes stay deliberately open, each because another module owns it and
a mirror here could only drift:

- rows of the wide ``episodes`` view and of a user's own SELECT -- their
  columns ARE data, described alongside the rows by :class:`ColumnDescriptor`;
- DuckDB ``SUMMARIZE`` rows, whose key set varies by DuckDB version;
- the pipeline manifest, owned and version-stamped by ``hflow.manifest``;
- a dag run's ``conf`` (:class:`RuntimeRunSummary`), which is whatever the
  trigger sent -- ``hflow.runtime.AirflowClient.ingest`` owns the shape of
  the ones this API mints, but a run started from Airflow's own UI can carry
  anything, so no model here could describe it honestly.

Nullability follows the catalog's DDL (``hflow.catalog.TABLE_COLUMN_DDL``),
which declares no NOT NULL: a field a stored row could carry as NULL is typed
nullable, so odd data is served honestly instead of turning into a 500 from
response validation.

Two of these models are also the sidecar's ON-DISK shape
(:class:`SavedQueryEntry`, :class:`PinnedManifestEntry`) -- and so is
everything they nest (:class:`CheckCoverageEntry`): ``_sidecar`` stores
exactly what the API serves, so the registry a user can read with ``jq`` and
the payload the API returns can never disagree. Changing any of them
therefore changes the stored format, which ``_sidecar.STATE_VERSION`` guards.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hflow.manifest import StepKind, StepManifest
from hflow.steps import Stage

# --- shared vocabularies -----------------------------------------------------

# Vocabularies the SDK already owns as closed enums (``hflow.steps.Stage``,
# ``hflow.manifest.StepKind``) are annotated with the enum itself rather than
# restated as a Literal: pydantic serializes a StrEnum to its value, so the
# wire bytes are unchanged while the schema publishes the closed set and the
# adapters below stop unwrapping with ``.value``. The aliases here are for
# vocabularies no module owns as a type.
#
# hflow.curation owns the canonical ok/quarantined rule in SQL (the wide
# ``episodes`` view's ``CASE WHEN quarantined THEN 'quarantined' ELSE 'ok'
# END``). This alias is the UI's one restatement of that vocabulary: the
# status filter, the facet values, and the dossier's derived status all use
# it, and _catalog.episode_status_for_quarantine_flag is the one place that
# derives a value of it from a raw flag.
EpisodeStatus = Literal["ok", "quarantined"]

# Stored ``success`` is a stringified boolean whose casing varies by producer,
# so the filter matches case-insensitively on these two spellings.
SuccessFilterValue = Literal["true", "false"]

ListingOrder = Literal["asc", "desc"]

RuntimeSource = Literal["bundle", "remote"]

# How a stage sub-DAG run was attributed to a master run. "heuristic" is the
# only honest answer available: Airflow stores no parent-run link, so the
# attribution is by time window alone (see _graph._matched_stage_run).
StageRunMatch = Literal["heuristic"]

# Which cheap-first tier a registered step runs in (hflow.App._ordered_checks).
StepTier = Literal[1, 2]

# What a browsable catalog relation is, as information_schema reports it.
CatalogTableKind = Literal["view", "table"]


class ColumnDescriptor(BaseModel):
    """One result column as DuckDB's ``DESCRIBE`` reports it."""

    name: str
    type: str


class ValueCount(BaseModel):
    """One value and how many episodes carry it."""

    value: str
    count: int


# --- /api/v1/health, /api/v1/config -------------------------------------------


class HealthResponse(BaseModel):
    """The liveness answer; served without a session token."""

    ok: bool


class WorkspaceCapabilities(BaseModel):
    """What this launch can actually do over this data root.

    ``runtime`` means ADDRESSED (a rendered bundle or an exported remote URL),
    not reachable -- /runtime/status owns liveness.
    """

    catalog: bool
    media: bool
    curation: bool = Field(
        description="Whether the curation studio's durable state can be written at "
        "all: saved queries, the pinned-manifest registry, and the manifest files "
        "need a LOCAL data root, so a bucket-backed workspace answers 501 for every "
        "one of them and the frontend should not offer them."
    )
    runtime: bool
    pipeline: bool


class WorkspaceConfigResponse(BaseModel):
    """What this server is serving, and what the frontend may offer.

    Deliberately carries no Airflow deep-link base: /runtime/status is the one
    owner of the runtime's addressing facts, including its web URL.
    """

    mode: Literal["local"]
    read_only: bool
    hflow_version: str
    hflow_ui_version: str
    data_root: str
    workspace_id: str | None
    capabilities: WorkspaceCapabilities
    run_profiles: list[str] = Field(
        description="Live run-profile names from hflow.steps.RUN_PROFILES, "
        "served so the frontend never hardcodes them."
    )
    ingest_modes: list[str] = Field(
        description="Live ingest modes from hflow.steps.IngestMode; same contract as run_profiles."
    )


# --- /api/v1/episodes ---------------------------------------------------------


class EpisodePageResponse(BaseModel):
    """One filtered, ordered page of the wide ``episodes`` view."""

    rows: list[dict[str, Any]] = Field(
        description="Rows of the wide episodes view. Its columns are data (one per "
        "measurement key present at open time), so they are described by "
        "'columns' rather than enumerated here."
    )
    total: int = Field(description="Rows matching the SAME filters, ignoring limit/offset.")
    columns: list[ColumnDescriptor]
    sql: str = Field(
        description="The SELECT compiled for exactly these filters, with values inlined "
        "so it is copy-pastable and runs against the same catalog."
    )


class EpisodeFacetsResponse(BaseModel):
    """Facet value counts over the wide episodes view; NULL buckets skipped.

    This model is the one owner of WHICH columns are faceted: ``_catalog``
    reads the column list off these fields rather than restating it.
    """

    task: list[ValueCount]
    operator: list[ValueCount]
    embodiment: list[ValueCount]
    status: list[ValueCount]
    pipeline_version: list[ValueCount]


class NumericHistogramBucket(BaseModel):
    """One histogram bucket: ``lo`` inclusive, ``hi`` inclusive on the last."""

    lo: float
    hi: float
    count: int


class NumericColumnStats(BaseModel):
    """A numeric column's mini-distribution under the current filters."""

    name: str
    kind: Literal["numeric"] = "numeric"
    buckets: list[NumericHistogramBucket]


class CategoricalColumnStats(BaseModel):
    """A low-cardinality column's top values under the current filters."""

    name: str
    kind: Literal["categorical"] = "categorical"
    values: list[ValueCount]
    other_count: int = Field(description="Non-null rows beyond the served top values.")


EpisodeColumnStats = Annotated[
    NumericColumnStats | CategoricalColumnStats, Field(discriminator="kind")
]


class EpisodeStatsResponse(BaseModel):
    """Per-column mini-distributions; degenerate columns are omitted entirely."""

    columns: list[EpisodeColumnStats]


class DossierEpisode(BaseModel):
    """The episode's own ``episodes_latest`` row plus the two derived fields.

    ``extra="allow"``: every column of that row rides along unchanged, because
    the catalog's columns are data this module cannot enumerate.
    """

    model_config = ConfigDict(extra="allow")

    status: EpisodeStatus
    quarantine_tags: list[str] = Field(
        description="Parsed out of the row's quarantine_tags_json; empty when not quarantined."
    )


class EpisodeMeasurementRecord(BaseModel):
    """One measurement, latest per key."""

    key: str | None
    value_double: float | None
    value_text: str | None
    value_bool: bool | None
    check_name: str | None
    check_version: str | None
    recorded_at: str | None


class EpisodeCheckRunRecord(BaseModel):
    """One recorded check invocation."""

    check_name: str | None
    check_version: str | None
    critical: bool | None
    status: str | None
    duration_s: float | None
    error: str | None
    recorded_at: str | None
    run_fingerprint: str | None


class EpisodeIntervalRecord(BaseModel):
    """One interval of the episode's LATEST run.

    ``check_version`` rides in from that run's ``check_runs`` row (a LEFT
    JOIN -- the intervals table carries no version of its own).
    """

    label: str | None
    start_ns: int | None
    end_ns: int | None
    check_name: str | None
    check_version: str | None


class EpisodeTagRecord(BaseModel):
    """One tag of the episode's LATEST run."""

    tag: str | None
    check_name: str | None
    recorded_at: str | None


class EpisodeMediaArtifact(BaseModel):
    """One cataloged media artifact and, when servable, its byte URL."""

    name: str
    uri: str
    url: str | None = Field(
        description="Same-origin byte-serving path, or null when the cataloged file "
        "is missing or lands outside the workspace data root."
    )


class EpisodeDossierResponse(BaseModel):
    """Everything the episode page shows for one episode."""

    episode: DossierEpisode
    measurements: list[EpisodeMeasurementRecord]
    check_runs: list[EpisodeCheckRunRecord]
    intervals: list[EpisodeIntervalRecord]
    tags: list[EpisodeTagRecord]
    history: list[dict[str, Any]] = Field(
        description="Every append of this episode, newest first: raw episodes_raw rows, "
        "whose columns are the catalog's (see EpisodePageResponse.rows)."
    )
    media: list[EpisodeMediaArtifact]
    canonical_url: str | None


class TimelineInterval(BaseModel):
    """One interval placed on the episode's axis, in absolute ns and in
    seconds RELATIVE to the span start (both computed server-side)."""

    label: str | None
    start_ns: int | None
    end_ns: int | None
    start_s: float | None
    end_s: float | None
    check_name: str | None
    kind: str = Field(
        description="Colour group: the label's '<kind>:<topic>' prefix, else the "
        "whole label, else the check that produced it."
    )


class TimelineMeasurement(BaseModel):
    """One numeric measurement, ready to draw as a bar."""

    key: str
    value: float
    unit: str | None = Field(
        description="Inferred from the key's unit suffix; null when no dimension is known."
    )


class EpisodeTimelineResponse(BaseModel):
    """One episode's time axis. All-null bounds mean the span is unknown --
    the UI must say so rather than draw a fabricated axis."""

    start_ns: int | None
    end_ns: int | None
    duration_s: float | None
    intervals: list[TimelineInterval]
    measurements: list[TimelineMeasurement]


# --- /api/v1/curation, /api/v1/queries, /api/v1/manifests ---------------------


class CurationPreviewResponse(BaseModel):
    """A user SELECT's first rows, its full count, and optional column stats."""

    columns: list[ColumnDescriptor]
    rows: list[dict[str, Any]] = Field(
        description="Rows of the user's own SELECT; its columns are described by 'columns'."
    )
    row_count: int = Field(description="Rows the SELECT returns in full, independent of limit.")
    truncated: bool
    column_stats: list[dict[str, Any]] | None = Field(
        description="DuckDB SUMMARIZE rows (column_name, column_type, min, max, "
        "null_percentage, ...). DuckDB owns that shape and varies it by version, "
        "so it is served as-is. Null unless the request asked for stats."
    )
    sql: str = Field(
        description="The logical wrapped SELECT, copy-pastable as-is. The executed "
        "statement adds a '* REPLACE (...)' projection rendering TIMESTAMPTZ columns "
        "as UTC ISO text (the locked connection cannot SET TimeZone), which is a "
        "rendering detail of these rows rather than part of the query a user wrote."
    )


class CheckCoverageEntry(BaseModel):
    """One check's coverage denominator over the WHOLE catalog, not the cut.

    Also the sidecar's stored shape, nested inside every stored manifest
    entry's ``coverage`` (see the module note).
    """

    check_name: str
    episodes_ran: int
    total_episodes: int
    fraction: float


class CurationReportResponse(BaseModel):
    """What a cut would contain, and what evidence backs it -- no files written."""

    row_count: int
    total_episodes: int
    coverage: list[CheckCoverageEntry]


class SavedQueryEntry(BaseModel):
    """One saved studio query.

    Also the sidecar's stored shape for a saved query (see the module note).
    """

    model_config = ConfigDict(populate_by_name=True)

    query_id: str = Field(alias="id")
    name: str
    sql: str
    updated_at: str = Field(description="ISO-8601 UTC.")


class SavedQueryListResponse(BaseModel):
    queries: list[SavedQueryEntry]


class PinnedManifestEntry(BaseModel):
    """One registry entry for an immutable pinned manifest file.

    Also the sidecar's stored shape for a manifest (see the module note).
    """

    model_config = ConfigDict(populate_by_name=True)

    manifest_id: str = Field(alias="id")
    name: str
    description: str
    sql: str
    manifest_path: str = Field(
        description="Data-root-relative, e.g. 'manifests/<slug>-<utc timestamp>.parquet'."
    )
    row_count: int
    total_episodes: int
    coverage: list[CheckCoverageEntry] = Field(description="Frozen at pin time.")
    created_at: str = Field(description="ISO-8601 UTC.")


class PinnedManifestListResponse(BaseModel):
    manifests: list[PinnedManifestEntry]


class CatalogTableDescription(BaseModel):
    """One browsable catalog relation and its live columns."""

    name: str
    kind: CatalogTableKind
    columns: list[ColumnDescriptor]


class CatalogTablesResponse(BaseModel):
    tables: list[CatalogTableDescription]


class CatalogTableSummaryResponse(BaseModel):
    """One relation's row count and DuckDB's own column profile."""

    row_count: int
    columns: list[dict[str, Any]] = Field(
        description="DuckDB SUMMARIZE rows; see CurationPreviewResponse.column_stats."
    )


# --- /api/v1/runtime ----------------------------------------------------------


class RuntimeHealthComponents(BaseModel):
    """Airflow's per-component health.

    This model is the one owner of WHICH components /runtime/status reports:
    ``_runtime`` reads the names off these fields. A component absent from the
    deployment (a minimal stack runs no triggerer) reports null.
    """

    metadatabase: str | None
    scheduler: str | None
    triggerer: str | None
    dag_processor: str | None


class RuntimeStatusResponse(BaseModel):
    """Whether this workspace's ingest runtime is addressed AND answering.

    Every field except ``available`` defaults to "not known", so an
    unavailable answer states only the facts it actually has -- there is no
    second hand-written shape for the unavailable case to drift from.
    """

    available: bool
    detail: str | None = Field(
        default=None, description="Why the runtime is unavailable; null when it is available."
    )
    source: RuntimeSource | None = None
    airflow_web_url: str | None = Field(
        default=None,
        description="Deep-link base for the Airflow web UI, AS ADDRESSED FROM THE "
        "WORKSPACE HOST. Only a local bundle records its own address; a remote "
        "endpoint's is unknown, never guessed.",
    )
    airflow_web_url_host_only: bool = Field(
        default=False,
        description="True when airflow_web_url is a loopback address, so it resolves "
        "only on the workspace host: a browser on another machine cannot follow it, "
        "and the runtime is reachable there only through a tunnel or a wider "
        "`hflow up --api-bind-host`.",
    )
    dag_id: str | None = None
    registered: bool | None = Field(
        default=None,
        description="Whether the master DAG is registered. Null means unknown (an auth "
        "or transient failure), which is not the same as false.",
    )
    health: RuntimeHealthComponents | None = None


class RuntimeRunSummary(BaseModel):
    """One master DAG run, reduced to what the Runs page shows."""

    dag_run_id: str | None
    state: str | None
    logical_date: str | None
    start_date: str | None
    end_date: str | None
    conf: dict[str, Any] = Field(description="The trigger's own input, forwarded verbatim.")


class StageRunSummary(BaseModel):
    """One stage sub-DAG run in a stage's recent strip."""

    dag_run_id: str | None
    state: str | None
    start_date: str | None
    end_date: str | None


class StageRecentRuns(BaseModel):
    """One stage's most recent runs. NOT correlated with any master run."""

    stage: Stage
    dag_id: str
    recent: list[StageRunSummary]


class RuntimeRunsResponse(BaseModel):
    runs: list[RuntimeRunSummary]
    stages: list[StageRecentRuns] | None = Field(
        description="Per-stage recent runs; null for a remote runtime, whose stage "
        "sub-DAG ids only a bundle manifest records."
    )


class IngestTriggerResponse(BaseModel):
    """What Airflow answered when the run was triggered."""

    dag_run_id: str | None
    state: str | None


# --- /api/v1/pipeline ---------------------------------------------------------


class PipelineStepManifest(BaseModel):
    """One registered step, exactly as ``hflow.manifest.StepManifest`` renders it."""

    name: str
    kind: StepKind
    version: str = Field(description="Content hash of the live function.")
    critical: bool
    requires: list[str]
    uses: str | None

    @classmethod
    def from_step_manifest(cls, step: StepManifest) -> "PipelineStepManifest":
        return cls(
            name=step.name,
            kind=step.kind,
            version=step.version,
            critical=step.critical,
            requires=list(step.requires),
            uses=step.uses,
        )


class PipelineStageLane(BaseModel):
    """One ingest-stage lane of the pipeline page."""

    stage: Stage
    engine_owned: bool = Field(
        description="True for lanes whose work is engine builtins (sync, media) "
        "rather than user-registered steps."
    )
    steps: list[PipelineStepManifest] = Field(
        description="The lane's registered steps in EXECUTION order (cheap tier "
        "first), the same order and the same steps /pipeline/graph serves as "
        "'user_steps' -- only the tier field differs."
    )


class ObservedCheckVersion(BaseModel):
    """What the catalog has SEEN of one (check, version) pair."""

    check_name: str | None
    check_version: str | None
    first_seen: str | None
    last_seen: str | None
    run_count: int


class StaleSummary(BaseModel):
    """How many recorded episodes are stale against the App's current versions."""

    pipeline_version: str
    count: int


class PipelineResponse(BaseModel):
    """The startup-imported App, described over this workspace's catalog."""

    manifest: dict[str, Any] = Field(
        description="The pipeline manifest exactly as hflow.manifest.PipelineManifest "
        "renders it. hflow.manifest owns that shape and stamps it with "
        "'manifest_version', so it is forwarded rather than mirrored here."
    )
    stages: list[PipelineStageLane]
    observed: list[ObservedCheckVersion]
    stale: StaleSummary | None = Field(
        description="Null when staleness is unknowable (no catalog yet)."
    )


# --- /api/v1/pipeline/graph, /api/v1/runtime/runs/{id}/graph -------------------


class DagTaskNodePayload(BaseModel):
    """One task of a generated DAG (mirrors ``hflow.runtime.DagTaskNode``)."""

    task_id: str
    summary: str
    mapped: bool = Field(description="Dynamically mapped: one instance per planned batch.")
    deferred: bool = Field(description="Defers instead of holding a worker slot.")


class DagTopologyPayload(BaseModel):
    """One DAG's real shape: its tasks and their real dependency edges."""

    dag_id: str
    tasks: list[DagTaskNodePayload]
    edges: list[tuple[str, str]] = Field(
        description="[upstream, downstream] task-id pairs, in declaration order."
    )


class PipelineEngineStep(BaseModel):
    """Engine work inside one stage that no manifest lists."""

    name: str
    summary: str


class PipelineUserStep(PipelineStepManifest):
    """A registered step as the graph endpoint serves it.

    ``tier`` mirrors ``hflow.App._ordered_checks``: tier 2 is exactly the steps
    declaring ``requires`` or ``uses``. Steps within a tier have NO ordering.
    """

    tier: StepTier

    @classmethod
    def from_step_manifest_in_tier(cls, step: StepManifest, tier: StepTier) -> "PipelineUserStep":
        return cls(**PipelineStepManifest.from_step_manifest(step).model_dump(), tier=tier)


class QuarantineGate(BaseModel):
    """The one real cross-step edge, served as its own object rather than as
    an edge in either graph."""

    from_stage: Stage
    to_stages: list[Stage]
    critical_step_names: list[str]
    explanation: str


class PipelineGraphStage(BaseModel):
    """One stage lane of the pipeline graph: its DAG plus what runs inside it."""

    stage: Stage
    title: str
    description: str
    gate_task_id: str
    trigger_task_id: str
    enabling_profiles: list[str]
    dag: DagTopologyPayload
    engine_steps: list[PipelineEngineStep]
    user_steps: list[PipelineUserStep]


class PipelineGraphResponse(BaseModel):
    """The ingest DAG's shape merged with the pipeline's own steps."""

    dag_ids_known: bool = Field(
        description="False when no runtime is addressed: the dag ids are display-only."
    )
    steps_known: bool = Field(
        description="False without --pipeline: what runs inside process_batch is unknown."
    )
    master: DagTopologyPayload
    stages: list[PipelineGraphStage]
    quarantine_gate: QuarantineGate | None = Field(
        description="Null exactly when steps_known is false."
    )


class RunTaskInstance(BaseModel):
    """One Airflow task instance, reduced to what the graph draws."""

    task_id: str | None
    state: str | None
    start_date: str | None
    end_date: str | None
    queued_at: str | None = Field(
        description="When the scheduler queued the task, so a replay can tell "
        "'waiting for a worker' from 'running'. Airflow may omit it."
    )
    try_number: int | None
    map_index: int = Field(description="-1 means the task is not mapped.")
    duration_s: float | None


class MappedFanOutSummary(BaseModel):
    """The fan-out's live split, counted server-side over EVERY mapped instance.

    Complete on its own: ``by_state`` partitions all ``total`` instances of
    ``task_id`` (an instance Airflow has not scheduled yet counts under
    ``no_status``), so ``total == sum(by_state.values())`` always holds and a
    client never has to recount the raw instances to size or colour the
    fan-out. Only a replay at some earlier instant is a different fact, and
    that one the server cannot answer.
    """

    task_id: str
    total: int = Field(
        description="Instances reported for the mapped task. Before the fan-out expands "
        "Airflow reports one unexpanded instance, which is counted -- that is the "
        "truth at that moment."
    )
    by_state: dict[str, int]


class RunGraphMaster(BaseModel):
    """The master run's own live state."""

    dag_run_id: str
    state: str | None
    tasks: list[RunTaskInstance]


class RunGraphStage(BaseModel):
    """One stage's live state for this master run, or explicit nulls when the
    stage never ran for it."""

    stage: Stage
    dag_id: str
    dag_run_id: str | None
    state: str | None
    match: StageRunMatch | None = Field(
        description="How this stage run was attributed to the master run. Airflow "
        "stores no parent-run link, so the only honest answer is 'heuristic' -- the "
        "earliest stage run started inside this master run's own window -- or null "
        "(nothing matched). Two master runs OVERLAPPING in time can still be "
        "attributed the same stage run."
    )
    tasks: list[RunTaskInstance]
    mapped_summary: MappedFanOutSummary | None


class RunGraphResponse(BaseModel):
    """One master run's live state over the ingest topology."""

    master: RunGraphMaster
    stages: list[RunGraphStage]


# --- byte-serving routes ------------------------------------------------------

# The three routes that answer with FILE BYTES rather than JSON. Declared so
# the schema says "binary" instead of the empty schema FastAPI publishes for a
# bare Response return type; the routes pair this with
# ``response_class=FileResponse``, which is what drops the phantom
# application/json entry beside it.
BINARY_FILE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "The file's bytes. An allowlisted inert media type (image, audio, "
        "video) is served inline under its own content type; anything else -- and every "
        "download -- is opaque application/octet-stream.",
        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
    }
}
