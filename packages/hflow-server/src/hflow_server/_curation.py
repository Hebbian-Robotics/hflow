"""The curation studio API: preview/report/pin, manifests, queries, tables.

User-supplied SQL only ever runs on a CONSTRAINED connection
(``hflow.open_catalog_connection(..., constrained=True)`` /
``hflow.curate(..., constrained=True)``): the catalog is materialized in
memory at open, file access and extension loading are locked out, so the SQL
can read the data but can never touch the catalog's files -- hosted parity,
and defense in depth even locally. The server wraps that SQL as a subquery
(``SELECT ... FROM (<sql>)``) for LIMITing, counting, and SUMMARIZE, so a
smuggled second statement is a parser error, and every DuckDB parser/binder
error travels back as a 400 whose detail is DuckDB's own message -- the
useful part -- never a 500.

Workspace convention: pinned manifests are immutable files at
``<data_root>/manifests/<slug>-<utc timestamp>.parquet`` -- never the
engine's default ``<data_root>/manifest.parquet``, which the CLI's curate
silently overwrites. A pin refuses loudly rather than overwrite anything.
The registry describing them lives in the sidecar (see ``_sidecar``).
"""

import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from hflow.curation import (
    CurationReport,
    NonSingleSelectQueryError,
    curate,
    reject_non_single_select,
)
from hflow.dataset import ManifestAlreadyExistsError, write_dataset_manifest
from hflow.workspace import Workspace
from hflow_server import _catalog, _connections, _media, _sidecar
from hflow_server._contract import (
    BINARY_FILE_RESPONSES,
    CatalogTableDescription,
    CatalogTableKind,
    CatalogTablesResponse,
    CatalogTableSummaryResponse,
    CheckCoverageEntry,
    ColumnDescriptor,
    CurationPreviewResponse,
    CurationReportResponse,
    PinnedManifestEntry,
    PinnedManifestListResponse,
    SavedQueryEntry,
    SavedQueryListResponse,
)
from hflow_server._settings import ServerSettings, refuse_when_read_only

# BROWSING ORDER only, never membership: which relations exist is
# hflow.open_catalog_connection's fact, read live off information_schema (see
# _browsable_relations), so a view the SDK adds or renames shows up here
# instead of 404ing from the summary route or 500ing on a DESCRIBE. A
# relation missing from this tuple simply sorts after the familiar ones.
CATALOG_TABLE_BROWSING_ORDER = (
    "episodes",
    "episodes_latest",
    "episodes_raw",
    "check_runs",
    "check_runs_latest",
    "measurements",
    "measurements_latest",
    "observations",
    "observations_latest",
    "tags",
    "intervals",
    "ingest_failures",
)

_TIMESTAMPTZ_TYPE = "TIMESTAMP WITH TIME ZONE"

# What a read-only launch refuses on this router; the sentence around it (and
# the 403) belongs to _settings.refuse_when_read_only.
_STUDIO_WRITE_ACTIONS = "pinning manifests and editing saved queries are"

# Upper bounds on everything that can be persisted into the sidecar (which is
# fully re-read and re-serialized on every list request): a name, a
# description, one SQL body, and the number of stored entries. Generous for
# real use, but they make the one file this server writes outside manifests/
# bounded instead of unbounded.
_MAX_NAME_LENGTH = 200
_MAX_DESCRIPTION_LENGTH = 2000
_MAX_SQL_LENGTH = 100_000
_MAX_SAVED_QUERIES = 1000
_MAX_PINNED_MANIFESTS = 1000


class PreviewRequest(BaseModel):
    sql: str = Field(max_length=_MAX_SQL_LENGTH)
    limit: int = Field(default=100, ge=1, le=1000)
    stats: bool = False


class ReportRequest(BaseModel):
    sql: str = Field(max_length=_MAX_SQL_LENGTH)


class PinRequest(BaseModel):
    sql: str = Field(max_length=_MAX_SQL_LENGTH)
    name: str = Field(min_length=1, max_length=_MAX_NAME_LENGTH)
    description: str = Field(default="", max_length=_MAX_DESCRIPTION_LENGTH)


class SavedQueryCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=_MAX_NAME_LENGTH)
    sql: str = Field(max_length=_MAX_SQL_LENGTH)


class SavedQueryUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=_MAX_NAME_LENGTH)
    sql: str | None = Field(default=None, max_length=_MAX_SQL_LENGTH)


# Fallback filename slug when a name has no ASCII alphanumerics (a name in a
# non-Latin script, or symbols only). The full Unicode name is still stored on
# the registry entry; only the on-disk filename uses the slug, and the
# timestamp suffix keeps every filename unique regardless.
_FALLBACK_MANIFEST_SLUG = "manifest"


def slugified_manifest_name(raw_name: str) -> str:
    """The user-given name as a filename slug: lowercase, [a-z0-9-], dashes
    collapsed. Names with no ASCII alphanumerics (e.g. ``数据集``, ``!!!``)
    slug to the fallback rather than being refused."""
    slug = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")
    return slug or _FALLBACK_MANIFEST_SLUG


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _manifest_timestamp() -> str:
    # Microsecond precision: pins of the same name in the same second still
    # get distinct files (pins never overwrite; collisions are refused).
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _stripped_sql_or_refuse(raw_sql: str) -> str:
    """The user SQL with trailing semicolons dropped (they break subquerying)."""
    stripped_sql = raw_sql.strip().rstrip(";").strip()
    if not stripped_sql:
        raise HTTPException(status_code=400, detail="sql must be a non-empty SELECT")
    return stripped_sql


def _bad_sql_refusal(error: duckdb.Error) -> HTTPException:
    # DuckDB's parser/binder message IS the useful diagnostic; bad SQL is the
    # caller's mistake, never a server fault (so 400, never 500).
    return HTTPException(status_code=400, detail=str(error))


def _reject_non_single_select(user_sql: str) -> None:
    """Refuse anything that is not exactly one SELECT statement.

    ``hflow.reject_non_single_select`` owns the rule (exactly one statement
    whose type is SELECT, judged by parsing with ``extract_statements``
    WITHOUT executing anything); this route only maps its two failure modes
    onto the two client-facing 400s: DuckDB's own diagnostic for SQL the
    parser rejects, the fixed sentence for SQL that is well-formed but not a
    single SELECT.
    """
    try:
        reject_non_single_select(user_sql)
    except duckdb.Error as error:
        raise _bad_sql_refusal(error) from error
    except NonSingleSelectQueryError:
        raise HTTPException(
            status_code=400, detail="sql must be exactly one read-only SELECT statement"
        ) from None


def _browsable_relations(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, CatalogTableKind]:
    """Every relation this catalog connection registered, in browsing order.

    ``hflow.open_catalog_connection`` owns WHICH relations exist;
    ``information_schema`` is that fact as the live connection reports it, so
    this endpoint and the summary route below both derive membership from the
    connection they already hold rather than from a second list here.
    """
    kind_rows = connection.execute(
        "SELECT table_name, table_type FROM information_schema.tables"
    ).fetchall()
    kind_by_name: dict[str, CatalogTableKind] = {
        str(table_name): ("view" if str(table_type).upper() == "VIEW" else "table")
        for table_name, table_type in kind_rows
    }
    unfamiliar_position = len(CATALOG_TABLE_BROWSING_ORDER)
    return {
        name: kind_by_name[name]
        for name in sorted(
            kind_by_name,
            key=lambda name: (
                CATALOG_TABLE_BROWSING_ORDER.index(name)
                if name in CATALOG_TABLE_BROWSING_ORDER
                else unfamiliar_position,
                name,
            ),
        )
    }


def _described_columns(
    connection: duckdb.DuckDBPyConnection, user_sql: str
) -> list[ColumnDescriptor]:
    described_rows = connection.execute(f"DESCRIBE SELECT * FROM ({user_sql})").fetchall()
    return [ColumnDescriptor(name=str(row[0]), type=str(row[1])) for row in described_rows]


def _timestamp_replace_clause(columns: list[ColumnDescriptor]) -> str:
    """A ``* REPLACE (...)`` clause rendering TIMESTAMPTZ results as ISO UTC text.

    Materializing a TIMESTAMPTZ into Python requires pytz (deliberately not a
    dependency), and the locked connection cannot ``SET TimeZone`` -- so the
    rendering converts to UTC in SQL (``AT TIME ZONE 'UTC'`` yields the naive
    UTC wall time) and appends the offset literally. A TIMESTAMPTZ nested
    inside a LIST/STRUCT is rendered whole via CAST to text.
    """
    replacements: list[str] = []
    for column in columns:
        quoted_name = _catalog.quoted_identifier(column.name)
        if column.type == _TIMESTAMPTZ_TYPE:
            replacements.append(
                f"strftime({quoted_name} AT TIME ZONE 'UTC', '%Y-%m-%dT%H:%M:%S.%f') "
                f"|| '+00:00' AS {quoted_name}"
            )
        elif _TIMESTAMPTZ_TYPE in column.type:
            replacements.append(f"CAST({quoted_name} AS VARCHAR) AS {quoted_name}")
    return f"REPLACE ({', '.join(replacements)})" if replacements else ""


def run_preview(
    connection: duckdb.DuckDBPyConnection, user_sql: str, *, limit: int, include_stats: bool
) -> CurationPreviewResponse:
    """Preview rows, the full count, and (optionally) SUMMARIZE column stats."""
    columns = _described_columns(connection, user_sql)
    replace_clause = _timestamp_replace_clause(columns)
    select_head = f"SELECT * {replace_clause}" if replace_clause else "SELECT *"
    rows = _catalog.fetched_json_safe_rows(
        connection.execute(f"{select_head} FROM ({user_sql}) LIMIT ?", [limit])
    )
    count_row = connection.execute(f"SELECT count(*) FROM ({user_sql})").fetchone()
    row_count = int(count_row[0]) if count_row is not None else 0
    # SUMMARIZE over the SAME timestamp-replaced projection the rows use: the
    # constrained connection cannot SET TimeZone (locked at open), so a bare
    # SUMMARIZE would stringify TIMESTAMPTZ min/max/quartiles in the host's
    # timezone -- inconsistent with (and a different calendar day from) the
    # UTC ISO text the preview rows already carry.
    column_stats = (
        _catalog.fetched_json_safe_rows(
            connection.execute(f"SUMMARIZE {select_head} FROM ({user_sql})")
        )
        if include_stats
        else None
    )
    return CurationPreviewResponse(
        columns=columns,
        rows=rows,
        row_count=row_count,
        truncated=row_count > len(rows),
        column_stats=column_stats,
        # The LOGICAL wrapper, deliberately without select_head's REPLACE: the
        # timestamp rendering is how these rows are transported, not part of
        # the query a user wrote, so what is served stays copy-pastable -- the
        # same split (and the same wording) _catalog's episode listing makes.
        sql=f"SELECT * FROM ({user_sql}) LIMIT {limit}",
    )


def _curated_or_refused(data_root: str, user_sql: str, *, output: Path | None) -> CurationReport:
    try:
        return curate(
            Workspace.parse(data_root).catalog_root, user_sql, output=output, constrained=True
        )
    except (FileNotFoundError, ValueError) as error:
        raise _connections.catalog_unavailable_refusal(error) from error
    except duckdb.Error as error:
        raise _bad_sql_refusal(error) from error


def _coverage_entries(report: CurationReport) -> list[CheckCoverageEntry]:
    """One curation report's coverage as the served (and pinned) entries."""
    return [
        CheckCoverageEntry(
            check_name=entry.check_name,
            episodes_ran=entry.episodes_ran,
            total_episodes=entry.total_episodes,
            fraction=entry.fraction,
        )
        for entry in report.coverage
    ]


def create_curation_router(settings: ServerSettings) -> APIRouter:
    """Every curation-studio route, closed over one launch's settings."""
    router = APIRouter(prefix="/api/v1")
    # FastAPI runs these sync endpoints on a threadpool, so two overlapping
    # writes (double-submit, two tabs) would both read the same base sidecar
    # and the later store would silently drop the earlier's entry. One
    # process-wide lock serializes the whole load->modify->store of every
    # mutating route -- sufficient for the single-server design.
    sidecar_write_lock = threading.Lock()

    # A SidecarError is already an HTTPException, so these three only bind the
    # data root -- nothing catches and re-raises, and a refusal raised inside
    # them reaches the client with the status and detail _sidecar chose.
    def loaded_sidecar_state() -> _sidecar.SidecarState:
        return _sidecar.load_sidecar_state(settings.data_root)

    def stored_sidecar_state(state: _sidecar.SidecarState) -> None:
        _sidecar.store_sidecar_state(settings.data_root, state)

    @router.post("/curation/preview")
    def run_curation_preview(request: PreviewRequest) -> CurationPreviewResponse:
        user_sql = _stripped_sql_or_refuse(request.sql)
        _reject_non_single_select(user_sql)
        with _connections.opened_constrained_connection_or_refuse(settings.data_root) as connection:
            try:
                return run_preview(
                    connection, user_sql, limit=request.limit, include_stats=request.stats
                )
            except duckdb.Error as error:
                raise _bad_sql_refusal(error) from error

    @router.post("/curation/report")
    def run_curation_report(request: ReportRequest) -> CurationReportResponse:
        user_sql = _stripped_sql_or_refuse(request.sql)
        _reject_non_single_select(user_sql)
        report = _curated_or_refused(settings.data_root, user_sql, output=None)
        return CurationReportResponse(
            row_count=report.row_count,
            total_episodes=report.total_episodes,
            coverage=_coverage_entries(report),
        )

    @router.post("/curation/pin")
    def pin_manifest(request: PinRequest) -> PinnedManifestEntry:
        refuse_when_read_only(settings, disabled_actions=_STUDIO_WRITE_ACTIONS)
        user_sql = _stripped_sql_or_refuse(request.sql)
        _reject_non_single_select(user_sql)
        manifest_slug = slugified_manifest_name(request.name)
        with sidecar_write_lock:
            # Load (and thereby validate) the sidecar BEFORE writing the
            # manifest, so a corrupt registry never strands an unregistered
            # manifest file. The whole load->curate->store runs under the lock
            # so a concurrent write cannot drop this pin's acknowledged entry.
            state = loaded_sidecar_state()
            if len(state.manifests) >= _MAX_PINNED_MANIFESTS:
                raise HTTPException(
                    status_code=409,
                    detail=f"this workspace already has {_MAX_PINNED_MANIFESTS} pinned "
                    "manifests (the registry cap); remove some before pinning more",
                )
            # The SDK owns writing a manifest into a workspace (staging, the
            # create-if-absent publish, the filename convention), so pinning
            # works on a bucket-backed workspace and cannot drift from
            # `hflow dataset create`. Constrained, because this SQL is the
            # tenant's.
            try:
                written = write_dataset_manifest(
                    Workspace.parse(settings.data_root),
                    name=request.name,
                    sql=user_sql,
                    constrained=True,
                    file_stem=f"{manifest_slug}-{_manifest_timestamp()}",
                )
            except ManifestAlreadyExistsError as error:
                raise HTTPException(
                    status_code=409,
                    detail=f"{error} -- retry the pin",
                ) from error
            except (FileNotFoundError, ValueError) as error:
                raise _connections.catalog_unavailable_refusal(error) from error
            except duckdb.Error as error:
                raise _bad_sql_refusal(error) from error
            entry = PinnedManifestEntry(
                manifest_id=uuid.uuid4().hex,
                name=request.name,
                description=request.description,
                sql=user_sql,
                manifest_path=written.relative_key,
                row_count=written.report.row_count,
                total_episodes=written.report.total_episodes,
                coverage=_coverage_entries(written.report),
                created_at=_utc_now_iso(),
            )
            stored_sidecar_state(
                _sidecar.SidecarState(
                    saved_queries=state.saved_queries, manifests=(*state.manifests, entry)
                )
            )
        return entry

    @router.get("/manifests")
    def list_manifests() -> PinnedManifestListResponse:
        state = loaded_sidecar_state()
        newest_first = sorted(state.manifests, key=lambda entry: entry.created_at, reverse=True)
        return PinnedManifestListResponse(manifests=newest_first)

    @router.get(
        "/manifests/{manifest_id}/download",
        response_class=FileResponse,
        responses=BINARY_FILE_RESPONSES,
    )
    def download_manifest(manifest_id: str) -> FileResponse:
        state = loaded_sidecar_state()
        entry = next(
            (manifest for manifest in state.manifests if manifest.manifest_id == manifest_id),
            None,
        )
        if entry is None:
            raise HTTPException(
                status_code=404, detail=f"no pinned manifest with id {manifest_id!r}"
            )
        # Fetched through the storage root, so a bucket-backed workspace serves
        # its manifests like any other: fetch downloads into the mirror there
        # and is the file itself for a local root.
        try:
            manifest_file = Workspace.parse(settings.data_root).storage_root.fetch(
                entry.manifest_path
            )
        except ValueError as error:
            # A hand-edited registry key that escapes the root. The storage
            # layer refuses it before any filesystem call, one step earlier
            # than the served-file containment check below -- so it has to
            # land on the same 403, and the path is never echoed back.
            raise HTTPException(
                status_code=403,
                detail="pinned manifest path is not inside this workspace",
            ) from error
        except (FileNotFoundError, OSError) as error:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"pinned manifest {entry.manifest_path!r} is registered but its file "
                    f"is not in this workspace: {error}"
                ),
            ) from error
        # The same strict-resolve + containment check media serving uses: even
        # a hand-edited registry path can only serve workspace files. Its
        # refusal is already an HTTPException, so it needs no rewrapping here.
        resolved_file = _media.resolve_served_file(str(manifest_file), data_root=settings.data_root)
        return _media.served_file_response(
            resolved_file, attachment_filename=Path(entry.manifest_path).name
        )

    @router.get("/queries")
    def list_saved_queries() -> SavedQueryListResponse:
        state = loaded_sidecar_state()
        return SavedQueryListResponse(queries=list(state.saved_queries))

    @router.post("/queries")
    def create_saved_query(request: SavedQueryCreateRequest) -> SavedQueryEntry:
        refuse_when_read_only(settings, disabled_actions=_STUDIO_WRITE_ACTIONS)
        query_name = request.name.strip()
        if not query_name:
            raise HTTPException(status_code=400, detail="name must be non-empty")
        entry = SavedQueryEntry(
            query_id=uuid.uuid4().hex,
            name=query_name,
            sql=_stripped_sql_or_refuse(request.sql),
            updated_at=_utc_now_iso(),
        )
        with sidecar_write_lock:
            state = loaded_sidecar_state()
            if len(state.saved_queries) >= _MAX_SAVED_QUERIES:
                raise HTTPException(
                    status_code=409,
                    detail=f"this workspace already has {_MAX_SAVED_QUERIES} saved queries "
                    "(the sidecar cap); remove some before saving more",
                )
            stored_sidecar_state(
                _sidecar.SidecarState(
                    saved_queries=(*state.saved_queries, entry), manifests=state.manifests
                )
            )
        return entry

    @router.put("/queries/{query_id}")
    def update_saved_query(query_id: str, request: SavedQueryUpdateRequest) -> SavedQueryEntry:
        refuse_when_read_only(settings, disabled_actions=_STUDIO_WRITE_ACTIONS)
        with sidecar_write_lock:
            state = loaded_sidecar_state()
            existing = next(
                (entry for entry in state.saved_queries if entry.query_id == query_id), None
            )
            if existing is None:
                raise HTTPException(status_code=404, detail=f"no saved query with id {query_id!r}")
            updated_name = existing.name
            if request.name is not None:
                updated_name = request.name.strip()
                if not updated_name:
                    raise HTTPException(status_code=400, detail="name must be non-empty")
            updated_sql = (
                _stripped_sql_or_refuse(request.sql) if request.sql is not None else existing.sql
            )
            updated_entry = SavedQueryEntry(
                query_id=query_id, name=updated_name, sql=updated_sql, updated_at=_utc_now_iso()
            )
            stored_sidecar_state(
                _sidecar.SidecarState(
                    saved_queries=tuple(
                        updated_entry if entry.query_id == query_id else entry
                        for entry in state.saved_queries
                    ),
                    manifests=state.manifests,
                )
            )
        return updated_entry

    @router.delete("/queries/{query_id}", status_code=204)
    def delete_saved_query(query_id: str) -> Response:
        refuse_when_read_only(settings, disabled_actions=_STUDIO_WRITE_ACTIONS)
        with sidecar_write_lock:
            state = loaded_sidecar_state()
            if all(entry.query_id != query_id for entry in state.saved_queries):
                raise HTTPException(status_code=404, detail=f"no saved query with id {query_id!r}")
            stored_sidecar_state(
                _sidecar.SidecarState(
                    saved_queries=tuple(
                        entry for entry in state.saved_queries if entry.query_id != query_id
                    ),
                    manifests=state.manifests,
                )
            )
        return Response(status_code=204)

    @router.get("/catalog/tables")
    def list_catalog_tables() -> CatalogTablesResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            return CatalogTablesResponse(
                tables=[
                    CatalogTableDescription(
                        name=table_name,
                        kind=kind,
                        columns=[
                            ColumnDescriptor(name=str(row[0]), type=str(row[1]))
                            for row in connection.execute(
                                f"DESCRIBE {_catalog.quoted_identifier(table_name)}"
                            ).fetchall()
                        ],
                    )
                    for table_name, kind in _browsable_relations(connection).items()
                ]
            )

    @router.get("/catalog/tables/{table_name}/summary")
    def read_catalog_table_summary(table_name: str) -> CatalogTableSummaryResponse:
        with _connections.opened_workspace_connection_or_refuse(settings.data_root) as connection:
            # Identifier-validated against the relations this connection
            # actually registered: anything else -- including SQL-shaped names
            # -- is simply an unknown table, and nothing unvalidated ever
            # reaches the interpolations below.
            browsable = _browsable_relations(connection)
            if table_name not in browsable:
                raise HTTPException(
                    status_code=404,
                    detail=f"unknown catalog table {table_name!r}; one of: {', '.join(browsable)}",
                )
            quoted_table = _catalog.quoted_identifier(table_name)
            count_row = connection.execute(f"SELECT count(*) FROM {quoted_table}").fetchone()
            return CatalogTableSummaryResponse(
                row_count=int(count_row[0]) if count_row is not None else 0,
                columns=_catalog.fetched_json_safe_rows(
                    connection.execute(f"SUMMARIZE SELECT * FROM {quoted_table}")
                ),
            )

    return router
