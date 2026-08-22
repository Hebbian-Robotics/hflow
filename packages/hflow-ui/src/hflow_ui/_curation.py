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

Workspace convention (M1): pinned manifests are immutable files at
``<data_root>/manifests/<slug>-<utc timestamp>.parquet`` -- never the
engine's default ``<data_root>/manifest.parquet``, which the CLI's curate
silently overwrites. A pin refuses loudly rather than overwrite anything.
The registry describing them lives in the sidecar (see ``_sidecar``).
"""

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from hflow.curation import CurationReport, curate, open_catalog_connection
from hflow.workspace import Workspace
from hflow_ui import _catalog, _media, _sidecar
from hflow_ui._settings import UiSettings

MANIFESTS_DIRECTORY_NAME = "manifests"

# The views hflow.open_catalog_connection registers, in browsing order.
CATALOG_TABLE_NAMES = (
    "episodes",
    "episodes_latest",
    "episodes_raw",
    "check_runs",
    "measurements",
    "measurements_latest",
    "tags",
    "intervals",
)

_TIMESTAMPTZ_TYPE = "TIMESTAMP WITH TIME ZONE"


class PreviewRequest(BaseModel):
    sql: str
    limit: int = Field(default=100, ge=1, le=1000)
    stats: bool = False


class ReportRequest(BaseModel):
    sql: str


class PinRequest(BaseModel):
    sql: str
    name: str = Field(min_length=1)
    description: str = ""


class SavedQueryCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    sql: str


class SavedQueryUpdateRequest(BaseModel):
    name: str | None = None
    sql: str | None = None


def slugified_manifest_name(raw_name: str) -> str:
    """The user-given name as a filename slug: lowercase, [a-z0-9-], dashes collapsed."""
    return re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")


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


def _sidecar_refusal(error: _sidecar.SidecarError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.detail)


def _open_constrained_connection_or_refuse(data_root: str) -> duckdb.DuckDBPyConnection:
    """The connection user SQL runs on. Its configuration is locked at open,
    so the M0 ``SET TimeZone`` pin cannot apply here; timestamp columns are
    instead rendered to UTC ISO text in SQL (``_timestamp_replace_clause``)."""
    try:
        return open_catalog_connection(Workspace.parse(data_root).catalog_root, constrained=True)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _open_live_connection_or_refuse(data_root: str) -> duckdb.DuckDBPyConnection:
    """An unconstrained (UTC-pinned) connection for the server's OWN queries."""
    try:
        return _catalog.open_workspace_connection(data_root)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _described_columns(
    connection: duckdb.DuckDBPyConnection, user_sql: str
) -> list[dict[str, str]]:
    described_rows = connection.execute(f"DESCRIBE SELECT * FROM ({user_sql})").fetchall()
    return [{"name": str(row[0]), "type": str(row[1])} for row in described_rows]


def _timestamp_replace_clause(columns: list[dict[str, str]]) -> str:
    """A ``* REPLACE (...)`` clause rendering TIMESTAMPTZ results as ISO UTC text.

    Materializing a TIMESTAMPTZ into Python requires pytz (deliberately not a
    dependency), and the locked connection cannot ``SET TimeZone`` -- so the
    rendering converts to UTC in SQL (``AT TIME ZONE 'UTC'`` yields the naive
    UTC wall time) and appends the offset literally. A TIMESTAMPTZ nested
    inside a LIST/STRUCT is rendered whole via CAST to text.
    """
    replacements: list[str] = []
    for column in columns:
        quoted_name = _catalog.quoted_identifier(column["name"])
        if column["type"] == _TIMESTAMPTZ_TYPE:
            replacements.append(
                f"strftime({quoted_name} AT TIME ZONE 'UTC', '%Y-%m-%dT%H:%M:%S.%f') "
                f"|| '+00:00' AS {quoted_name}"
            )
        elif _TIMESTAMPTZ_TYPE in column["type"]:
            replacements.append(f"CAST({quoted_name} AS VARCHAR) AS {quoted_name}")
    return f"REPLACE ({', '.join(replacements)})" if replacements else ""


def run_preview(
    connection: duckdb.DuckDBPyConnection, user_sql: str, *, limit: int, include_stats: bool
) -> dict[str, object]:
    """Preview rows, the full count, and (optionally) SUMMARIZE column stats."""
    columns = _described_columns(connection, user_sql)
    replace_clause = _timestamp_replace_clause(columns)
    select_head = f"SELECT * {replace_clause}" if replace_clause else "SELECT *"
    rows = _catalog.fetched_json_safe_rows(
        connection.execute(f"{select_head} FROM ({user_sql}) LIMIT ?", [limit])
    )
    count_row = connection.execute(f"SELECT count(*) FROM ({user_sql})").fetchone()
    row_count = int(count_row[0]) if count_row is not None else 0
    column_stats = (
        _catalog.fetched_json_safe_rows(connection.execute(f"SUMMARIZE SELECT * FROM ({user_sql})"))
        if include_stats
        else None
    )
    return {
        "columns": columns,
        "rows": rows,
        "row_count": row_count,
        "truncated": row_count > len(rows),
        "column_stats": column_stats,
        "sql": f"SELECT * FROM ({user_sql}) LIMIT {limit}",
    }


def _curated_or_refused(data_root: str, user_sql: str, *, output: Path | None) -> CurationReport:
    try:
        return curate(
            Workspace.parse(data_root).catalog_root, user_sql, output=output, constrained=True
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except duckdb.Error as error:
        raise _bad_sql_refusal(error) from error


def _coverage_entries(report: CurationReport) -> tuple[_sidecar.CoverageEntry, ...]:
    return tuple(
        _sidecar.CoverageEntry(
            check_name=entry.check_name,
            episodes_ran=entry.episodes_ran,
            total_episodes=entry.total_episodes,
            fraction=entry.fraction,
        )
        for entry in report.coverage
    )


def create_curation_router(settings: UiSettings) -> APIRouter:
    """Every M1 curation-studio route, closed over one launch's settings."""
    router = APIRouter(prefix="/api/v1")

    def refuse_when_read_only() -> None:
        if settings.read_only:
            raise HTTPException(
                status_code=403,
                detail="this workspace UI is running read-only; "
                "pinning manifests and editing saved queries are disabled",
            )

    def loaded_sidecar_state() -> _sidecar.SidecarState:
        try:
            return _sidecar.load_sidecar_state(settings.data_root)
        except _sidecar.SidecarError as error:
            raise _sidecar_refusal(error) from error

    def stored_sidecar_state(state: _sidecar.SidecarState) -> None:
        try:
            _sidecar.store_sidecar_state(settings.data_root, state)
        except _sidecar.SidecarError as error:
            raise _sidecar_refusal(error) from error

    def local_data_root_or_refuse() -> Path:
        try:
            return _sidecar.local_data_root(settings.data_root)
        except _sidecar.SidecarError as error:
            raise _sidecar_refusal(error) from error

    @router.post("/curation/preview")
    def run_curation_preview(request: PreviewRequest) -> JSONResponse:
        user_sql = _stripped_sql_or_refuse(request.sql)
        connection = _open_constrained_connection_or_refuse(settings.data_root)
        try:
            payload = run_preview(
                connection, user_sql, limit=request.limit, include_stats=request.stats
            )
        except duckdb.Error as error:
            raise _bad_sql_refusal(error) from error
        finally:
            connection.close()
        return JSONResponse(payload)

    @router.post("/curation/report")
    def run_curation_report(request: ReportRequest) -> JSONResponse:
        user_sql = _stripped_sql_or_refuse(request.sql)
        report = _curated_or_refused(settings.data_root, user_sql, output=None)
        return JSONResponse(
            {
                "row_count": report.row_count,
                "total_episodes": report.total_episodes,
                "coverage": [entry.to_json_dict() for entry in _coverage_entries(report)],
            }
        )

    @router.post("/curation/pin")
    def pin_manifest(request: PinRequest) -> JSONResponse:
        refuse_when_read_only()
        user_sql = _stripped_sql_or_refuse(request.sql)
        manifest_slug = slugified_manifest_name(request.name)
        if not manifest_slug:
            raise HTTPException(
                status_code=400, detail="name must contain at least one letter or digit"
            )
        # Load (and thereby validate) the sidecar BEFORE writing the manifest,
        # so a corrupt registry never strands an unregistered manifest file.
        state = loaded_sidecar_state()
        manifests_directory = local_data_root_or_refuse() / MANIFESTS_DIRECTORY_NAME
        manifest_file = manifests_directory / (f"{manifest_slug}-{_manifest_timestamp()}.parquet")
        if manifest_file.exists():
            raise HTTPException(
                status_code=409,
                detail=f"manifest file {manifest_file.name} already exists; "
                "pinned manifests are immutable and never overwritten -- retry the pin",
            )
        report = _curated_or_refused(settings.data_root, user_sql, output=manifest_file)
        entry = _sidecar.PinnedManifest(
            manifest_id=uuid.uuid4().hex,
            name=request.name,
            description=request.description,
            sql=user_sql,
            manifest_path=f"{MANIFESTS_DIRECTORY_NAME}/{manifest_file.name}",
            row_count=report.row_count,
            total_episodes=report.total_episodes,
            coverage=_coverage_entries(report),
            created_at=_utc_now_iso(),
        )
        stored_sidecar_state(
            _sidecar.SidecarState(
                saved_queries=state.saved_queries, manifests=(*state.manifests, entry)
            )
        )
        return JSONResponse(entry.to_json_dict())

    @router.get("/manifests")
    def list_manifests() -> JSONResponse:
        state = loaded_sidecar_state()
        newest_first = sorted(state.manifests, key=lambda entry: entry.created_at, reverse=True)
        return JSONResponse({"manifests": [entry.to_json_dict() for entry in newest_first]})

    @router.get("/manifests/{manifest_id}/download")
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
        manifest_file = local_data_root_or_refuse() / entry.manifest_path
        try:
            # The same strict-resolve + containment check media serving uses:
            # even a hand-edited registry path can only serve workspace files.
            resolved_file = _media.resolve_served_file(
                str(manifest_file), data_root=settings.data_root
            )
        except _media.MediaResolutionError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error
        return _media.served_file_response(
            resolved_file, attachment_filename=Path(entry.manifest_path).name
        )

    @router.get("/queries")
    def list_saved_queries() -> JSONResponse:
        state = loaded_sidecar_state()
        return JSONResponse({"queries": [entry.to_json_dict() for entry in state.saved_queries]})

    @router.post("/queries")
    def create_saved_query(request: SavedQueryCreateRequest) -> JSONResponse:
        refuse_when_read_only()
        query_name = request.name.strip()
        if not query_name:
            raise HTTPException(status_code=400, detail="name must be non-empty")
        entry = _sidecar.SavedQuery(
            query_id=uuid.uuid4().hex,
            name=query_name,
            sql=_stripped_sql_or_refuse(request.sql),
            updated_at=_utc_now_iso(),
        )
        state = loaded_sidecar_state()
        stored_sidecar_state(
            _sidecar.SidecarState(
                saved_queries=(*state.saved_queries, entry), manifests=state.manifests
            )
        )
        return JSONResponse(entry.to_json_dict())

    @router.put("/queries/{query_id}")
    def update_saved_query(query_id: str, request: SavedQueryUpdateRequest) -> JSONResponse:
        refuse_when_read_only()
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
        updated_entry = _sidecar.SavedQuery(
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
        return JSONResponse(updated_entry.to_json_dict())

    @router.delete("/queries/{query_id}", status_code=204)
    def delete_saved_query(query_id: str) -> Response:
        refuse_when_read_only()
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
    def list_catalog_tables() -> JSONResponse:
        connection = _open_live_connection_or_refuse(settings.data_root)
        try:
            kind_rows = connection.execute(
                "SELECT table_name, table_type FROM information_schema.tables"
            ).fetchall()
            kind_by_name = {
                str(table_name): ("view" if str(table_type).upper() == "VIEW" else "table")
                for table_name, table_type in kind_rows
            }
            tables = [
                {
                    "name": table_name,
                    "kind": kind_by_name.get(table_name, "view"),
                    "columns": [
                        {"name": str(row[0]), "type": str(row[1])}
                        for row in connection.execute(
                            f"DESCRIBE {_catalog.quoted_identifier(table_name)}"
                        ).fetchall()
                    ],
                }
                for table_name in CATALOG_TABLE_NAMES
            ]
        finally:
            connection.close()
        return JSONResponse({"tables": tables})

    @router.get("/catalog/tables/{table_name}/summary")
    def read_catalog_table_summary(table_name: str) -> JSONResponse:
        # Identifier-validated against the fixed table list: anything else --
        # including SQL-shaped names -- is simply an unknown table.
        if table_name not in CATALOG_TABLE_NAMES:
            raise HTTPException(
                status_code=404,
                detail=f"unknown catalog table {table_name!r}; "
                f"one of: {', '.join(CATALOG_TABLE_NAMES)}",
            )
        quoted_table = _catalog.quoted_identifier(table_name)
        connection = _open_live_connection_or_refuse(settings.data_root)
        try:
            count_row = connection.execute(f"SELECT count(*) FROM {quoted_table}").fetchone()
            row_count = int(count_row[0]) if count_row is not None else 0
            summary_rows = _catalog.fetched_json_safe_rows(
                connection.execute(f"SUMMARIZE SELECT * FROM {quoted_table}")
            )
        finally:
            connection.close()
        return JSONResponse({"row_count": row_count, "columns": summary_rows})

    return router
