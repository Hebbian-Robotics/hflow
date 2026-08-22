"""The query layer: per-request DuckDB connections over one workspace catalog.

Every request opens (and closes) a FRESH connection: the wide ``episodes``
view binds one column per measurement key present at open time (see
``hflow.curation``), so a held connection would never show keys recorded
after startup. Opening is cheap -- the views read local Parquet directly.

Two boundary rules hold everywhere here:

- Filter VALUES travel as DuckDB bind parameters, never string-interpolated.
  The only identifiers ever interpolated are validated against the live
  view's DESCRIBE output (``order_by``) or are literal constants (facet
  columns), then double-quoted.
- ``recorded_at`` leaves DuckDB as ISO-8601 TEXT: materializing a
  TIMESTAMPTZ into Python requires pytz, which hflow deliberately does not
  depend on. The connection is pinned to UTC so the rendering is stable
  across host timezones.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from urllib.parse import quote

import duckdb

from hflow.curation import open_catalog_connection
from hflow.workspace import Workspace
from hflow_ui._media import is_uri_servable

CONTACT_SHEET_CHECK_NAME = "media/contact_sheet"
ARTIFACT_KEY_PREFIX = "artifact/"

_FACET_COLUMN_NAMES = ("task", "operator", "embodiment", "status", "pipeline_version")
_SEARCHED_COLUMN_NAMES = ("episode_id", "task", "operator")

_ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


class UnknownOrderColumnError(ValueError):
    """``order_by`` named a column the live episodes view does not have."""


def open_workspace_connection(data_root: str) -> duckdb.DuckDBPyConnection:
    """One fresh connection over ``<data_root>/catalog`` (see the module note)."""
    connection = open_catalog_connection(Workspace.parse(data_root).catalog_root)
    connection.execute("SET TimeZone = 'UTC'")
    return connection


def _recorded_at_as_iso_text(qualified_column: str = "recorded_at") -> str:
    return f"strftime({qualified_column}, '{_ISO_TIMESTAMP_FORMAT}') AS recorded_at"


def json_safe_value(value: object) -> object:
    """One catalog cell as a JSON-legal value.

    Datetimes become ISO-8601 strings (a safety net -- timestamp columns are
    already rendered to TEXT in SQL) and NaN/inf doubles become null: both
    are illegal in JSON and would otherwise poison the whole payload.
    """
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fetched_json_safe_rows(executed_query: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    column_names = [str(column[0]) for column in executed_query.description or []]
    return [
        {name: json_safe_value(cell) for name, cell in zip(column_names, row, strict=True)}
        for row in executed_query.fetchall()
    ]


@dataclass(frozen=True)
class EpisodeListFilters:
    """Parsed /api/v1/episodes filter params -- values only, never SQL."""

    tasks: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    embodiments: tuple[str, ...] = ()
    status: str | None = None
    success: str | None = None
    search: str | None = None


@dataclass(frozen=True)
class EpisodePage:
    """One page of the wide episodes view, ready to serialize."""

    rows: list[dict[str, object]]
    total: int
    columns: list[dict[str, str]]
    sql: str


def _quoted_identifier(column_name: str) -> str:
    return '"' + column_name.replace('"', '""') + '"'


def _escaped_like_fragment(raw_value: str) -> str:
    """User text made literal inside a LIKE pattern (backslash-escaped)."""
    return raw_value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _compiled_conditions(filters: EpisodeListFilters) -> tuple[list[str], list[str]]:
    conditions: list[str] = []
    parameters: list[str] = []
    exact_match_columns = (
        ("task", filters.tasks),
        ("operator", filters.operators),
        ("embodiment", filters.embodiments),
    )
    for column_name, values in exact_match_columns:
        if values:
            placeholders = ", ".join("?" for _ in values)
            conditions.append(f"{_quoted_identifier(column_name)} IN ({placeholders})")
            parameters.extend(values)
    if filters.status is not None:
        conditions.append('"status" = ?')
        parameters.append(filters.status)
    if filters.success is not None:
        # Stored success is a stringified boolean whose casing varies by the
        # recording producer; the filter accepts "true"/"false" regardless.
        conditions.append('lower("success") = ?')
        parameters.append(filters.success)
    if filters.search:
        like_pattern = "%" + _escaped_like_fragment(filters.search) + "%"
        disjuncts = " OR ".join(
            f"{_quoted_identifier(name)} ILIKE ? ESCAPE '\\'" for name in _SEARCHED_COLUMN_NAMES
        )
        conditions.append("(" + disjuncts + ")")
        parameters.extend([like_pattern] * len(_SEARCHED_COLUMN_NAMES))
    return conditions, parameters


def _rendered_display_sql(parameterized_sql: str, parameters: Sequence[str]) -> str:
    """The compiled SQL with values inlined as quoted literals.

    Execution always uses the parameterized form; this rendering exists so
    the UI can show -- and the user can paste into ``hflow curate`` -- a
    runnable query. Values are single-quoted with internal quotes doubled,
    the same idiom as ``hflow.curation``.
    """
    pieces = parameterized_sql.split("?")
    if len(pieces) != len(parameters) + 1:
        raise ValueError("placeholder count does not match parameter count")
    rendered = pieces[0]
    for parameter_value, following_piece in zip(parameters, pieces[1:], strict=True):
        rendered += "'" + str(parameter_value).replace("'", "''") + "'" + following_piece
    return rendered


def described_episode_columns(connection: duckdb.DuckDBPyConnection) -> list[dict[str, str]]:
    """The wide view's live columns as ``{"name", "type"}`` pairs."""
    return [
        {"name": str(row[0]), "type": str(row[1])}
        for row in connection.execute("DESCRIBE episodes").fetchall()
    ]


def query_episode_page(
    connection: duckdb.DuckDBPyConnection,
    filters: EpisodeListFilters,
    *,
    order_by: str,
    descending: bool,
    limit: int,
    offset: int,
) -> EpisodePage:
    """One filtered, ordered page plus the total over the SAME filters."""
    columns = described_episode_columns(connection)
    live_column_names = {column["name"] for column in columns}
    if order_by not in live_column_names:
        raise UnknownOrderColumnError(
            f"unknown order_by column {order_by!r}; order by one of the episodes view's "
            "columns (the 'columns' field of this endpoint lists them)"
        )
    conditions, parameters = _compiled_conditions(filters)
    where_sql = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    direction = "DESC" if descending else "ASC"
    query_tail = (
        f"FROM episodes{where_sql} "
        f"ORDER BY {_quoted_identifier(order_by)} {direction} LIMIT {limit} OFFSET {offset}"
    )
    # The executed form renders recorded_at to ISO text in SQL (see the module
    # note); the displayed form stays the logical query a user would write.
    executed_sql = f"SELECT * REPLACE ({_recorded_at_as_iso_text()}) {query_tail}"
    display_sql = f"SELECT * {query_tail}"
    rows = _fetched_json_safe_rows(connection.execute(executed_sql, parameters))
    count_row = connection.execute(
        f"SELECT count(*) FROM episodes{where_sql}", parameters
    ).fetchone()
    total = int(count_row[0]) if count_row is not None else 0
    return EpisodePage(
        rows=rows,
        total=total,
        columns=columns,
        sql=_rendered_display_sql(display_sql, parameters),
    )


def query_episode_facets(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, list[dict[str, object]]]:
    """Facet value counts over the wide episodes view; NULL buckets skipped."""
    facets: dict[str, list[dict[str, object]]] = {}
    for facet_column_name in _FACET_COLUMN_NAMES:
        quoted_column = _quoted_identifier(facet_column_name)
        value_counts = connection.execute(
            f"SELECT {quoted_column} AS value, count(*) AS value_count FROM episodes "
            f"WHERE {quoted_column} IS NOT NULL "
            "GROUP BY 1 ORDER BY value_count DESC, value ASC"
        ).fetchall()
        facets[facet_column_name] = [
            {"value": str(value), "count": int(count)} for value, count in value_counts
        ]
    return facets


def find_media_uri(
    connection: duckdb.DuckDBPyConnection, episode_id: str, artifact_name: str
) -> str | None:
    """The cataloged URI behind one (episode, artifact name), if recorded."""
    row = connection.execute(
        "SELECT value_text FROM measurements_latest "
        "WHERE episode_id = ? AND check_name = ? AND key = ? AND value_text IS NOT NULL",
        [episode_id, CONTACT_SHEET_CHECK_NAME, ARTIFACT_KEY_PREFIX + artifact_name],
    ).fetchone()
    return str(row[0]) if row is not None and row[0] is not None else None


def find_canonical_uri(connection: duckdb.DuckDBPyConnection, episode_id: str) -> str | None:
    """The latest cataloged canonical-file URI for one episode, if known."""
    row = connection.execute(
        "SELECT uri FROM episodes_latest WHERE episode_id = ?", [episode_id]
    ).fetchone()
    return str(row[0]) if row is not None and row[0] is not None else None


def query_episode_dossier(
    connection: duckdb.DuckDBPyConnection, episode_id: str, *, data_root: str
) -> dict[str, object] | None:
    """Everything the episode page shows, or ``None`` when the id is unknown."""
    episode_rows = _fetched_json_safe_rows(
        connection.execute(
            f"SELECT * REPLACE ({_recorded_at_as_iso_text()}) "
            "FROM episodes_latest WHERE episode_id = ?",
            [episode_id],
        )
    )
    if not episode_rows:
        return None
    episode_row = episode_rows[0]
    raw_quarantine_tags = episode_row.get("quarantine_tags_json")
    quarantine_tags = (
        [str(tag) for tag in json.loads(str(raw_quarantine_tags))] if raw_quarantine_tags else []
    )
    episode: dict[str, object] = {
        **episode_row,
        "status": "quarantined" if episode_row.get("quarantined") else "ok",
        "quarantine_tags": quarantine_tags,
    }

    measurements = _fetched_json_safe_rows(
        connection.execute(
            "SELECT key, value_double, value_text, value_bool, check_name, check_version, "
            f"{_recorded_at_as_iso_text()} "
            "FROM measurements_latest WHERE episode_id = ? ORDER BY key",
            [episode_id],
        )
    )
    check_runs = _fetched_json_safe_rows(
        connection.execute(
            "SELECT check_name, check_version, critical, status, duration_s, error, "
            f"{_recorded_at_as_iso_text()}, run_fingerprint "
            "FROM check_runs WHERE episode_id = ? ORDER BY recorded_at DESC, check_name ASC",
            [episode_id],
        )
    )
    # Intervals and tags are the episode's LATEST run only -- the current
    # evidence. check_version rides in from that run's check_runs row because
    # the intervals table does not carry one itself.
    intervals = _fetched_json_safe_rows(
        connection.execute(
            """
            SELECT i.label, i.start_ns, i.end_ns, i.check_name, r.check_version
            FROM intervals AS i
            JOIN episodes_latest AS e
              ON i.episode_id = e.episode_id AND i.run_fingerprint = e.run_fingerprint
            LEFT JOIN check_runs AS r
              ON r.episode_id = i.episode_id AND r.run_fingerprint = i.run_fingerprint
                 AND r.check_name = i.check_name
            WHERE i.episode_id = ?
            ORDER BY i.start_ns, i.label
            """,
            [episode_id],
        )
    )
    tags = _fetched_json_safe_rows(
        connection.execute(
            f"SELECT t.tag, t.check_name, {_recorded_at_as_iso_text('t.recorded_at')} "
            "FROM tags AS t "
            "JOIN episodes_latest AS e "
            "  ON t.episode_id = e.episode_id AND t.run_fingerprint = e.run_fingerprint "
            "WHERE t.episode_id = ? ORDER BY t.tag",
            [episode_id],
        )
    )
    history = _fetched_json_safe_rows(
        connection.execute(
            f"SELECT * REPLACE ({_recorded_at_as_iso_text()}) "
            "FROM episodes_raw WHERE episode_id = ? "
            "ORDER BY recorded_at DESC, run_fingerprint DESC",
            [episode_id],
        )
    )

    media_rows = connection.execute(
        "SELECT key, value_text FROM measurements_latest "
        "WHERE episode_id = ? AND check_name = ? AND key LIKE ? AND value_text IS NOT NULL "
        "ORDER BY key",
        [episode_id, CONTACT_SHEET_CHECK_NAME, ARTIFACT_KEY_PREFIX + "%"],
    ).fetchall()
    quoted_episode_id = quote(episode_id, safe="")
    media: list[dict[str, object]] = []
    for key, artifact_uri in media_rows:
        artifact_name = str(key).removeprefix(ARTIFACT_KEY_PREFIX)
        served_url = (
            f"/api/v1/episodes/{quoted_episode_id}/media/{quote(artifact_name, safe='/')}"
            if is_uri_servable(str(artifact_uri), data_root=data_root)
            else None
        )
        media.append({"name": artifact_name, "uri": str(artifact_uri), "url": served_url})

    canonical_uri = episode_row.get("uri")
    canonical_url = (
        f"/api/v1/episodes/{quoted_episode_id}/canonical"
        if isinstance(canonical_uri, str) and is_uri_servable(canonical_uri, data_root=data_root)
        else None
    )
    return {
        "episode": episode,
        "measurements": measurements,
        "check_runs": check_runs,
        "intervals": intervals,
        "tags": tags,
        "history": history,
        "media": media,
        "canonical_url": canonical_url,
    }
