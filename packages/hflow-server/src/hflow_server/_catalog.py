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
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Literal, TypeVar
from urllib.parse import quote

import duckdb
from pydantic import BaseModel

from hflow.app import ARTIFACT_MEASUREMENT_KEY_PREFIX, MEDIA_CONTACT_SHEET_STEP_NAME
from hflow.curation import open_catalog_connection
from hflow.workspace import Workspace
from hflow_server._contract import (
    CategoricalColumnStats,
    ColumnDescriptor,
    DossierEpisode,
    EpisodeCheckRunRecord,
    EpisodeColumnStats,
    EpisodeDossierResponse,
    EpisodeFacetsResponse,
    EpisodeIntervalRecord,
    EpisodeMeasurementRecord,
    EpisodeMediaArtifact,
    EpisodePageResponse,
    EpisodeStatsResponse,
    EpisodeStatus,
    EpisodeTagRecord,
    EpisodeTimelineResponse,
    NumericColumnStats,
    NumericHistogramBucket,
    SuccessFilterValue,
    TimelineInterval,
    TimelineMeasurement,
    ValueCount,
)
from hflow_server._media import is_uri_servable

# The faceted columns, owned by the response model itself so the served keys
# and the columns actually counted can never diverge.
_FACET_COLUMN_NAMES = tuple(EpisodeFacetsResponse.model_fields)
_SEARCHED_COLUMN_NAMES = ("episode_id", "task", "operator")

# %z renders the locked-UTC offset as "+00" on DuckDB 1.5.5, which JS
# Date.parse rejects and the frontend's offset-stripping regex misses; the
# connection is pinned to UTC, so render the wall time and append the offset
# literally -- matching _curation._timestamp_replace_clause's "+00:00".
_ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
_ISO_UTC_OFFSET_SUFFIX = "+00:00"


class UnknownOrderColumnError(ValueError):
    """``order_by`` named a column the live episodes view does not have."""


def open_workspace_connection(data_root: str) -> duckdb.DuckDBPyConnection:
    """One fresh connection over ``<data_root>/catalog`` (see the module note)."""
    connection = open_catalog_connection(Workspace.parse(data_root).catalog_root)
    connection.execute("SET TimeZone = 'UTC'")
    return connection


def utc_iso_text(timestamp_expression: str, alias: str) -> str:
    """SQL rendering a (UTC-pinned) timestamp expression as ISO-8601 text.

    ``timestamp_expression`` and ``alias`` are code-owned constants, never
    user input.
    """
    return (
        f"strftime({timestamp_expression}, '{_ISO_TIMESTAMP_FORMAT}') "
        f"|| '{_ISO_UTC_OFFSET_SUFFIX}' AS {alias}"
    )


def _recorded_at_as_iso_text(qualified_column: str = "recorded_at") -> str:
    return utc_iso_text(qualified_column, "recorded_at")


def json_safe_value(value: object) -> object:
    """One DuckDB cell as a JSON-legal value.

    Datetimes become ISO-8601 strings (a safety net -- timestamp columns are
    already rendered to TEXT in SQL) and NaN/inf doubles become null: both
    are illegal in JSON and would otherwise poison the whole payload. The
    remaining branches exist for the curation studio, where arbitrary user
    SELECTs can materialize types JSON cannot carry (DECIMAL literals,
    BLOBs, INTERVALs, nested LISTs/STRUCTs): containers are converted
    element-wise and anything else is rendered as text -- a legal query must
    never 500 over its result types.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return json_safe_value(float(value))
    if isinstance(value, list | tuple):
        return [json_safe_value(element) for element in value]
    if isinstance(value, dict):
        return {str(key): json_safe_value(element) for key, element in value.items()}
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", errors="replace")
    return str(value)


def fetched_json_safe_rows(executed_query: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    column_names = [str(column[0]) for column in executed_query.description or []]
    return [
        {name: json_safe_value(cell) for name, cell in zip(column_names, row, strict=True)}
        for row in executed_query.fetchall()
    ]


_ContractRecord = TypeVar("_ContractRecord", bound=BaseModel)


def _validated_records(
    record_model: type[_ContractRecord], executed_query: duckdb.DuckDBPyConnection
) -> list[_ContractRecord]:
    """A fixed-column query's rows as contract records.

    Rows pass through :func:`json_safe_value` first, so a NaN double is
    already null by the time the model sees it.
    """
    return [record_model.model_validate(row) for row in fetched_json_safe_rows(executed_query)]


@dataclass(frozen=True)
class EpisodeListFilters:
    """Parsed /api/v1/episodes filter params -- values only, never SQL.

    ``status`` and ``success`` keep the refined types the HTTP boundary
    already parsed them into: this layer never re-checks them, and a caller
    cannot hand it a spelling the SQL below would silently match nothing for.
    """

    tasks: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    embodiments: tuple[str, ...] = ()
    status: EpisodeStatus | None = None
    success: SuccessFilterValue | None = None
    search: str | None = None


def episode_status_for_quarantine_flag(quarantined: object) -> EpisodeStatus:
    """One episode's status derived from its stored ``quarantined`` flag.

    hflow.curation owns the CANONICAL rule as SQL -- the wide ``episodes``
    view's ``CASE WHEN quarantined THEN 'quarantined' ELSE 'ok' END``. The
    dossier reads ``episodes_latest``, which carries the raw flag instead of
    that derived column, so this is the server's single Python restatement of the
    rule; every path that needs a status from a flag calls here.
    """
    return "quarantined" if quarantined else "ok"


def quoted_identifier(column_name: str) -> str:
    return '"' + column_name.replace('"', '""') + '"'


def _escaped_like_fragment(raw_value: str) -> str:
    """User text made literal inside a LIKE pattern (backslash-escaped)."""
    return raw_value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _quoted_sql_literal(value: str) -> str:
    """One string value as a single-quoted SQL literal (internal quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class _CompiledFilters:
    """The WHERE conditions in two parallel forms plus the bind values.

    ``executed_conditions`` carry ``?`` placeholders bound by ``parameters``;
    ``display_conditions`` inline the same values as quoted literals. Building
    the display form here (rather than by splitting rendered SQL on '?') means
    an order_by identifier that itself contains '?' can never be miscounted as
    a placeholder.
    """

    executed_conditions: list[str]
    display_conditions: list[str]
    parameters: list[str]

    def executed_where(self) -> str:
        return (
            (" WHERE " + " AND ".join(self.executed_conditions)) if self.executed_conditions else ""
        )

    def display_where(self) -> str:
        return (
            (" WHERE " + " AND ".join(self.display_conditions)) if self.display_conditions else ""
        )


def _compiled_conditions(filters: EpisodeListFilters) -> _CompiledFilters:
    executed_conditions: list[str] = []
    display_conditions: list[str] = []
    parameters: list[str] = []
    exact_match_columns = (
        ("task", filters.tasks),
        ("operator", filters.operators),
        ("embodiment", filters.embodiments),
    )
    for column_name, values in exact_match_columns:
        if values:
            quoted_column = quoted_identifier(column_name)
            placeholders = ", ".join("?" for _ in values)
            executed_conditions.append(f"{quoted_column} IN ({placeholders})")
            inlined = ", ".join(_quoted_sql_literal(value) for value in values)
            display_conditions.append(f"{quoted_column} IN ({inlined})")
            parameters.extend(values)
    if filters.status is not None:
        executed_conditions.append('"status" = ?')
        display_conditions.append(f'"status" = {_quoted_sql_literal(filters.status)}')
        parameters.append(filters.status)
    if filters.success is not None:
        # Stored success is a stringified boolean whose casing varies by the
        # recording producer; the filter accepts "true"/"false" regardless.
        executed_conditions.append('lower("success") = ?')
        display_conditions.append(f'lower("success") = {_quoted_sql_literal(filters.success)}')
        parameters.append(filters.success)
    if filters.search:
        like_pattern = "%" + _escaped_like_fragment(filters.search) + "%"
        pattern_literal = _quoted_sql_literal(like_pattern)
        executed_disjuncts = " OR ".join(
            f"{quoted_identifier(name)} ILIKE ? ESCAPE '\\'" for name in _SEARCHED_COLUMN_NAMES
        )
        display_disjuncts = " OR ".join(
            f"{quoted_identifier(name)} ILIKE {pattern_literal} ESCAPE '\\'"
            for name in _SEARCHED_COLUMN_NAMES
        )
        executed_conditions.append("(" + executed_disjuncts + ")")
        display_conditions.append("(" + display_disjuncts + ")")
        parameters.extend([like_pattern] * len(_SEARCHED_COLUMN_NAMES))
    return _CompiledFilters(
        executed_conditions=executed_conditions,
        display_conditions=display_conditions,
        parameters=parameters,
    )


def described_episode_columns(connection: duckdb.DuckDBPyConnection) -> list[ColumnDescriptor]:
    """The wide view's live columns."""
    return [
        ColumnDescriptor(name=str(row[0]), type=str(row[1]))
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
) -> EpisodePageResponse:
    """One filtered, ordered page plus the total over the SAME filters."""
    columns = described_episode_columns(connection)
    live_column_names = {column.name for column in columns}
    if order_by not in live_column_names:
        raise UnknownOrderColumnError(
            f"unknown order_by column {order_by!r}; order by one of the episodes view's "
            "columns (the 'columns' field of this endpoint lists them)"
        )
    compiled = _compiled_conditions(filters)
    direction = "DESC" if descending else "ASC"
    # episode_id is unique in the wide view, so it is a deterministic
    # tiebreaker: without it, ordering by any column with duplicate values
    # (task, status, ...) leaves ties unstable across DuckDB's per-query
    # parallel sort, so successive OFFSET pages could overlap or drop rows.
    order_clause = f'ORDER BY {quoted_identifier(order_by)} {direction}, "episode_id" ASC'
    executed_tail = (
        f"FROM episodes{compiled.executed_where()} {order_clause} LIMIT {limit} OFFSET {offset}"
    )
    display_tail = (
        f"FROM episodes{compiled.display_where()} {order_clause} LIMIT {limit} OFFSET {offset}"
    )
    # The executed form renders recorded_at to ISO text in SQL (see the module
    # note); the displayed form stays the logical query a user would write.
    executed_sql = f"SELECT * REPLACE ({_recorded_at_as_iso_text()}) {executed_tail}"
    display_sql = f"SELECT * {display_tail}"
    rows = fetched_json_safe_rows(connection.execute(executed_sql, compiled.parameters))
    count_row = connection.execute(
        f"SELECT count(*) FROM episodes{compiled.executed_where()}", compiled.parameters
    ).fetchone()
    total = int(count_row[0]) if count_row is not None else 0
    return EpisodePageResponse(rows=rows, total=total, columns=columns, sql=display_sql)


def query_episode_facets(connection: duckdb.DuckDBPyConnection) -> EpisodeFacetsResponse:
    """Facet value counts over the wide episodes view; NULL buckets skipped."""
    facets: dict[str, list[ValueCount]] = {}
    for facet_column_name in _FACET_COLUMN_NAMES:
        quoted_column = quoted_identifier(facet_column_name)
        value_counts = connection.execute(
            f"SELECT {quoted_column} AS value, count(*) AS value_count FROM episodes "
            f"WHERE {quoted_column} IS NOT NULL "
            "GROUP BY 1 ORDER BY value_count DESC, value ASC"
        ).fetchall()
        facets[facet_column_name] = [
            ValueCount(value=str(value), count=int(count)) for value, count in value_counts
        ]
    return EpisodeFacetsResponse.model_validate(facets)


# /api/v1/episodes/stats shape knobs: ~12 histogram buckets per numeric
# column, top 8 values per categorical column (the remainder is other_count),
# and "low-cardinality" capped so id-like columns never masquerade as facets.
HISTOGRAM_BUCKET_COUNT = 12
TOP_VALUE_LIMIT = 8
LOW_CARDINALITY_LIMIT = 32

_NUMERIC_STAT_TYPES = frozenset(
    {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "FLOAT",
        "DOUBLE",
    }
)
_CATEGORICAL_STAT_TYPES = frozenset({"VARCHAR", "BOOLEAN"})

# Which mini-distribution a column earns: the same two words the served
# models discriminate on (_contract.NumericColumnStats.kind /
# CategoricalColumnStats.kind), so the two dispatches below are checked
# against the closed set rather than against a bare string.
_StatKind = Literal["numeric", "categorical"]


def _stat_kind(duckdb_type: str) -> _StatKind | None:
    """ "numeric"/"categorical" for distributable column types, else ``None``
    (timestamps, JSON blobs, and nested types have no mini-distribution)."""
    normalized_type = duckdb_type.upper()
    if normalized_type in _NUMERIC_STAT_TYPES or normalized_type.startswith("DECIMAL"):
        return "numeric"
    if normalized_type in _CATEGORICAL_STAT_TYPES:
        return "categorical"
    return None


@dataclass(frozen=True)
class _NumericColumnPlan:
    """One numeric column that earned a histogram, with its bucket geometry."""

    name: str
    minimum: float
    maximum: float

    @property
    def bucket_width(self) -> float:
        return (self.maximum - self.minimum) / HISTOGRAM_BUCKET_COUNT


@dataclass(frozen=True)
class _CategoricalColumnPlan:
    """One low-cardinality column that earned a top-values breakdown."""

    name: str
    non_null_count: int


def query_episode_stats(
    connection: duckdb.DuckDBPyConnection, filters: EpisodeListFilters
) -> EpisodeStatsResponse:
    """Per-column mini-distributions over the CURRENT filter set.

    Reuses the episode list's filter compilation (one source of truth), so
    the sparkbars always describe exactly the rows the table shows. Two
    scans total: one aggregate pass classifying every candidate column
    (skipping degenerate ones -- all NULL, a single value, NaN/inf-poisoned
    numerics, id-like all-unique or over-the-cap categoricals), then one
    UNION ALL query computing every surviving column's histogram buckets or
    top values against a shared filtered CTE.
    """
    compiled = _compiled_conditions(filters)
    parameters = compiled.parameters
    where_sql = compiled.executed_where()
    candidate_columns = [
        (column.name, kind)
        for column in described_episode_columns(connection)
        if (kind := _stat_kind(column.type)) is not None
    ]
    if not candidate_columns:
        return EpisodeStatsResponse(columns=[])

    aggregate_expressions: list[str] = []
    for column_name, kind in candidate_columns:
        quoted_column = quoted_identifier(column_name)
        if kind == "numeric":
            aggregate_expressions.extend(
                (
                    f"count({quoted_column})",
                    f"min(CAST({quoted_column} AS DOUBLE))",
                    f"max(CAST({quoted_column} AS DOUBLE))",
                )
            )
        else:
            aggregate_expressions.extend(
                (f"count({quoted_column})", f"count(DISTINCT {quoted_column})")
            )
    aggregate_row = connection.execute(
        f"SELECT {', '.join(aggregate_expressions)} FROM episodes{where_sql}", parameters
    ).fetchone()
    if aggregate_row is None:
        return EpisodeStatsResponse(columns=[])

    plans: list[_NumericColumnPlan | _CategoricalColumnPlan] = []
    value_index = 0
    for column_name, kind in candidate_columns:
        if kind == "numeric":
            non_null_count, minimum, maximum = aggregate_row[value_index : value_index + 3]
            value_index += 3
            if int(non_null_count or 0) == 0 or minimum is None or maximum is None:
                continue
            minimum, maximum = float(minimum), float(maximum)
            # NaN/inf values poison min/max (NaN sorts above everything in
            # DuckDB), so a non-finite bound marks the whole column degenerate.
            # The span (max - min) can itself overflow to inf even when both
            # bounds are finite (e.g. -1.7e308 and 1.7e308); an inf bucket
            # width would be interpolated as the bare token "inf" into the
            # histogram SQL, so require a finite span too.
            if (
                not (math.isfinite(minimum) and math.isfinite(maximum))
                or not math.isfinite(maximum - minimum)
                or minimum >= maximum
            ):
                continue
            plans.append(_NumericColumnPlan(name=column_name, minimum=minimum, maximum=maximum))
        else:
            non_null_count, distinct_count = aggregate_row[value_index : value_index + 2]
            value_index += 2
            non_null_count, distinct_count = int(non_null_count or 0), int(distinct_count or 0)
            if distinct_count < 2 or distinct_count > LOW_CARDINALITY_LIMIT:
                continue
            if distinct_count == non_null_count and distinct_count > 2:
                # Every value unique: an identifier, not a distribution.
                continue
            plans.append(_CategoricalColumnPlan(name=column_name, non_null_count=non_null_count))
    if not plans:
        return EpisodeStatsResponse(columns=[])

    union_branches: list[str] = []
    for plan in plans:
        quoted_column = quoted_identifier(plan.name)
        name_literal = _quoted_sql_literal(plan.name)
        if isinstance(plan, _NumericColumnPlan):
            # Bounds are data-derived finite floats (never user input), so
            # their repr()s are safe SQL literals.
            union_branches.append(
                f"SELECT {name_literal} AS column_name, "
                f"least(CAST(floor((CAST({quoted_column} AS DOUBLE) - {plan.minimum!r}) "
                f"/ {plan.bucket_width!r}) AS BIGINT), {HISTOGRAM_BUCKET_COUNT - 1}) "
                "AS bucket_index, "
                "CAST(NULL AS VARCHAR) AS value, count(*) AS bucket_count "
                f"FROM filtered WHERE {quoted_column} IS NOT NULL GROUP BY 2"
            )
        else:
            union_branches.append(
                "SELECT * FROM ("
                f"SELECT {name_literal} AS column_name, CAST(NULL AS BIGINT) AS bucket_index, "
                f"CAST({quoted_column} AS VARCHAR) AS value, count(*) AS bucket_count "
                f"FROM filtered WHERE {quoted_column} IS NOT NULL "
                f"GROUP BY 3 ORDER BY bucket_count DESC, value ASC LIMIT {TOP_VALUE_LIMIT})"
            )
    # One query for every column: the shared CTE binds the filter parameters
    # exactly once and each branch aggregates the same filtered rows.
    distribution_rows = connection.execute(
        f"WITH filtered AS (SELECT * FROM episodes{where_sql})\n"
        + "\nUNION ALL\n".join(union_branches),
        parameters,
    ).fetchall()

    bucket_counts_by_column: dict[str, dict[int, int]] = {}
    value_counts_by_column: dict[str, list[tuple[str, int]]] = {}
    for column_name, bucket_index, value, count in distribution_rows:
        if bucket_index is not None:
            bucket_counts_by_column.setdefault(str(column_name), {})[int(bucket_index)] = int(count)
        else:
            value_counts_by_column.setdefault(str(column_name), []).append((str(value), int(count)))

    stat_columns: list[EpisodeColumnStats] = []
    for plan in plans:
        if isinstance(plan, _NumericColumnPlan):
            bucket_counts = bucket_counts_by_column.get(plan.name, {})
            stat_columns.append(
                NumericColumnStats(
                    name=plan.name,
                    buckets=[
                        NumericHistogramBucket(
                            lo=plan.minimum + index * plan.bucket_width,
                            hi=(
                                plan.maximum
                                if index == HISTOGRAM_BUCKET_COUNT - 1
                                else plan.minimum + (index + 1) * plan.bucket_width
                            ),
                            count=bucket_counts.get(index, 0),
                        )
                        for index in range(HISTOGRAM_BUCKET_COUNT)
                    ],
                )
            )
        else:
            # UNION ALL guarantees no cross-branch order; re-rank here.
            top_values = sorted(
                value_counts_by_column.get(plan.name, ()),
                key=lambda entry: (-entry[1], entry[0]),
            )
            stat_columns.append(
                CategoricalColumnStats(
                    name=plan.name,
                    values=[ValueCount(value=value, count=count) for value, count in top_values],
                    other_count=plan.non_null_count - sum(count for _value, count in top_values),
                )
            )
    return EpisodeStatsResponse(columns=stat_columns)


def find_media_uri(
    connection: duckdb.DuckDBPyConnection, episode_id: str, artifact_name: str
) -> str | None:
    """The cataloged URI behind one (episode, artifact name), if recorded."""
    row = connection.execute(
        "SELECT value_text FROM measurements_latest "
        "WHERE episode_id = ? AND check_name = ? AND key = ? AND value_text IS NOT NULL",
        [
            episode_id,
            MEDIA_CONTACT_SHEET_STEP_NAME,
            ARTIFACT_MEASUREMENT_KEY_PREFIX + artifact_name,
        ],
    ).fetchone()
    return str(row[0]) if row is not None and row[0] is not None else None


def find_canonical_uri(connection: duckdb.DuckDBPyConnection, episode_id: str) -> str | None:
    """The latest cataloged canonical-file URI for one episode, if known."""
    row = connection.execute(
        "SELECT uri FROM episodes_latest WHERE episode_id = ?", [episode_id]
    ).fetchone()
    return str(row[0]) if row is not None and row[0] is not None else None


def query_latest_run_intervals(
    connection: duckdb.DuckDBPyConnection, episode_id: str
) -> list[EpisodeIntervalRecord]:
    """One episode's intervals from its LATEST run -- the current evidence.

    ``check_version`` rides in from that run's ``check_runs`` row because the
    intervals table does not carry one itself. One owner for this join: the
    dossier and the timeline must never disagree about which run's intervals
    an episode "has".
    """
    return _validated_records(
        EpisodeIntervalRecord,
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
        ),
    )


def query_episode_dossier(
    connection: duckdb.DuckDBPyConnection, episode_id: str, *, data_root: str
) -> EpisodeDossierResponse | None:
    """Everything the episode page shows, or ``None`` when the id is unknown."""
    episode_rows = fetched_json_safe_rows(
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
    episode = DossierEpisode.model_validate(
        {
            **episode_row,
            "status": episode_status_for_quarantine_flag(episode_row.get("quarantined")),
            "quarantine_tags": quarantine_tags,
        }
    )

    measurements = _validated_records(
        EpisodeMeasurementRecord,
        connection.execute(
            "SELECT key, value_double, value_text, value_bool, check_name, check_version, "
            f"{_recorded_at_as_iso_text()} "
            "FROM measurements_latest WHERE episode_id = ? ORDER BY key",
            [episode_id],
        ),
    )
    check_runs = _validated_records(
        EpisodeCheckRunRecord,
        connection.execute(
            "SELECT check_name, check_version, critical, status, duration_s, error, "
            f"{_recorded_at_as_iso_text()}, run_fingerprint "
            "FROM check_runs WHERE episode_id = ? ORDER BY recorded_at DESC, check_name ASC",
            [episode_id],
        ),
    )
    # Intervals and tags are the episode's LATEST run only -- the current
    # evidence.
    intervals = query_latest_run_intervals(connection, episode_id)
    tags = _validated_records(
        EpisodeTagRecord,
        connection.execute(
            f"SELECT t.tag, t.check_name, {_recorded_at_as_iso_text('t.recorded_at')} "
            "FROM tags AS t "
            "JOIN episodes_latest AS e "
            "  ON t.episode_id = e.episode_id AND t.run_fingerprint = e.run_fingerprint "
            "WHERE t.episode_id = ? ORDER BY t.tag",
            [episode_id],
        ),
    )
    history = fetched_json_safe_rows(
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
        [episode_id, MEDIA_CONTACT_SHEET_STEP_NAME, ARTIFACT_MEASUREMENT_KEY_PREFIX + "%"],
    ).fetchall()
    quoted_episode_id = quote(episode_id, safe="")
    media: list[EpisodeMediaArtifact] = []
    for key, artifact_uri in media_rows:
        artifact_name = str(key).removeprefix(ARTIFACT_MEASUREMENT_KEY_PREFIX)
        served_url = (
            f"/api/v1/episodes/{quoted_episode_id}/media/{quote(artifact_name, safe='/')}"
            if is_uri_servable(str(artifact_uri), data_root=data_root)
            else None
        )
        media.append(
            EpisodeMediaArtifact(name=artifact_name, uri=str(artifact_uri), url=served_url)
        )

    canonical_uri = episode_row.get("uri")
    canonical_url = (
        f"/api/v1/episodes/{quoted_episode_id}/canonical"
        if isinstance(canonical_uri, str) and is_uri_servable(canonical_uri, data_root=data_root)
        else None
    )
    return EpisodeDossierResponse(
        episode=episode,
        measurements=measurements,
        check_runs=check_runs,
        intervals=intervals,
        tags=tags,
        history=history,
        media=media,
        canonical_url=canonical_url,
    )


NANOSECONDS_PER_SECOND = 1_000_000_000

# Timeline span derivation. Interval times are nanoseconds of LOG time, so an
# episode with intervals carries its own axis; an episode without them can
# still have a length if some check measured one. A measurement key naming a
# duration supplies that length: the token after the key's last '_' picks the
# unit, and a duration key with NO recognized suffix at all
# (``episode_duration``) is read as SECONDS -- the convention every hflow
# example follows.
#
# The two tables below are one fact split in two, and neither may be read
# alone: _UNIT_BY_KEY_SUFFIX owns which suffixes name a dimension at all (and
# what to call it), _NANOSECONDS_PER_DURATION_UNIT owns which of those
# dimensions are TIMES and how long one is. Every key of the second is a key
# of the first. A suffix the first knows and the second does not is a
# NON-time dimension (hz, pct, count, bytes, deg), so a key like
# ``duty_cycle_duration_pct`` measures no length -- reading it as seconds
# would both contradict the "45 %" its own bar is labelled with and stretch
# the episode's axis by 1e9.
_DURATION_KEY_TOKEN = "duration"
_NANOSECONDS_PER_DURATION_UNIT: dict[str, float] = {
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
    "sec": 1e9,
    "secs": 1e9,
    "second": 1e9,
    "seconds": 1e9,
    "min": 6e10,
    "mins": 6e10,
    "minute": 6e10,
    "minutes": 6e10,
}
_DEFAULT_DURATION_UNIT_NANOSECONDS = 1e9

# Units the measurement bars label themselves with, by the same key suffix.
# Absent from this table means "no unit known" -- the bar shows the bare
# number rather than inventing a dimension.
_UNIT_BY_KEY_SUFFIX: dict[str, str] = {
    "ns": "ns",
    "us": "us",
    "ms": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "hz": "Hz",
    "pct": "%",
    "percent": "%",
    "ratio": "ratio",
    "count": "count",
    "bytes": "bytes",
    "mb": "MB",
    "gb": "GB",
    "m": "m",
    "mm": "mm",
    "cm": "cm",
    "km": "km",
    "deg": "deg",
    "rad": "rad",
    "kg": "kg",
    "n": "N",
}


def _measurement_key_suffix(key: str) -> str:
    """The unit-bearing tail of a measurement key (``max_gap_ms`` -> ``ms``)."""
    return key.rsplit("_", 1)[-1].lower() if "_" in key else ""


def _duration_nanoseconds(key: str, value: float) -> float | None:
    """A duration-naming measurement converted to nanoseconds, if it is one.

    ``None`` for anything that is not a length, INCLUDING a key that says
    "duration" but carries a suffix naming another dimension (see the note
    above the tables): a measurement the bars label "45 %" must not also
    claim the episode ran for 45 seconds.
    """
    if _DURATION_KEY_TOKEN not in key.lower() or not math.isfinite(value) or value <= 0:
        return None
    key_suffix = _measurement_key_suffix(key)
    unit_scale = _NANOSECONDS_PER_DURATION_UNIT.get(key_suffix)
    if unit_scale is not None:
        return value * unit_scale
    if key_suffix in _UNIT_BY_KEY_SUFFIX:
        return None
    return value * _DEFAULT_DURATION_UNIT_NANOSECONDS


def _interval_kind(label: str | None, check_name: str | None) -> str:
    """The colour group for one interval label.

    Labels are conventionally ``<kind>:<topic>`` (``gap:/imu``,
    ``joint_discontinuity:/joint_states``), so the prefix is the group. A
    label with no prefix groups by itself; an empty label falls back to the
    check that produced it, which is the only honest grouping left.
    """
    text = label.strip() if label is not None else ""
    if not text:
        return check_name if check_name else "interval"
    prefix = text.split(":", 1)[0].strip()
    return prefix or text


def _relative_seconds(absolute_ns: int | None, span_start_ns: int | None) -> float | None:
    if span_start_ns is None or absolute_ns is None:
        return None
    return (absolute_ns - span_start_ns) / NANOSECONDS_PER_SECOND


def query_episode_timeline(
    connection: duckdb.DuckDBPyConnection, episode_id: str
) -> EpisodeTimelineResponse | None:
    """One episode's time axis, computed server-side (``None`` when unknown).

    The span comes from the latest run's intervals, extended by any duration
    measurement that claims a longer episode; an episode with no intervals but
    a duration measurement gets a zero-based axis; an episode with neither
    gets nulls, and a client says the span is unknown rather than drawing a
    fabricated axis.
    """
    if (
        connection.execute(
            "SELECT 1 FROM episodes_latest WHERE episode_id = ?", [episode_id]
        ).fetchone()
        is None
    ):
        return None

    interval_rows = query_latest_run_intervals(connection, episode_id)
    measurement_rows = connection.execute(
        "SELECT key, value_double FROM measurements_latest "
        "WHERE episode_id = ? AND value_double IS NOT NULL ORDER BY key",
        [episode_id],
    ).fetchall()
    numeric_measurements = [
        (str(key), float(value))
        for key, value in measurement_rows
        # NaN/inf poison a bar chart exactly as they poison JSON: drop them.
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]

    interval_starts = [row.start_ns for row in interval_rows if row.start_ns is not None]
    interval_ends = [row.end_ns for row in interval_rows if row.end_ns is not None]
    # Several duration-ish measurements: the largest wins, because the span
    # must contain every interval AND every claimed duration.
    claimed_durations_ns = [
        duration_ns
        for key, value in numeric_measurements
        if (duration_ns := _duration_nanoseconds(key, value)) is not None
    ]
    longest_claimed_duration_ns = max(claimed_durations_ns) if claimed_durations_ns else None

    start_ns: int | None = None
    end_ns: int | None = None
    if interval_starts:
        start_ns = min(interval_starts)
        end_ns = max([*interval_ends, start_ns])
        if longest_claimed_duration_ns is not None:
            end_ns = max(end_ns, start_ns + int(longest_claimed_duration_ns))
    elif longest_claimed_duration_ns is not None:
        start_ns, end_ns = 0, int(longest_claimed_duration_ns)

    duration_s = (
        (end_ns - start_ns) / NANOSECONDS_PER_SECOND
        if start_ns is not None and end_ns is not None
        else None
    )
    return EpisodeTimelineResponse(
        start_ns=start_ns,
        end_ns=end_ns,
        duration_s=duration_s,
        intervals=[
            TimelineInterval(
                label=row.label,
                start_ns=row.start_ns,
                end_ns=row.end_ns,
                start_s=_relative_seconds(row.start_ns, start_ns),
                end_s=_relative_seconds(row.end_ns, start_ns),
                check_name=row.check_name,
                kind=_interval_kind(row.label, row.check_name),
            )
            for row in interval_rows
        ],
        measurements=[
            TimelineMeasurement(
                key=key, value=value, unit=_UNIT_BY_KEY_SUFFIX.get(_measurement_key_suffix(key))
            )
            for key, value in numeric_measurements
        ],
    )
