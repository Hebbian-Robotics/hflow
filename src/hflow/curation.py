"""Curation: a SQL query over the catalog, out comes ``manifest.parquet``.

The researcher interface is a SQL query -- "a curation is now a SQL query",
as Dyna's article puts it -- with zero services: DuckDB reads the catalog's
Parquet directly. Nothing here is required -- the catalog is plain Parquet
and users can point DuckDB, pandas, or anything else at it (the
no-obfuscation tenet); this module just registers convenient views and
applies the manifest-last idiom to the output.

Views registered on the connection:

- ``episodes_raw``, ``check_runs``, ``measurements``, ``observations``, ``tags``,
  ``intervals`` -- the long tables, exactly as stored.
- ``episodes_latest`` -- one row per SOURCE RECORDING (most recent append).
  Ranking per source rather than per ``episode_id`` is what makes this the
  current corpus: reprocessing rewrites the canonical file and therefore
  mints a new content-addressed ``episode_id``, so a per-episode ranking
  keeps every superseded generation alive and a plain ``SELECT`` over the
  wide view double-counts every reprocessed recording. Both generations even
  share one ``uri``, because publication overwrites in place, so the older
  row's ``episode_id`` no longer hashes the bytes at its own address.
  Re-running only a CHECK does not change the canonical bytes: that appends
  a second ``run_fingerprint`` under one ``episode_id``, and both rankings
  agree on it. ``coalesce`` covers rows that recorded no ``source_uri``
  (optional on ``Catalog.append_episode``, so a direct
  ``write_canonical_episode`` caller can omit it): such an episode is its
  own source. This view is the only place that expression lives -- every
  consumer, ``stale_episodes`` included, reads it rather than re-deriving
  the window.
- ``measurements_latest`` -- one row per (episode_id, key), most recent by
  the OWNING episode's recorded_at (joined in), not the measurement row's
  own -- the latter can go stale independently of the episode it belongs to.
- ``observations_latest`` -- every timestamped observation field from the
  latest run of each (episode, check), switched as one coherent result.
- ``episodes`` -- the wide view for everyday queries: latest episode rows,
  a ``status`` column (``'quarantined'``/``'unverified'``/``'ok'``), and one numeric column
  per measurement key (booleans as 0/1; text-valued measurements stay in the
  long table). A measurement key claiming an episode column -- or this
  derived ``status`` -- is refused at append time: pivoted beside the real
  column, DuckDB would silently rename it to ``<key>_1``, and ``SELECT task``
  would return the metadata where the measurement was meant. Name keys
  ``<topic>/<metric>`` and the collision cannot arise.

Dataset-level reporting always includes coverage denominators: which checks
ran on what fraction of episodes, because a statistic over half a delivery
must not look like a statistic over all of it.
"""

import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import duckdb

from hflow.catalog import (
    EPISODES_VIEW_STATUS_COLUMN,
    TABLE_COLUMN_DDL,
    episode_status_case_sql,
)
from hflow.format import CATALOG_FORMAT_VERSION
from hflow.ingest_ledger import INGEST_FAILURES_COLUMN_DDL, INGEST_FAILURES_TABLE_NAME
from hflow.steps import RAN_STATUSES
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    StorageRoot,
    is_bucket_url,
    parse_storage_root,
)

_LONG_TABLE_NAMES = (
    "episodes_raw",
    "check_runs",
    "measurements",
    "observations",
    "tags",
    "intervals",
    INGEST_FAILURES_TABLE_NAME,
)

# The status rule rendered for each of the two episodes view shapes. Both come
# from the one builder, so item 5 of #164 -- the wide and narrow paths agreeing
# -- holds by construction rather than by two edits staying in step.
_STATUS_CASE_ALIASED = episode_status_case_sql(
    quarantined_column="e.quarantined", check_runs_relation="check_runs_latest"
)
_TABLE_DIRECTORIES = {
    "episodes_raw": "episodes",
    "check_runs": "check_runs",
    "measurements": "measurements",
    "observations": "observations",
    "tags": "tags",
    "intervals": "intervals",
    # The complement of `episodes`: attempts that produced no row there. Not a
    # member of catalog.TABLE_COLUMN_DDL, whose machinery assumes every table
    # is keyed by (episode_id, run_fingerprint), so it is registered for
    # reading here and written by its own module.
    INGEST_FAILURES_TABLE_NAME: INGEST_FAILURES_TABLE_NAME,
}

# "The check actually ran on this episode", owned by hflow.steps because
# dataset membership asks the same question and the two answers must agree.
_RAN_STATUSES = tuple(status.value for status in RAN_STATUSES)


def _column_ddl_for(directory_name: str) -> str:
    """The stored columns of one catalog directory, for an EMPTY relation.

    An empty table still has to be definable, or every downstream view breaks
    on a workspace that has not recorded that kind of row yet. The episode
    tables' shapes are the catalog's; the failure ledger's is its own.
    """
    if directory_name == INGEST_FAILURES_TABLE_NAME:
        return INGEST_FAILURES_COLUMN_DDL
    return TABLE_COLUMN_DDL[directory_name]


@dataclass(frozen=True)
class CheckCoverage:
    """Coverage denominator for one check across cataloged episodes."""

    check_name: str
    episodes_ran: int
    total_episodes: int

    @property
    def fraction(self) -> float:
        """Fraction of total cataloged episodes this check ran on (0.0 to 1.0)."""
        return self.episodes_ran / self.total_episodes if self.total_episodes else 0.0


@dataclass(frozen=True)
class CurationReport:
    """Outcome of a curation SQL query and manifest generation.

    ``manifest_path`` is a Path for local manifests, a str object URL for
    bucket manifests, and None when ``curate()`` ran with ``output=None``,
    which reports row count and coverage without writing anything.
    """

    manifest_path: Path | str | None
    row_count: int
    total_episodes: int
    coverage: list[CheckCoverage]

    def summary(self) -> str:
        if self.manifest_path is None:
            manifest = "manifest: (not written; dry run)"
        else:
            manifest = f"manifest: {self.manifest_path}"
        lines = [
            f"{manifest} ({self.row_count} rows, from {self.total_episodes} cataloged episodes)"
        ]
        lines.append("coverage (episodes each check ran on):")
        for entry in sorted(self.coverage, key=lambda item: item.check_name):
            lines.append(
                f"  {entry.check_name}: {entry.episodes_ran}/{entry.total_episodes} "
                f"({entry.fraction * 100:.0f}%)"
            )
        if not self.coverage:
            lines.append("  (no check runs recorded)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class _BucketManifestDestination:
    """A manifest headed for an object store, staged locally first."""

    parent_url: str
    object_name: str
    staging_dir: Path


@dataclass(frozen=True)
class _LocalManifestDestination:
    """A manifest headed for a local path, staged in a private sibling dir."""

    final_path: Path
    staging_dir: Path


# None: curate() ran report-only, nothing written.
_ManifestDestination = _BucketManifestDestination | _LocalManifestDestination | None


@dataclass(frozen=True)
class StaleEpisode:
    """One episode whose latest cataloged run predates the current versions."""

    episode_id: str
    uri: str
    source_uri: str | None
    pipeline_version: str
    schema_version: str


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _verify_catalog_format(location: StorageRoot) -> None:
    try:
        found_version = location.read_bytes("format_version").decode().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{location} is not a catalog root (no format_version marker); "
            "expected the location a Catalog was created with, e.g. <data_root>/catalog"
        ) from None
    if found_version != CATALOG_FORMAT_VERSION:
        raise ValueError(
            f"catalog at {location} has format version {found_version!r}; "
            f"this build reads version {CATALOG_FORMAT_VERSION!r}"
        )


def _local_query_root(location: StorageRoot) -> Path:
    """A local directory DuckDB can glob: the root itself, or the synced mirror.

    Bucket catalogs download only the table files their mirror lacks (table
    files are append-only and content-named, so an existing local file is
    final) -- DuckDB then queries plain local Parquet and needs no
    object-store credentials of its own.
    """
    match location:
        case LocalStorageRoot(path=local_root):
            return local_root
        case BucketStorageRoot():
            return location.sync_into_mirror(tuple(_TABLE_DIRECTORIES.values()))


def _apply_connection_constraints(
    connection: duckdb.DuckDBPyConnection, allowed_directories: Sequence[Path]
) -> None:
    """Confine one connection to ``allowed_directories``, then lock it.

    DuckDB's file functions can otherwise read or write any path the process
    reaches, auto-install extensions, and open network resources -- fine when
    the user owns the machine, unacceptable when the SQL is tenant-supplied
    (a hosted, self-serve curation surface). ``allowed_directories`` is a
    read AND write allowlist, so callers must pass only directories the SQL
    may also write into -- never the catalog root itself. Locking the
    configuration last means the SQL cannot lift the restriction.
    """
    if allowed_directories:
        quoted_directories = ", ".join(
            _quote_sql_string(str(Path(directory).resolve())) for directory in allowed_directories
        )
        connection.execute(f"SET allowed_directories = [{quoted_directories}]")
    connection.execute("SET enable_external_access = false")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    connection.execute("SET lock_configuration = true")


def open_catalog_connection(
    catalog_root: "Path | str | StorageRoot", *, constrained: bool = False
) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with all catalog views registered.

    Public on purpose: ``curate()`` is a convenience, not a gate -- take the
    connection and explore. ``catalog_root`` may be a local directory or a
    bucket prefix (``gs://.../catalog``).

    ``constrained=True`` is the service posture for running SQL that is not
    the operator's own (a hosted curation endpoint): the catalog is
    materialized into the connection once at open (so the SQL touches data,
    never the catalog's files -- DuckDB's directory allowlist permits writes
    too, and the append-only catalog must not be writable by tenant SQL),
    extension auto-install/auto-load is off, all other file access is
    refused, and the configuration is locked. The default stays unrestricted
    for local exploration.
    """
    location = parse_storage_root(catalog_root)
    _verify_catalog_format(location)
    root = _local_query_root(location)
    return _open_connection_over_root(root, constrained=constrained, writable_directories=())


def _open_connection_over_root(
    root: Path, *, constrained: bool, writable_directories: Sequence[Path]
) -> duckdb.DuckDBPyConnection:
    """Register the catalog views over one local query root.

    Constrained connections materialize the long tables in memory BEFORE the
    constraints land, so the locked allowlist holds only
    ``writable_directories`` (``curate``'s private manifest staging) -- the
    catalog's own files stay unreachable for reads and, crucially, writes.
    """
    connection = duckdb.connect()
    _register_catalog_relations(
        connection,
        root,
        constrained=constrained,
        writable_directories=writable_directories,
    )
    return connection


def _register_catalog_relations(
    connection: duckdb.DuckDBPyConnection,
    root: Path,
    *,
    constrained: bool,
    writable_directories: Sequence[Path],
) -> None:
    """Register every HFlow catalog relation on an existing connection."""

    for view_name in _LONG_TABLE_NAMES:
        directory_name = _TABLE_DIRECTORIES[view_name]
        table_directory = root / directory_name
        if any(table_directory.glob("*.parquet")):
            pattern = _quote_sql_string(str(table_directory / "*.parquet"))
            if constrained:
                # A real table, not a view: reads happen here, once, while
                # file access is still allowed; tenant SQL later queries the
                # in-memory copy and never touches the catalog's files.
                connection.execute(
                    f"CREATE TABLE {view_name} AS "
                    f"SELECT * FROM read_parquet({pattern}, union_by_name=true)"
                )
            else:
                connection.execute(
                    f"CREATE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet({pattern}, union_by_name=true)"
                )
        else:
            # An empty catalog table: an empty relation with the real schema
            # keeps every downstream view and query definable.
            connection.execute(f"CREATE TABLE {view_name} ({_column_ddl_for(directory_name)})")

    if constrained:
        _apply_connection_constraints(connection, list(writable_directories))

    connection.execute(
        """
        CREATE VIEW episodes_latest AS
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY coalesce(source_uri, episode_id)
                ORDER BY recorded_at DESC, run_fingerprint DESC
            ) AS row_rank FROM episodes_raw
        ) WHERE row_rank = 1
        """
    )
    connection.execute(
        """
        CREATE VIEW measurements_latest AS
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT
                m.* EXCLUDE (recorded_at),
                -- Rank (and report) by the EPISODE's recorded_at, not this
                -- table's own: episodes/<file_stem>.parquet is the only file
                -- create-if-absent guarantees a single writer for, so it is
                -- the sole trustworthy recorded_at once episodes_latest has
                -- already picked a winner. A repair pass that wins the
                -- episodes race can still crash before reaching measurements
                -- (see #51); replayed appends now reconcile stale dependents
                -- (catalog._reconcile_replayed_append), but until a replay
                -- happens -- and on mirrors synced before it -- this table's
                -- own recorded_at can disagree. Ranking off it directly
                -- could then pick a different run_fingerprint than
                -- episodes_latest for the same episode_id, stitching rows
                -- from two different runs together. The inner join also
                -- means a run whose episodes file hasn't landed yet (crash
                -- debris mid-append, not yet retried) is invisible here,
                -- matching append_episode's own "episodes existing is what
                -- proves an append complete" idiom.
                e.recorded_at,
                row_number() OVER (
                    PARTITION BY m.episode_id, m.key
                    ORDER BY e.recorded_at DESC, m.run_fingerprint DESC
                ) AS row_rank
            FROM measurements m
            JOIN episodes_raw e USING (episode_id, run_fingerprint)
        ) WHERE row_rank = 1
        """
    )
    connection.execute(
        """
        CREATE VIEW check_runs_latest AS
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT
                c.* EXCLUDE (recorded_at),
                -- Ranked by the EPISODE's recorded_at for the same reason
                -- measurements_latest is: episodes is the only table whose
                -- single writer is guaranteed, so ranking off this table's own
                -- recorded_at could pick a different run_fingerprint than
                -- episodes_latest did and stitch two runs together.
                e.recorded_at,
                row_number() OVER (
                    PARTITION BY c.episode_id, c.check_name
                    ORDER BY e.recorded_at DESC, c.run_fingerprint DESC
                ) AS row_rank
            FROM check_runs c
            JOIN episodes_raw e USING (episode_id, run_fingerprint)
        ) WHERE row_rank = 1
        """
    )
    connection.execute(
        """
        CREATE VIEW observations_latest AS
        SELECT observations.* REPLACE (latest_check_run.recorded_at AS recorded_at)
        FROM observations
        JOIN check_runs_latest latest_check_run
          USING (episode_id, run_fingerprint, check_name, check_version)
        """
    )
    measurement_keys = [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT key FROM measurements_latest ORDER BY key"
        ).fetchall()
    ]
    if measurement_keys:
        # A PIVOT inside a view needs its values enumerated, so the wide view
        # is bound to the keys present when this connection was opened; a key
        # recorded later appears on the next open (curate() always reopens).
        quoted_keys = ", ".join(_quote_sql_string(key) for key in measurement_keys)
        connection.execute(
            f"""
            CREATE VIEW episodes AS
            SELECT
                e.* EXCLUDE (quarantined),
                {_STATUS_CASE_ALIASED} AS {EPISODES_VIEW_STATUS_COLUMN},
                m.* EXCLUDE (episode_id)
            FROM episodes_latest e
            LEFT JOIN (
                PIVOT (
                    SELECT
                        episode_id,
                        key,
                        coalesce(
                            value_double,
                            CASE
                                WHEN value_bool THEN 1.0
                                WHEN value_bool IS NOT NULL THEN 0.0
                            END
                        ) AS value
                    FROM measurements_latest
                ) ON key IN ({quoted_keys}) USING first(value) GROUP BY episode_id
            ) m USING (episode_id)
            """
        )
    else:
        connection.execute(
            f"""
            CREATE VIEW episodes AS
            SELECT
                e.* EXCLUDE (quarantined),
                {_STATUS_CASE_ALIASED} AS {EPISODES_VIEW_STATUS_COLUMN}
            FROM episodes_latest e
            """
        )


def _sync_catalog_mirror(catalog_root: "Path | str | StorageRoot") -> None:
    """Download any catalog table files a bucket mirror lacks.

    Local catalogs are already on disk; bucket catalogs need a fresh sync
    before polling can see newly appended Parquet files.
    """
    location = parse_storage_root(catalog_root)
    if isinstance(location, BucketStorageRoot):
        location.sync_into_mirror(tuple(_TABLE_DIRECTORIES.values()))


def _completed_append_exists(catalog_root: "Path | str | StorageRoot") -> bool:
    """Whether a completed catalog append is present.

    ``Catalog.append_episode`` writes the episodes file last, so any
    ``episodes/*.parquet`` object proves the whole append finished.
    """
    location = parse_storage_root(catalog_root)
    if isinstance(location, BucketStorageRoot):
        _sync_catalog_mirror(location)
        episodes_dir = location.mirror / "episodes"
        return episodes_dir.is_dir() and any(episodes_dir.glob("*.parquet"))
    return any(location.path.joinpath("episodes").glob("*.parquet"))


def _refresh_local_catalog_connection(
    connection: duckdb.DuckDBPyConnection, catalog_root: "Path | str | StorageRoot"
) -> None:
    """Replace an empty catalog surface after its first append completes.

    An empty catalog has real in-memory tables because DuckDB cannot define a
    ``read_parquet`` view over a glob with no matches. Once the episodes file
    from the first append exists, all dependent table files are complete too:
    ``Catalog.append_episode`` deliberately writes the episodes file last.
    Replacing the empty tables with the normal Parquet-backed views in one
    transaction lets a long-running local explorer see that first run without
    replacing its DuckDB connection.

    For bucket catalogs the query root is the synced mirror directory; this
    re-syncs before re-registering views so the first remote append is
    visible. Constrained connections have locked their configuration and
    materialized their data and must not call this.
    """
    location = parse_storage_root(catalog_root)
    _verify_catalog_format(location)
    query_root = _local_query_root(location)

    derived_view_names = (
        "episodes",
        "observations_latest",
        "check_runs_latest",
        "measurements_latest",
        "episodes_latest",
    )
    try:
        connection.execute("BEGIN TRANSACTION")
        for view_name in derived_view_names:
            connection.execute(f"DROP VIEW {view_name}")

        for relation_name in _LONG_TABLE_NAMES:
            relation_type_row = connection.execute(
                """
                SELECT table_type
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_name = ?
                """,
                [relation_name],
            ).fetchone()
            if relation_type_row is None:
                raise RuntimeError(f"catalog relation {relation_name!r} is missing")
            relation_type = str(relation_type_row[0])
            drop_kind = "VIEW" if relation_type == "VIEW" else "TABLE"
            connection.execute(f"DROP {drop_kind} {relation_name}")

        _register_catalog_relations(
            connection,
            query_root,
            constrained=False,
            writable_directories=(),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


class NonSingleSelectQueryError(ValueError):
    """SQL that parsed cleanly but is not exactly one SELECT statement.

    The single-SELECT rule's failure modes stay distinct so a caller can face
    them differently: a syntactically invalid string raises the parser's
    ``duckdb.Error`` with DuckDB's own diagnostic, while this exception is a
    WELL-FORMED input that just is not the one allowed shape — a lone
    ``CREATE TABLE``, a multi-statement string, or no statements at all.
    """


def reject_non_single_select(sql: str) -> None:
    """Raise ``NonSingleSelectQueryError`` unless *sql* is one SELECT statement.

    ``connection.execute()`` and ``connection.sql()`` both run **every**
    semicolon-separated statement in the input and return only the last
    result, so a tenant-supplied string can smuggle a second statement
    (``SELECT ...; CREATE TABLE pwned AS SELECT ...``) and it will execute
    silently.  ``extract_statements`` parses the text **without executing
    it** and lets us count statements and check their types before anything
    touches the connection.  A syntactically invalid string propagates the
    parser's ``duckdb.Error`` untouched — that is NOT this exception — so
    the two failure modes stay distinguishable.
    """
    parser_connection = duckdb.connect()
    try:
        statements = parser_connection.extract_statements(sql)
    finally:
        parser_connection.close()
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise NonSingleSelectQueryError("sql must be exactly one SELECT statement")


def _stage_manifest_and_count(
    connection: duckdb.DuckDBPyConnection, sql: str, staged_manifest: Path
) -> int:
    """COPY the query's result to the staged manifest; return its row count.

    ``sql`` is tenant-supplied on constrained connections.
    ``reject_non_single_select`` parses the input **without executing it**
    and refuses any payload that is not exactly one SELECT statement —
    including multi-statement injections separated by semicolons (e.g.
    ``SELECT ...; CREATE TABLE pwned AS SELECT ...``) and DDL-only input.
    ``connection.sql()`` is **not** the guard: on duckdb 1.5.5 it silently
    executes all statements and returns ``None`` for multi-statement input.
    ``write_parquet`` then materialises the single confirmed-safe relation
    to the staged path.

    A parser ``duckdb.Error`` here is surfaced as the same ``ValueError``
    this function has always raised for library callers; the
    parse-failure/rule-rejection distinction the service relies on is intact
    on ``reject_non_single_select`` itself.

    The row-count query only ever receives an internally-generated path,
    so it keeps the existing f-string form.
    """
    try:
        reject_non_single_select(sql)
    except duckdb.Error as exc:
        raise ValueError("sql must be exactly one SELECT statement") from exc
    connection.sql(sql).write_parquet(str(staged_manifest))

    (row_count,) = connection.execute(
        f"SELECT count(*) FROM read_parquet({_quote_sql_string(str(staged_manifest))})"
    ).fetchone() or (0,)
    return int(row_count)


def _collect_coverage(connection: duckdb.DuckDBPyConnection) -> tuple[int, list[CheckCoverage]]:
    # Both halves count CURRENT generations only. Off episodes_raw the
    # denominator grows by one every time a recording is reprocessed, while
    # the numerator keeps crediting check runs from canonicals that have
    # since been superseded -- so a reprocessed corpus reports coverage over
    # a corpus that does not exist. Coverage is the honesty feature (a
    # statistic over half a delivery must not look like a statistic over all
    # of it), so it is the last place to count a stale denominator.
    (total_episodes,) = connection.execute("SELECT count(*) FROM episodes_latest").fetchone() or (
        0,
    )
    if total_episodes == 0:
        return 0, []
    status_list = ", ".join(_quote_sql_string(status) for status in _RAN_STATUSES)
    # Joined on episode_id alone, NOT on (episode_id, run_fingerprint): a
    # stage-profile rerun appends check_runs rows only for the steps that
    # ran, under a fresh fingerprint, so matching the latest fingerprint too
    # would report every check uncovered after any relabel pass.
    coverage_rows = connection.execute(
        f"""
        SELECT check_name, count(DISTINCT episode_id) AS episodes_ran
        FROM check_runs WHERE status IN ({status_list})
          AND episode_id IN (SELECT episode_id FROM episodes_latest)
        GROUP BY check_name ORDER BY check_name
        """
    ).fetchall()
    return int(total_episodes), [
        CheckCoverage(
            check_name=str(check_name),
            episodes_ran=int(episodes_ran),
            total_episodes=int(total_episodes),
        )
        for check_name, episodes_ran in coverage_rows
    ]


def stale_episodes(
    catalog_root: "Path | str | StorageRoot",
    *,
    pipeline_version: str,
    schema_version: str | None = None,
) -> list[StaleEpisode]:
    """Episodes whose latest cataloged run was produced by other versions.

    The selective-reprocessing half of the version-stamp story: the corpus is
    assumed permanently mixed-version, and this lists exactly which episodes
    are stale against the versions you pass (``App.pipeline_version`` is the
    default source of the current one), so only those get re-ingested.

    Staleness is judged per **source recording**, on its most recent run:
    reprocessing rewrites the canonical file and therefore mints a new
    content-addressed ``episode_id``, so grouping by episode would keep
    reporting every superseded canonical forever. A source already
    reprocessed to the current versions is not stale, whatever its history
    says. That is exactly what ``episodes_latest`` selects, so this reads the
    view instead of carrying its own copy of the ranking.

    Versions are opaque identities: "stale" means *different*, never ordered
    comparison. Feed each result's ``source_uri`` back into ingestion
    (``hflow ingest`` / ``app.process``) to reprocess it.
    """
    connection = open_catalog_connection(catalog_root)
    try:
        predicate = "pipeline_version IS DISTINCT FROM ?"
        parameters: list[str] = [pipeline_version]
        if schema_version is not None:
            predicate += " OR schema_version IS DISTINCT FROM ?"
            parameters.append(schema_version)
        rows = connection.execute(
            f"""
            SELECT episode_id, uri, source_uri, pipeline_version, schema_version
            FROM episodes_latest WHERE {predicate} ORDER BY episode_id
            """,
            parameters,
        ).fetchall()
    finally:
        connection.close()
    return [
        StaleEpisode(
            episode_id=str(episode_id),
            uri=str(uri),
            source_uri=str(source_uri) if source_uri is not None else None,
            pipeline_version=str(stale_pipeline_version),
            schema_version=str(stale_schema_version),
        )
        for episode_id, uri, source_uri, stale_pipeline_version, stale_schema_version in rows
    ]


def curate(
    catalog_root: "Path | str | StorageRoot",
    sql: str,
    *,
    output: Path | str | None = None,
    constrained: bool = False,
) -> CurationReport:
    """Run ``sql`` over the catalog views; write the result as a manifest.

    ``sql`` is any SELECT (the ``episodes`` wide view covers everyday cuts).
    The manifest is written manifest-last: to a temp file, renamed (or
    uploaded) into place only after the query completed -- a partial manifest
    is unreachable. ``catalog_root`` and ``output`` each accept a local path
    or a bucket URL (``gs://.../manifest.parquet`` uploads the manifest).
    With ``output=None`` the query still runs (row count + coverage
    reporting) but nothing is written.

    ``constrained=True`` runs the SQL on a locked-down connection (see
    :func:`open_catalog_connection`) whose only file access is the
    manifest's own private staging directory -- the posture a hosted service
    uses for tenant-supplied SQL.
    """
    # file:// output URLs mean a local file, matching parse_storage_root's
    # convention for roots -- without this, the Path() branch below would
    # treat "file:///x" as a literal relative directory name.
    if isinstance(output, str) and output.startswith("file://"):
        file_url_path = output.removeprefix("file://")
        if not file_url_path.startswith("/"):
            raise ValueError(f"manifest output URL {output!r} must be absolute (file:///abs/path)")
        output = Path(file_url_path)

    location = parse_storage_root(catalog_root)
    _verify_catalog_format(location)
    query_root = _local_query_root(location)

    with ExitStack() as cleanup:
        # Resolve the manifest destination BEFORE opening the connection: a
        # constrained connection's directory allowlist is locked at open
        # time, so the staging directory must already be known. One variant
        # per destination shape, so a half-resolved state (a bucket target
        # with no staging directory) is unrepresentable.
        destination: _ManifestDestination = None
        if isinstance(output, str) and is_bucket_url(output):
            manifest_parent_url, _, manifest_name = output.rpartition("/")
            if not manifest_name:
                raise ValueError(f"manifest output URL {output!r} names no object")
            destination = _BucketManifestDestination(
                parent_url=manifest_parent_url,
                object_name=manifest_name,
                staging_dir=Path(
                    cleanup.enter_context(tempfile.TemporaryDirectory(prefix="hflow-manifest-"))
                ),
            )
        elif output is not None:
            local_manifest = Path(output)
            local_manifest.parent.mkdir(parents=True, exist_ok=True)
            # A PRIVATE staging subdirectory, never the output's parent: the
            # constrained allowlist is recursive and covers reads too, so
            # allowlisting the parent would hand tenant SQL everything beside
            # the manifest (a bare filename would expose the whole working
            # directory). Same filesystem, so the final replace stays atomic.
            destination = _LocalManifestDestination(
                final_path=local_manifest,
                staging_dir=Path(
                    cleanup.enter_context(
                        tempfile.TemporaryDirectory(
                            prefix=".hflow-manifest-", dir=local_manifest.parent
                        )
                    )
                ),
            )
        writable_directories = () if destination is None else (destination.staging_dir,)

        connection = _open_connection_over_root(
            query_root, constrained=constrained, writable_directories=writable_directories
        )
        cleanup.callback(connection.close)

        total_episodes, coverage = _collect_coverage(connection)

        manifest_path: Path | str | None
        match destination:
            case _BucketManifestDestination(
                parent_url=parent_url, object_name=object_name, staging_dir=staging_dir
            ):
                staged_manifest = staging_dir / object_name
                row_count = _stage_manifest_and_count(connection, sql, staged_manifest)
                manifest_path = BucketStorageRoot(parent_url).publish(staged_manifest, object_name)
            case _LocalManifestDestination(final_path=final_path, staging_dir=staging_dir):
                staged_manifest = staging_dir / final_path.name
                row_count = _stage_manifest_and_count(connection, sql, staged_manifest)
                staged_manifest.replace(final_path)
                manifest_path = final_path
            case None:
                (row_count,) = connection.execute(f"SELECT count(*) FROM ({sql})").fetchone() or (
                    0,
                )
                manifest_path = None

        return CurationReport(
            manifest_path=manifest_path,
            row_count=int(row_count),
            total_episodes=total_episodes,
            coverage=coverage,
        )
