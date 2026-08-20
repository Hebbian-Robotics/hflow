"""Curation: a SQL query over the catalog, out comes ``manifest.parquet``.

Dyna's researcher interface ("a curation is now a SQL query") with zero
services: DuckDB reads the catalog's Parquet directly. Nothing here is
required -- the catalog is plain Parquet and users can point DuckDB, pandas,
or anything else at it (the no-obfuscation tenet); this module just registers
convenient views and applies the manifest-last idiom to the output.

Views registered on the connection:

- ``episodes_raw``, ``check_runs``, ``measurements``, ``tags``,
  ``intervals`` -- the long tables, exactly as stored.
- ``episodes_latest`` -- one row per episode_id (most recent append).
- ``measurements_latest`` -- one row per (episode_id, key), most recent by
  the OWNING episode's recorded_at (joined in), not the measurement row's
  own -- the latter can go stale independently of the episode it belongs to.
- ``episodes`` -- the wide view for everyday queries: latest episode rows,
  a ``status`` column (``'quarantined'``/``'ok'``), and one numeric column
  per measurement key (booleans as 0/1; text-valued measurements stay in the
  long table). A measurement key that collides with an episode column name
  is an error -- rename the measurement.

Dataset-level reporting always includes coverage denominators: which checks
ran on what fraction of episodes, because a statistic over half a delivery
must not look like a statistic over all of it.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

from hflow.catalog import TABLE_COLUMN_DDL
from hflow.format import CATALOG_FORMAT_VERSION
from hflow.steps import CheckStatus
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    StorageRoot,
    is_bucket_url,
    parse_storage_root,
)

_LONG_TABLE_NAMES = ("episodes_raw", "check_runs", "measurements", "tags", "intervals")
_TABLE_DIRECTORIES = {
    "episodes_raw": "episodes",
    "check_runs": "check_runs",
    "measurements": "measurements",
    "tags": "tags",
    "intervals": "intervals",
}

# Statuses that mean "the check actually ran on this episode" for coverage.
_RAN_STATUSES = (CheckStatus.PASSED.value, CheckStatus.FAILED.value, CheckStatus.MEASURED.value)


@dataclass(frozen=True)
class CheckCoverage:
    check_name: str
    episodes_ran: int
    total_episodes: int

    @property
    def fraction(self) -> float:
        return self.episodes_ran / self.total_episodes if self.total_episodes else 0.0


@dataclass(frozen=True)
class CurationReport:
    # A Path for local manifests, the object URL (str) for bucket manifests,
    # None when curate() ran without writing one.
    manifest_path: Path | str | None
    row_count: int
    total_episodes: int
    coverage: list[CheckCoverage]

    def summary(self) -> str:
        lines = [
            f"manifest: {self.manifest_path} ({self.row_count} rows, "
            f"from {self.total_episodes} cataloged episodes)"
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


def open_catalog_connection(catalog_root: "Path | str | StorageRoot") -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with all catalog views registered.

    Public on purpose: ``curate()`` is a convenience, not a gate -- take the
    connection and explore. ``catalog_root`` may be a local directory or a
    bucket prefix (``gs://.../catalog``).
    """
    location = parse_storage_root(catalog_root)
    _verify_catalog_format(location)
    root = _local_query_root(location)
    connection = duckdb.connect()

    for view_name in _LONG_TABLE_NAMES:
        directory_name = _TABLE_DIRECTORIES[view_name]
        table_directory = root / directory_name
        if any(table_directory.glob("*.parquet")):
            pattern = _quote_sql_string(str(table_directory / "*.parquet"))
            connection.execute(
                f"CREATE VIEW {view_name} AS "
                f"SELECT * FROM read_parquet({pattern}, union_by_name=true)"
            )
        else:
            # An empty catalog table: an empty relation with the real schema
            # keeps every downstream view and query definable.
            connection.execute(f"CREATE TABLE {view_name} ({TABLE_COLUMN_DDL[directory_name]})")

    connection.execute(
        """
        CREATE VIEW episodes_latest AS
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY episode_id
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
                -- (see #51) -- a later retry then early-returns on
                -- exists(episodes) and never revisits it, leaving this
                -- table's own recorded_at stale forever. Ranking off it
                -- directly could then pick a different run_fingerprint than
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
                CASE WHEN e.quarantined THEN 'quarantined' ELSE 'ok' END AS status,
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
            """
            CREATE VIEW episodes AS
            SELECT
                * EXCLUDE (quarantined),
                CASE WHEN quarantined THEN 'quarantined' ELSE 'ok' END AS status
            FROM episodes_latest
            """
        )
    return connection


def _collect_coverage(connection: duckdb.DuckDBPyConnection) -> tuple[int, list[CheckCoverage]]:
    (total_episodes,) = connection.execute(
        "SELECT count(DISTINCT episode_id) FROM episodes_raw"
    ).fetchone() or (0,)
    if total_episodes == 0:
        return 0, []
    status_list = ", ".join(_quote_sql_string(status) for status in _RAN_STATUSES)
    coverage_rows = connection.execute(
        f"""
        SELECT check_name, count(DISTINCT episode_id) AS episodes_ran
        FROM check_runs WHERE status IN ({status_list})
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

    Staleness is judged per **source recording** (``source_uri``, falling
    back to ``episode_id`` when none was recorded), on its most recent run:
    reprocessing rewrites the canonical file and therefore mints a new
    content-addressed ``episode_id``, so grouping by episode would keep
    reporting every superseded canonical forever. A source already
    reprocessed to the current versions is not stale, whatever its history
    says.

    Versions are content hashes: "stale" means *different*, never ordered
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
            SELECT episode_id, uri, source_uri, pipeline_version, schema_version FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY coalesce(source_uri, episode_id)
                    ORDER BY recorded_at DESC, run_fingerprint DESC
                ) AS row_rank FROM episodes_raw
            ) WHERE row_rank = 1 AND ({predicate}) ORDER BY episode_id
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
) -> CurationReport:
    """Run ``sql`` over the catalog views; write the result as a manifest.

    ``sql`` is any SELECT (the ``episodes`` wide view covers everyday cuts).
    The manifest is written manifest-last: to a temp file, renamed (or
    uploaded) into place only after the query completed -- a partial manifest
    is unreachable. ``catalog_root`` and ``output`` each accept a local path
    or a bucket URL (``gs://.../manifest.parquet`` uploads the manifest).
    With ``output=None`` the query still runs (row count + coverage
    reporting) but nothing is written.
    """
    # file:// output URLs mean a local file, matching parse_storage_root's
    # convention for roots -- without this, the Path() branch below would
    # treat "file:///x" as a literal relative directory name.
    if isinstance(output, str) and output.startswith("file://"):
        file_url_path = output.removeprefix("file://")
        if not file_url_path.startswith("/"):
            raise ValueError(f"manifest output URL {output!r} must be absolute (file:///abs/path)")
        output = Path(file_url_path)

    connection = open_catalog_connection(catalog_root)
    try:
        total_episodes, coverage = _collect_coverage(connection)

        manifest_path: Path | str | None = None
        if isinstance(output, str) and is_bucket_url(output):
            manifest_parent_url, _, manifest_name = output.rpartition("/")
            if not manifest_name:
                raise ValueError(f"manifest output URL {output!r} names no object")
            with tempfile.TemporaryDirectory(prefix="hflow-manifest-") as staging_name:
                staged_manifest = Path(staging_name) / manifest_name
                connection.execute(
                    f"COPY ({sql}) TO {_quote_sql_string(str(staged_manifest))} (FORMAT PARQUET)"
                )
                (row_count,) = connection.execute(
                    f"SELECT count(*) FROM read_parquet({_quote_sql_string(str(staged_manifest))})"
                ).fetchone() or (0,)
                manifest_path = BucketStorageRoot(manifest_parent_url).publish(
                    staged_manifest, manifest_name
                )
        elif output is not None:
            local_manifest = Path(output)
            local_manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = local_manifest.with_name(local_manifest.name + ".tmp")
            connection.execute(
                f"COPY ({sql}) TO {_quote_sql_string(str(temporary_path))} (FORMAT PARQUET)"
            )
            (row_count,) = connection.execute(
                f"SELECT count(*) FROM read_parquet({_quote_sql_string(str(temporary_path))})"
            ).fetchone() or (0,)
            temporary_path.replace(local_manifest)
            manifest_path = local_manifest
        else:
            (row_count,) = connection.execute(f"SELECT count(*) FROM ({sql})").fetchone() or (0,)

        return CurationReport(
            manifest_path=manifest_path,
            row_count=int(row_count),
            total_episodes=total_episodes,
            coverage=coverage,
        )
    finally:
        connection.close()
