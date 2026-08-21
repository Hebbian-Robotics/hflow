"""The episode catalog: append-only Parquet tables under the data root.

Dyna's warehouse, collapsed to its single-tenant equivalent
(docs/ARCHITECTURE.md, "Catalog and curation storage"): every ingest appends
one row per episode plus long-format measurement/tag/interval rows, all as
plain Parquet files DuckDB (or pandas, or anything) reads directly. No
services, no database.

Durability idioms honored here:

- **Content-addressed episodes**: ``episode_id`` is a hash of the canonical
  file's bytes, so re-ingesting an unchanged episode dedupes and a
  reprocessed episode (new ``pipeline_version``) is a distinct fact.
- **Create-if-absent appends**: each append writes one Parquet file per table
  named ``<episode_id>-<run_fingerprint>.parquet``; if the file already
  exists the append is a no-op. The fingerprint includes the observable
  outcome, so an exact replay deduplicates while a repaired retry appends.
- **Append, never overwrite**: re-running a changed check (new
  ``check_version``) adds new-version rows; curation picks or pins versions.

Tables (subdirectories of the catalog root, one file per append):

The catalog root may live on an object store (``gs://.../catalog``): appends
become store-native conditional puts and reads sync through the root's local
mirror (see ``hflow.storage``) -- the idioms above carry over unchanged.

- ``episodes``: identity, stamps, episode/v1 semantics, quarantine state.
- ``check_runs``: one row per (episode, check) invocation with its status --
  the coverage denominator, present even when a check produced nothing.
- ``measurements``: long format, one row per measurement key.
- ``tags`` / ``intervals``: as recorded by checks.
"""

import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from hflow.format import CATALOG_FORMAT_VERSION
from hflow.steps import CheckResult, CheckStatus, Interval, MeasurementValue
from hflow.storage import (
    BucketStorageRoot,
    LocalStorageRoot,
    StorageRoot,
    parse_storage_root,
)
from hflow.transform import EpisodeStamps

# episode/v1 keys promoted to first-class episode columns; everything else
# stays queryable in the metadata_json column.
_PROMOTED_EPISODE_KEYS = ("task", "operator", "success", "embodiment")

# Column DDL per table: the single owner of the catalog schema. The writer
# below and the curation views both derive from it.
TABLE_COLUMN_DDL: dict[str, str] = {
    "episodes": (
        "episode_id VARCHAR, run_fingerprint VARCHAR, uri VARCHAR, "
        "source_uri VARCHAR, schema_version VARCHAR, pipeline_version VARCHAR, "
        "robot_software_version VARCHAR, ffmpeg_version VARCHAR, "
        "task VARCHAR, operator VARCHAR, success VARCHAR, embodiment VARCHAR, "
        "metadata_json VARCHAR, quarantined BOOLEAN, "
        "quarantine_tags_json VARCHAR, recorded_at TIMESTAMPTZ"
    ),
    "check_runs": (
        "episode_id VARCHAR, run_fingerprint VARCHAR, check_name VARCHAR, "
        "check_version VARCHAR, critical BOOLEAN, status VARCHAR, "
        "duration_s DOUBLE, error VARCHAR, recorded_at TIMESTAMPTZ"
    ),
    "measurements": (
        "episode_id VARCHAR, run_fingerprint VARCHAR, check_name VARCHAR, "
        "check_version VARCHAR, key VARCHAR, value_double DOUBLE, "
        "value_text VARCHAR, value_bool BOOLEAN, recorded_at TIMESTAMPTZ"
    ),
    "tags": (
        "episode_id VARCHAR, run_fingerprint VARCHAR, check_name VARCHAR, "
        "tag VARCHAR, recorded_at TIMESTAMPTZ"
    ),
    "intervals": (
        "episode_id VARCHAR, run_fingerprint VARCHAR, check_name VARCHAR, "
        "label VARCHAR, start_ns BIGINT, end_ns BIGINT, recorded_at TIMESTAMPTZ"
    ),
}

_TABLE_NAMES = tuple(TABLE_COLUMN_DDL)

# Every table except episodes: written before it, force-aligned to its
# recorded_at afterwards (see append_episode's repair passes). Derived so a
# table added to TABLE_COLUMN_DDL automatically joins every dependent pass.
_DEPENDENT_TABLE_NAMES = tuple(name for name in TABLE_COLUMN_DDL if name != "episodes")

# Appends this process already verified (or repaired) as recorded_at-aligned,
# keyed by (location, file_stem). Once aligned a stem can never go stale
# again -- the episodes file is immutable and every post-commit dependent
# write carries its recorded_at -- so replays skip straight back to the
# single existence check instead of re-reading five files every time.
_reconciled_append_stems: set[tuple[str, str]] = set()

_FORMAT_MARKER_NAME = "format_version"


@dataclass(frozen=True)
class CheckRunRow:
    """The stored shape of one check invocation (see ``app.CheckRunReport``
    for the runtime shape; the conversion happens where a run is recorded)."""

    check_name: str
    check_version: str
    critical: bool
    status: CheckStatus
    duration_s: float
    error: str | None = None
    measurements: dict[str, MeasurementValue] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)

    @classmethod
    def from_result(
        cls,
        *,
        check_name: str,
        check_version: str,
        critical: bool,
        status: CheckStatus,
        duration_s: float,
        error: str | None,
        result: CheckResult | None,
    ) -> "CheckRunRow":
        return cls(
            check_name=check_name,
            check_version=check_version,
            critical=critical,
            status=status,
            duration_s=duration_s,
            error=error,
            measurements=dict(result.measurements) if result is not None else {},
            tags=list(result.tags) if result is not None else [],
            intervals=list(result.intervals) if result is not None else [],
        )


@dataclass(frozen=True)
class AppendResult:
    """What one catalog append did."""

    episode_id: str
    run_fingerprint: str
    written: bool  # False when this exact run was already recorded


@dataclass(frozen=True)
class LatestQuarantine:
    """The quarantine facts of an episode's most recent cataloged run."""

    quarantined: bool
    tags: list[str]


def latest_quarantine(
    catalog_root: "Path | str | StorageRoot", episode_id: str
) -> LatestQuarantine | None:
    """The latest recorded quarantine facts for ``episode_id``, or ``None``
    when the catalog (or the episode) is unknown.

    "Latest" follows the ``episodes_latest`` view semantics: most recent
    ``recorded_at``, ties broken by ``run_fingerprint``. Bucket catalogs sync
    the episodes table into their mirror first (only files the mirror lacks
    download; table files are append-only and content-named).
    """
    location = parse_storage_root(catalog_root)
    match location:
        case LocalStorageRoot(path=local_root):
            episodes_dir = local_root / "episodes"
        case BucketStorageRoot():
            location.sync_into_mirror(("episodes",))
            episodes_dir = location.mirror / "episodes"
    if not episodes_dir.is_dir() or not any(episodes_dir.glob("*.parquet")):
        return None
    glob_pattern = str(episodes_dir / "*.parquet").replace("'", "''")
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"""
            SELECT quarantined, quarantine_tags_json
            FROM read_parquet('{glob_pattern}', union_by_name=true)
            WHERE episode_id = ?
            ORDER BY recorded_at DESC, run_fingerprint DESC
            LIMIT 1
            """,
            [episode_id],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    quarantined, tags_json = row
    return LatestQuarantine(
        quarantined=bool(quarantined),
        tags=[str(tag) for tag in json.loads(tags_json)] if tags_json else [],
    )


def latest_quarantine_state(
    catalog_root: "Path | str | StorageRoot", episode_id: str
) -> bool | None:
    """Whether ``episode_id``'s most recent cataloged run left it quarantined.

    ``None`` when the catalog or the episode is unknown -- no catalog means
    no known quarantine, and callers proceed.
    """
    latest = latest_quarantine(catalog_root, episode_id)
    return None if latest is None else latest.quarantined


def content_episode_id(canonical_path: Path) -> str:
    """Content-address an episode: sha256 of the canonical file, 16 hex chars."""
    digest = hashlib.sha256()
    with canonical_path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _run_fingerprint(
    episode_id: str,
    pipeline_version: str,
    check_rows: Sequence[CheckRunRow],
    quarantine_tags: Sequence[str],
) -> str:
    """Identify one observable run outcome while keeping exact retries idempotent.

    Step versions identify intended behavior, not whether a particular
    attempt timed out or what it measured. Outcome data belongs in the append
    identity so a successful retry after a transient error is preserved,
    while replaying the exact same result remains a no-op.
    """
    check_outcomes = [
        {
            "check_name": row.check_name,
            "check_version": row.check_version,
            "critical": row.critical,
            "status": row.status.value,
            "error": row.error,
            "measurements": row.measurements,
            "tags": sorted(row.tags),
            "intervals": sorted(
                (interval.start_ns, interval.end_ns, interval.label) for interval in row.intervals
            ),
        }
        for row in check_rows
    ]
    # default=repr: measurement values are user data and may be non-JSON
    # scalars (numpy integers/bools reach here untouched); repr keeps the
    # fingerprint deterministic instead of crashing the whole append.
    check_outcomes.sort(
        key=lambda outcome: json.dumps(outcome, sort_keys=True, separators=(",", ":"), default=repr)
    )
    payload = json.dumps(
        {
            "episode_id": episode_id,
            "pipeline_version": pipeline_version,
            "check_outcomes": check_outcomes,
            "quarantine_tags": sorted(quarantine_tags),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _insert_dependent_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    episode_id: str,
    run_fingerprint: str,
    check_rows: Sequence[CheckRunRow],
    recorded_at: datetime,
) -> None:
    """Insert one append's dependent-table rows (everything except episodes).

    Shared by the initial append and the replay repair pass, so both write
    rows identical in every fingerprinted column -- between an append attempt
    and its repair only ``recorded_at`` (forced to the committed value) and
    ``duration_s`` (attempt wall clock, deliberately outside the run
    fingerprint) can differ.
    """
    for row in check_rows:
        connection.execute(
            "INSERT INTO check_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                episode_id,
                run_fingerprint,
                row.check_name,
                row.check_version,
                row.critical,
                row.status.value,  # stored as its string value (DB boundary)
                row.duration_s,
                row.error,
                recorded_at,
            ],
        )
        for key, value in row.measurements.items():
            value_double = (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None
            )
            value_text = value if isinstance(value, str) else None
            value_bool = value if isinstance(value, bool) else None
            connection.execute(
                "INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    episode_id,
                    run_fingerprint,
                    row.check_name,
                    row.check_version,
                    key,
                    value_double,
                    value_text,
                    value_bool,
                    recorded_at,
                ],
            )
        for tag in row.tags:
            connection.execute(
                "INSERT INTO tags VALUES (?, ?, ?, ?, ?)",
                [episode_id, run_fingerprint, row.check_name, tag, recorded_at],
            )
        for interval in row.intervals:
            connection.execute(
                "INSERT INTO intervals VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    episode_id,
                    run_fingerprint,
                    row.check_name,
                    interval.label,
                    interval.start_ns,
                    interval.end_ns,
                    recorded_at,
                ],
            )


class Catalog:
    """Append access to one catalog root (typically ``<data_root>/catalog``).

    The root is a local directory or a bucket prefix (``gs://.../catalog``);
    see ``hflow.storage``. On a bucket, create-if-absent appends become
    store-native conditional puts (``PutMode::Create``) -- the same
    durability idiom, now enforced by the object store itself -- and reads
    go through the root's local mirror.
    """

    def __init__(self, root: "Path | str | StorageRoot") -> None:
        self.location = parse_storage_root(root)
        # ``root`` stays a local directory in both modes: the root itself, or
        # the bucket's mirror (where table files land when synced for reads).
        match self.location:
            case LocalStorageRoot(path=local_path):
                self.root = local_path
                self.root.mkdir(parents=True, exist_ok=True)
            case BucketStorageRoot():
                self.root = self.location.workspace
        marker_content = (CATALOG_FORMAT_VERSION + "\n").encode()
        if not self.location.write_bytes_if_absent(_FORMAT_MARKER_NAME, marker_content):
            found_version = self.location.read_bytes(_FORMAT_MARKER_NAME).decode().strip()
            if found_version != CATALOG_FORMAT_VERSION:
                raise ValueError(
                    f"catalog at {self.location} has format version {found_version!r}; "
                    f"this build reads/writes version {CATALOG_FORMAT_VERSION!r}"
                )
        if isinstance(self.location, LocalStorageRoot):
            for table_name in _TABLE_NAMES:
                (self.root / table_name).mkdir(exist_ok=True)

    def table_dir(self, table_name: str) -> Path:
        """The LOCAL directory holding one table's Parquet files.

        For a bucket catalog this is the mirror subtree; call
        :meth:`sync_for_read` first when reading, so it reflects the store.
        """
        if table_name not in _TABLE_NAMES:
            raise KeyError(f"unknown catalog table {table_name!r}; tables: {_TABLE_NAMES}")
        return self.root / table_name

    def sync_for_read(self, table_names: Sequence[str] = _TABLE_NAMES) -> Path:
        """Make the local table directories reflect the store; returns the root.

        Local catalogs are already local (no-op). Bucket catalogs download
        only files the mirror lacks: table files are append-only and
        content-named, so an existing local file is final by convention.
        """
        for table_name in table_names:
            if table_name not in _TABLE_NAMES:
                raise KeyError(f"unknown catalog table {table_name!r}; tables: {_TABLE_NAMES}")
        if isinstance(self.location, BucketStorageRoot):
            self.location.sync_into_mirror(tuple(table_names))
        return self.root

    def append_episode(
        self,
        *,
        canonical_path: Path,
        stamps: EpisodeStamps,
        episode_metadata: dict[str, str],
        check_rows: Sequence[CheckRunRow],
        quarantine_tags: Sequence[str] = (),
        source_uri: str | None = None,
        uri: str | None = None,
    ) -> AppendResult:
        """Record one outcome; replaying that exact outcome is idempotent.

        ``uri`` is the address the ``episodes.uri`` column records for the
        canonical file -- pass the published object URL for bucket data
        roots; ``None`` records the resolved local path (the file's real
        address for local roots).
        """
        episode_id = content_episode_id(canonical_path)
        run_fingerprint = _run_fingerprint(
            episode_id,
            stamps.pipeline_version,
            check_rows,
            quarantine_tags,
        )
        file_stem = f"{episode_id}-{run_fingerprint}"

        # Create-if-absent: the episodes file is written last (manifest-last,
        # in miniature), so its existence proves the whole append completed.
        # A replay is also the retry lane for #51's residual crash window (a
        # winner that died between creating episodes and force-aligning the
        # dependents), so it reconciles dependent recorded_at before
        # returning instead of trusting the previous attempt blindly.
        if self.location.exists(f"episodes/{file_stem}.parquet"):
            self._reconcile_replayed_append(
                file_stem=file_stem,
                episode_id=episode_id,
                run_fingerprint=run_fingerprint,
                check_rows=check_rows,
            )
            return AppendResult(
                episode_id=episode_id, run_fingerprint=run_fingerprint, written=False
            )

        recorded_at = datetime.now(UTC)
        promoted = {key: episode_metadata.get(key) for key in _PROMOTED_EPISODE_KEYS}
        extra_metadata = {
            key: value
            for key, value in episode_metadata.items()
            if key not in _PROMOTED_EPISODE_KEYS
        }

        connection = duckdb.connect()
        try:
            for table_name, column_ddl in TABLE_COLUMN_DDL.items():
                connection.execute(f"CREATE TABLE {table_name} ({column_ddl})")
            connection.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    episode_id,
                    run_fingerprint,
                    uri if uri is not None else str(canonical_path.resolve()),
                    source_uri,
                    stamps.schema_version,
                    stamps.pipeline_version,
                    stamps.robot_software_version,
                    stamps.ffmpeg_version,
                    promoted["task"],
                    promoted["operator"],
                    promoted["success"],
                    promoted["embodiment"],
                    json.dumps(extra_metadata, sort_keys=True),
                    bool(quarantine_tags),
                    json.dumps(list(quarantine_tags)),
                    recorded_at,
                ],
            )
            _insert_dependent_rows(
                connection,
                episode_id=episode_id,
                run_fingerprint=run_fingerprint,
                check_rows=check_rows,
                recorded_at=recorded_at,
            )

            # Build every table file locally first, then store dependents
            # before the episodes file (see above). store_file_if_absent is
            # atomic in both modes: rename-into-place locally, a conditional
            # put (PutMode::Create) on a bucket.
            #
            # A dependent's create-if-absent can lose to either a genuinely
            # stale file (a crashed earlier append -- its episodes file never
            # landed, or we would have returned written=False above) or a
            # LIVE concurrent racer appending the identical outcome right
            # now: those two cases are indistinguishable from here (both
            # look like "the dependent exists, episodes doesn't"), so a
            # dependent that loses here is deliberately left alone instead
            # of guessed at -- see the repair pass below.
            with tempfile.TemporaryDirectory(prefix="hflow-catalog-append-") as staging_name:
                staging_dir = Path(staging_name)
                for table_name in _TABLE_NAMES:
                    staged_file = staging_dir / f"{table_name}.parquet"
                    escaped = str(staged_file).replace("'", "''")
                    connection.execute(f"COPY {table_name} TO '{escaped}' (FORMAT PARQUET)")
                for table_name in _DEPENDENT_TABLE_NAMES:
                    staged_file = staging_dir / f"{table_name}.parquet"
                    dependent_key = f"{table_name}/{file_stem}.parquet"
                    self.location.store_file_if_absent(staged_file, dependent_key)
                episodes_created = self.location.store_file_if_absent(
                    staging_dir / "episodes.parquet", f"episodes/{file_stem}.parquet"
                )
                if episodes_created:
                    # store_file_if_absent on episodes is atomic, so at most
                    # one caller ever observes True: we are now the unique,
                    # durable owner of this file_stem, whether this is a
                    # solo append, the winner of a race against a concurrent
                    # duplicate, or a retry repairing a crashed earlier
                    # attempt. Force every dependent to carry OUR
                    # recorded_at -- unconditionally, since a losing
                    # dependent write above (crash debris or a concurrent
                    # racer that did not go on to win episodes) may hold a
                    # different one. Every file of ONE append must carry one
                    # recorded_at: the per-key "latest" views (curation.py)
                    # each independently rank rows by (recorded_at,
                    # run_fingerprint) per table, so a mismatch could let
                    # them pick two different runs as "latest" for the same
                    # episode and stitch together rows that never belonged
                    # to the same outcome. ``publish`` is an unconditional
                    # atomic replace (rename-into-place locally, a plain put
                    # on a bucket) -- safe here because we already hold the
                    # unique win on episodes.
                    for table_name in _DEPENDENT_TABLE_NAMES:
                        staged_file = staging_dir / f"{table_name}.parquet"
                        dependent_key = f"{table_name}/{file_stem}.parquet"
                        self.location.publish(staged_file, dependent_key)
                    # The winner just aligned everything itself; later replays
                    # in this process can skip the verification pass.
                    _reconciled_append_stems.add((str(self.location), file_stem))
        finally:
            connection.close()

        return AppendResult(
            episode_id=episode_id, run_fingerprint=run_fingerprint, written=episodes_created
        )

    def _reconcile_replayed_append(
        self,
        *,
        file_stem: str,
        episode_id: str,
        run_fingerprint: str,
        check_rows: Sequence[CheckRunRow],
    ) -> None:
        """Heal dependents a crashed repair pass left behind (#51's residual).

        The episodes file's ``recorded_at`` is the one trustworthy timestamp
        of a completed append: its create-if-absent is the atomic commit. A
        winner that crashed between creating it and force-aligning the
        dependents leaves dependent tables carrying a different (stale)
        ``recorded_at``; replays used to early-return without ever revisiting
        them, so the per-table "latest" rankings could stitch rows from two
        different runs together. A replay holds content identical in every
        fingerprinted column by construction -- the run fingerprint hashes
        the check outcomes (``duration_s``, attempt wall clock, is
        deliberately excluded and may differ) -- so it can safely rebuild any
        disagreeing dependent with the committed timestamp. ``publish`` is an
        atomic replace, and every repairer writes the same identity content
        with the same committed ``recorded_at``, so concurrent replays
        converge instead of fighting.

        Verified stems are memoized per process: the steady-state replay
        (the idempotent-dedupe hot path) pays this pass's five file reads at
        most once, then returns to the single existence check.
        """
        memo_key = (str(self.location), file_stem)
        if memo_key in _reconciled_append_stems:
            return
        episodes_file = self.location.fetch(f"episodes/{file_stem}.parquet")
        connection = duckdb.connect()
        try:
            # The committed timestamp stays inside DuckDB throughout:
            # materializing a TIMESTAMPTZ into Python would require pytz,
            # which is not a dependency.
            episodes_pattern = str(episodes_file).replace("'", "''")
            connection.execute(
                "CREATE TABLE committed_timestamp AS "
                f"SELECT recorded_at FROM read_parquet('{episodes_pattern}') LIMIT 1"
            )
            (committed_count,) = connection.execute(
                "SELECT count(*) FROM committed_timestamp"
            ).fetchone() or (0,)
            if committed_count == 0:
                # append_episode always inserts exactly one episodes row and
                # publishes atomically, so a zero-row episodes file is not a
                # state this system produces -- refuse loudly rather than
                # silently skipping reconciliation against a corrupt marker.
                raise ValueError(
                    f"catalog file episodes/{file_stem}.parquet exists but holds no rows -- "
                    "the append commit marker is corrupt, so dependent tables cannot be "
                    "reconciled; delete the empty file to let the next append rebuild "
                    "this episode's tables"
                )

            stale_table_names: list[str] = []
            for table_name in _DEPENDENT_TABLE_NAMES:
                dependent_key = f"{table_name}/{file_stem}.parquet"
                try:
                    dependent_file = self.location.fetch(dependent_key)
                except FileNotFoundError:
                    # Dependents are stored before episodes, so this should
                    # not happen; if it somehow did, rebuilding it is the
                    # correct healing move either way.
                    stale_table_names.append(table_name)
                    continue
                dependent_pattern = str(dependent_file).replace("'", "''")
                (mismatched_row_count,) = connection.execute(
                    f"SELECT count(*) FROM read_parquet('{dependent_pattern}') "
                    "WHERE recorded_at IS DISTINCT FROM "
                    "(SELECT recorded_at FROM committed_timestamp)"
                ).fetchone() or (0,)
                if mismatched_row_count:
                    stale_table_names.append(table_name)
            if not stale_table_names:
                _reconciled_append_stems.add(memo_key)
                return

            for table_name in _DEPENDENT_TABLE_NAMES:
                connection.execute(f"CREATE TABLE {table_name} ({TABLE_COLUMN_DDL[table_name]})")
            _insert_dependent_rows(
                connection,
                episode_id=episode_id,
                run_fingerprint=run_fingerprint,
                check_rows=check_rows,
                # A placeholder only: the COPY below REPLACEs recorded_at
                # with the committed timestamp, entirely inside DuckDB.
                recorded_at=datetime.now(UTC),
            )
            with tempfile.TemporaryDirectory(prefix="hflow-catalog-repair-") as staging_name:
                staging_dir = Path(staging_name)
                for table_name in stale_table_names:
                    staged_file = staging_dir / f"{table_name}.parquet"
                    escaped = str(staged_file).replace("'", "''")
                    connection.execute(
                        f"COPY (SELECT * REPLACE ((SELECT recorded_at FROM committed_timestamp) "
                        f"AS recorded_at) FROM {table_name}) TO '{escaped}' (FORMAT PARQUET)"
                    )
                    self.location.publish(staged_file, f"{table_name}/{file_stem}.parquet")
            _reconciled_append_stems.add(memo_key)
        finally:
            connection.close()
