"""The episode catalog: append-only Parquet tables under the data root.

The production warehouse pattern, collapsed to its single-tenant equivalent
(docs/ARCHITECTURE.md, "Catalog and curation storage"): every ingest appends
one row per episode plus long-format measurement/observation/tag/interval rows, all as
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
- ``observations``: long format, one row per field of timestamped evidence.
- ``tags`` / ``intervals``: as recorded by checks.
"""

import hashlib
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np

from hflow.format import CATALOG_FORMAT_VERSION
from hflow.steps import CheckResult, CheckStatus, Interval, MeasurementValue, Observation
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
        "episode_id VARCHAR, run_fingerprint VARCHAR, orchestrator_run_id VARCHAR, uri VARCHAR, "
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
    "observations": (
        "episode_id VARCHAR, run_fingerprint VARCHAR, check_name VARCHAR, "
        "check_version VARCHAR, observation_id VARCHAR, timestamp_ns BIGINT, "
        "key VARCHAR, value_double DOUBLE, value_text VARCHAR, value_bool BOOLEAN, "
        "recorded_at TIMESTAMPTZ"
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
# single existence check instead of re-reading every dependent file each time.
_reconciled_append_stems: set[tuple[str, str]] = set()

_FORMAT_MARKER_NAME = "format_version"

# DuckDB BIGINT columns store signed 64-bit integers. Validate timestamp
# evidence before any catalog files are committed so callers get contextual
# errors instead of a late database conversion failure.
_MAX_CATALOG_TIMESTAMP_NS = 2**63 - 1

# The quarantine state rendered as a queryable column by the curation
# views (curation.py builds both episodes views with it). Declared here
# because measurement keys are validated against the episodes view's full
# column list -- the table's columns plus this derived one.
EPISODES_VIEW_STATUS_COLUMN = "status"

# The three values that derived column takes. An episode is 'quarantined' when
# a critical check returned a False verdict, 'unverified' when a critical check
# CRASHED and so never produced one, and 'ok' when every critical check that ran
# answered. The middle value exists because the absence of a verdict used to be
# spelled the same way as a good one, which made "episodes I have checked for
# blur" silently include episodes where the blur check never finished (#164).
#
# 'unverified' is specifically the ERROR status and not "anything that is not
# passed". A critical check can also be SKIPPED (the episode was already
# quarantined upstream) or SUPERSEDED (the pipeline measures the same thing
# itself); neither is a crash and neither leaves the episode unchecked.
EPISODE_STATUS_OK = "ok"
EPISODE_STATUS_QUARANTINED = "quarantined"
EPISODE_STATUS_UNVERIFIED = "unverified"


def episode_status_case_sql(*, quarantined_column: str, check_runs_relation: str) -> str:
    """The canonical status rule as SQL, for one episode relation.

    Every SQL site that renders the status column builds it here, so the wide
    and narrow curation views and the snapshot export cannot drift apart.
    ``quarantined_column`` is the stored flag as spelled in the caller's scope
    (aliased or bare); ``check_runs_relation`` is a check-runs relation already
    narrowed to one row per (episode, check).
    """
    return (
        f"CASE WHEN {quarantined_column} THEN '{EPISODE_STATUS_QUARANTINED}' "
        f"WHEN EXISTS (SELECT 1 FROM {check_runs_relation} status_check_runs "
        f"WHERE status_check_runs.episode_id = {_episode_id_scope(quarantined_column)} "
        f"AND status_check_runs.critical "
        f"AND status_check_runs.status = '{CheckStatus.ERROR}') "
        f"THEN '{EPISODE_STATUS_UNVERIFIED}' "
        f"ELSE '{EPISODE_STATUS_OK}' END"
    )


def _episode_id_scope(quarantined_column: str) -> str:
    """``episode_id`` qualified by the same alias the caller used for the flag.

    The alias is required. An unqualified ``episode_id`` inside the correlated
    subquery would bind to the check-runs relation instead of the episode row,
    making the comparison a tautology and every non-quarantined episode
    unverified. That failure is silent and total, so it is refused here rather
    than left for a caller to discover.
    """
    alias, separator, _column = quarantined_column.rpartition(".")
    if not separator:
        raise ValueError(
            "quarantined_column must be qualified by the episode relation's alias "
            f"(got {quarantined_column!r}); an unqualified column would correlate "
            "the status subquery against itself"
        )
    return f"{alias}{separator}episode_id"


# Lowercased name -> canonical spelling of every column a measurement key
# must not claim. Derived from the episodes DDL plus the derived column
# above, so a new episode column is guarded without touching the check.
_EPISODES_VIEW_RESERVED_COLUMNS: dict[str, str] = {
    **{
        column.split()[0].lower(): column.split()[0]
        for column in TABLE_COLUMN_DDL["episodes"].split(",")
    },
    EPISODES_VIEW_STATUS_COLUMN.lower(): EPISODES_VIEW_STATUS_COLUMN,
}


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
    observations: list[Observation] = field(default_factory=list)
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
            observations=list(result.observations) if result is not None else [],
            tags=list(result.tags) if result is not None else [],
            intervals=list(result.intervals) if result is not None else [],
        )


@dataclass(frozen=True)
class AppendResult:
    """What one catalog append did."""

    episode_id: str
    run_fingerprint: str
    written: bool  # False when this exact run was already recorded


class QuarantineHistory:
    """Which episodes the catalog last recorded as quarantined, read once.

    The labels and media stages gate on this per episode, and the ``episodes``
    table is append-only with one parquet file per append -- so asking it one
    episode at a time costs one scan of a file set that grows with the corpus,
    per episode, plus one bucket-mirror sync each. Reading every quarantined
    episode in a single pass makes a whole stage batch pay that once.

    Scope is deliberately the batch, not the process: the snapshot is taken
    when the first lookup arrives, which is what the gate wants. The meta
    stage that decides quarantine has already finished by the time labels or
    media reads it, and the rows a stage appends for its own episodes as it
    runs cannot change another episode's answer.

    Only quarantined episodes are held. A clean episode and an unknown one
    lead the gate to the same decision -- proceed -- so carrying the rest of
    the corpus in memory would be paying to learn nothing.
    """

    def __init__(self, catalog_root: "Path | str | StorageRoot") -> None:
        location = parse_storage_root(catalog_root)
        match location:
            case LocalStorageRoot(path=local_root):
                self._episodes_dir = local_root / "episodes"
            case BucketStorageRoot():
                location.sync_into_mirror(("episodes",))
                self._episodes_dir = location.mirror / "episodes"
        self._tags_by_episode: dict[str, list[str]] | None = None

    def quarantine_tags(self, episode_id: str) -> list[str] | None:
        """The tags of the quarantine on ``episode_id``'s latest recorded run,
        or ``None`` when that run left it clean -- or when the catalog has
        never seen it, which the gate treats identically.

        "Latest" follows the ``episodes_latest`` view semantics: most recent
        ``recorded_at``, ties broken by ``run_fingerprint``.
        """
        if self._tags_by_episode is None:
            self._tags_by_episode = self._read_quarantined_episodes()
        return self._tags_by_episode.get(episode_id)

    def _read_quarantined_episodes(self) -> dict[str, list[str]]:
        if not self._episodes_dir.is_dir() or not any(self._episodes_dir.glob("*.parquet")):
            return {}
        glob_pattern = str(self._episodes_dir / "*.parquet").replace("'", "''")
        connection = duckdb.connect()
        try:
            # QUALIFY picks each episode's newest row first; the outer WHERE
            # then keeps only the ones that row calls quarantined. Filtering
            # before the window would instead find the newest QUARANTINED row,
            # resurrecting a quarantine that a later clean run had cleared.
            rows = connection.execute(
                f"""
                SELECT episode_id, quarantine_tags_json FROM (
                    SELECT episode_id, quarantined, quarantine_tags_json
                    FROM read_parquet('{glob_pattern}', union_by_name=true)
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY episode_id
                        ORDER BY recorded_at DESC, run_fingerprint DESC
                    ) = 1
                )
                WHERE quarantined
                """
            ).fetchall()
        finally:
            connection.close()
        return {
            str(episode_id): [str(tag) for tag in json.loads(tags_json)] if tags_json else []
            for episode_id, tags_json in rows
        }

    def close(self) -> None:
        """Drop the snapshot, so a reused history re-reads the table."""
        self._tags_by_episode = None

    def __enter__(self) -> "QuarantineHistory":
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


def latest_quarantine_tags(
    catalog_root: "Path | str | StorageRoot", episode_id: str
) -> list[str] | None:
    """One episode's carried-forward quarantine tags (see
    :meth:`QuarantineHistory.quarantine_tags`). Reuse a
    :class:`QuarantineHistory` when asking about more than one episode.
    """
    with QuarantineHistory(catalog_root) as history:
        return history.quarantine_tags(episode_id)


def content_episode_id(canonical_path: Path) -> str:
    """Content-address an episode: sha256 of the canonical file, 16 hex chars."""
    digest = hashlib.sha256()
    with canonical_path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _normalized_measurements(
    check_name: str, measurements: dict[str, MeasurementValue]
) -> dict[str, MeasurementValue]:
    """Coerce NumPy scalars to Python scalars; refuse anything else loudly.

    A non-finite float is refused here too, after the coercion so that
    ``np.float64("nan")`` is caught as well: NaN compares false against every
    threshold and its complement at once, so storing one drops the episode
    from both sides of a cut while the key still claims to be measured.

    An empty or whitespace-only key is refused as well: the wide view pivots
    every key into a column, so an empty key becomes a column whose name is
    the SQL expression that produced it -- a queryable surface with no name
    a person would write and no rename path (docs/CATALOG.md, "Naming
    measurement keys"). Refusing it here means nothing has been written yet.

    Real check code returns NumPy scalars (np.mean(), indexing, np.bool_
    verdicts), and only np.float64 subclasses a Python type -- the rest used
    to fall through every storage branch into all-NULL value columns while
    the key still landed. Coercion happens here, before anything consumes
    the values, so one np.float32(0.4) and float(0.4) outcome fingerprints
    and stores identically instead of splitting into two runs.
    """
    return _normalized_scalar_values(
        owner_description=f"check {check_name!r}",
        values=measurements,
        key_kind="measurement",
        empty_key_explanation=(
            "an empty key becomes a wide-view column named after the SQL that made it"
        ),
    )


def _normalized_scalar_values(
    *,
    owner_description: str,
    values: dict[str, MeasurementValue],
    key_kind: str,
    empty_key_explanation: str,
) -> dict[str, MeasurementValue]:
    """Normalize the typed scalar map shared by measurements and observations."""
    normalized: dict[str, MeasurementValue] = {}
    for key, value in values.items():
        # Before the value checks: the key is wrong independently of what it
        # holds, and reporting the value first would send a check author to
        # fix a NaN only to hit the real problem on the next run.
        if not isinstance(key, str):
            raise ValueError(
                f"{owner_description} set a {key_kind} key as {type(key).__name__}: "
                f"{key_kind} keys must be strings"
            )
        if not key.strip():
            raise ValueError(
                f"{owner_description} set {key!r} as a {key_kind} key: {key_kind} keys "
                f"must not be empty or whitespace-only ({empty_key_explanation})"
            )
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, MeasurementValue):
            raise ValueError(
                f"{owner_description} set {key_kind} {key!r} as {type(value).__name__}: "
                f"{key_kind} values hold one int/float/str/bool per key"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"{owner_description} set {key_kind} {key!r} as {value!r}: "
                "omit the key when there is no finite value"
            )
        normalized[key] = value
    return normalized


def _normalized_intervals(check_name: str, intervals: list[Interval]) -> list[Interval]:
    """Coerce NumPy scalar interval bounds to Python ints; refuse the rest.

    A user check computing segment boundaries from ``channel.to_numpy()``
    gets ``np.int64`` bounds from plain indexing, and those reach the run
    fingerprint without the by-hand ``int(...)`` every built-in applies. Like
    measurements, they are coerced here, before the fingerprint, so
    ``np.int64(5)`` and ``5`` describe one interval and hash to one outcome
    instead of splitting the run. A bound is nanoseconds and an ``int``: a
    float bound (``np.float64(5.0).item()`` included) is not a timestamp and
    is refused, naming the check and which bound.
    """
    normalized: list[Interval] = []
    for interval in intervals:
        bounds: dict[str, int] = {}
        for bound_name in ("start_ns", "end_ns"):
            value = getattr(interval, bound_name)
            if isinstance(value, np.generic):
                value = value.item()
            # bool subclasses int, so True satisfies the test below and would
            # store as 1 nanosecond. Excluded explicitly, the same way the port
            # and measurement boundaries do it.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"check {check_name!r} set interval {bound_name} as "
                    f"{type(value).__name__}: interval bounds are int nanoseconds"
                )
            if value > _MAX_CATALOG_TIMESTAMP_NS:
                raise ValueError(
                    f"check {check_name!r} interval {interval.label!r} set "
                    f"{bound_name}={value}: interval bounds must fit a DuckDB BIGINT"
                )
            bounds[bound_name] = value
        start_ns = bounds["start_ns"]
        end_ns = bounds["end_ns"]
        # The label, because a check emitting many intervals is otherwise a
        # needle in a haystack: the bounds alone do not say which one is wrong.
        # end_ns == start_ns is allowed on purpose -- an instant is a real thing
        # to record, and only end_ns < start_ns is nonsense.
        if start_ns < 0 or end_ns < 0:
            raise ValueError(
                f"check {check_name!r} set a negative bound on interval {interval.label!r} "
                f"(start_ns={start_ns}, end_ns={end_ns}): bounds are non-negative nanoseconds"
            )
        if end_ns < start_ns:
            raise ValueError(
                f"check {check_name!r} set an inverted interval {interval.label!r} "
                f"(start_ns={start_ns}, end_ns={end_ns}): end must be >= start"
            )
        normalized.append(replace(interval, **bounds))
    return normalized


def _normalized_observations(check_name: str, observations: list[Observation]) -> list[Observation]:
    """Normalize scalar fields and enforce an unambiguous per-check identity."""
    normalized: list[Observation] = []
    seen_observation_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, Observation):
            raise ValueError(
                f"check {check_name!r} emitted an observation as "
                f"{type(observation).__name__}: observations must be hflow.Observation values"
            )
        observation_id = observation.observation_id
        if not isinstance(observation_id, str):
            raise ValueError(
                f"check {check_name!r} set an observation id as "
                f"{type(observation_id).__name__}: observation ids must be strings"
            )
        if not observation_id.strip():
            raise ValueError(
                f"check {check_name!r} set an empty observation id: observation ids "
                "must be non-empty and stable within the check"
            )
        if observation_id != observation_id.strip():
            raise ValueError(
                f"check {check_name!r} set observation id {observation_id!r} with leading "
                "or trailing whitespace: observation ids are stored verbatim"
            )
        if observation_id in seen_observation_ids:
            raise ValueError(
                f"check {check_name!r} emitted duplicate observation id "
                f"{observation_id!r}: ids must be unique within one check result"
            )
        seen_observation_ids.add(observation_id)

        timestamp_ns = observation.timestamp_ns
        if isinstance(timestamp_ns, np.generic):
            timestamp_ns = timestamp_ns.item()
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise ValueError(
                f"check {check_name!r} observation {observation_id!r} set timestamp_ns "
                f"as {type(timestamp_ns).__name__}: observation timestamps are int nanoseconds"
            )
        if timestamp_ns > _MAX_CATALOG_TIMESTAMP_NS:
            raise ValueError(
                f"check {check_name!r} observation {observation_id!r} set "
                f"timestamp_ns={timestamp_ns}: timestamps must fit a DuckDB BIGINT"
            )
        if timestamp_ns < 0:
            raise ValueError(
                f"check {check_name!r} observation {observation_id!r} set negative "
                f"timestamp_ns={timestamp_ns}: timestamps are non-negative nanoseconds"
            )
        values = _normalized_scalar_values(
            owner_description=f"check {check_name!r} observation {observation_id!r}",
            values=observation.values,
            key_kind="observation field",
            empty_key_explanation="an unnamed field cannot be queried",
        )
        if not values:
            raise ValueError(
                f"check {check_name!r} observation {observation_id!r} has no values: "
                "omit empty observations"
            )
        normalized.append(
            Observation(
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                values=values,
            )
        )
    return normalized


def _raise_if_measurement_keys_shadow_episode_columns(
    check_rows: Sequence[CheckRunRow],
) -> None:
    """Refuse a run whose measurement key claims an ``episodes`` column.

    The wide ``episodes`` view pivots each numeric key into a column beside
    the promoted episode columns; DuckDB resolves such a collision by
    silently renaming the pivoted column to ``<key>_1``, so queries reading
    the episode column get the episode value and never the measurement.
    Keys compare case-insensitively -- DuckDB identifiers are, so ``Task``
    shadows ``task`` all the same.
    """
    shadowed = [
        f"{key!r} from {row.check_name!r} shadows {_EPISODES_VIEW_RESERVED_COLUMNS[key.lower()]!r}"
        for row in check_rows
        for key in row.measurements
        if key.lower() in _EPISODES_VIEW_RESERVED_COLUMNS
    ]
    if not shadowed:
        return
    raise ValueError(
        f"measurement keys collide with episodes columns: {'; '.join(shadowed)}. The wide "
        "episodes view pivots keys into columns beside those names, so DuckDB renames the "
        "measurement to <column>_1 and SELECT <column> returns the episode value instead. "
        "Rename the measurement key."
    )


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
    check_outcomes: list[dict[str, object]] = []
    for row in check_rows:
        check_outcome: dict[str, object] = {
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
        # Preserve the pre-observation fingerprint for every existing check
        # that emits none. Adding a compatible catalog feature must not make
        # an otherwise identical replay append a duplicate of old evidence.
        if row.observations:
            check_outcome["observations"] = sorted(
                (
                    observation.observation_id,
                    observation.timestamp_ns,
                    observation.values,
                )
                for observation in row.observations
            )
        check_outcomes.append(check_outcome)
    check_outcomes.sort(
        key=lambda outcome: json.dumps(outcome, sort_keys=True, separators=(",", ":"))
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
        for observation in row.observations:
            for key, value in observation.values.items():
                value_double = (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else None
                )
                value_text = value if isinstance(value, str) else None
                value_bool = value if isinstance(value, bool) else None
                connection.execute(
                    "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        episode_id,
                        run_fingerprint,
                        row.check_name,
                        row.check_version,
                        observation.observation_id,
                        observation.timestamp_ns,
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
        orchestrator_run_id: str | None = None,
    ) -> AppendResult:
        """Record one outcome; replaying that exact outcome is idempotent.

        ``uri`` is the address the ``episodes.uri`` column records for the
        canonical file -- pass the published object URL for bucket data
        roots; ``None`` records the resolved local path (the file's real
        address for local roots).

        ``orchestrator_run_id`` records WHICH orchestrated run produced this row.
        It is provenance, not identity: it is deliberately absent from
        :func:`_run_fingerprint`, because that hash exists so replaying an
        identical outcome is a no-op, and folding a per-run value into it
        would make every rerun append a duplicate of data already stored.

        The consequence is worth stating rather than discovering. A replay
        returns ``written=False`` above without touching the stored row, so
        this column names the run that FIRST recorded an outcome, not every
        run that has since produced it. That is the honest reading of an
        append that did nothing. ``None`` (the local dev loop, any caller
        outside the runtime) records NULL.
        """
        # One stored representation of "no orchestrator". A blank arrives
        # easily from an adapter reading its own environment
        # (``os.environ.get("RUN_ID", "")``), and storing it verbatim would
        # leave NULL and '' both meaning unorchestrated, so an IS NULL query
        # would miss rows and a filter could match a value naming nothing.
        # Blank-or-whitespace only: any other id is stored VERBATIM, because
        # it has to compare equal to what the orchestrator's own API reports.
        if orchestrator_run_id is not None and not orchestrator_run_id.strip():
            orchestrator_run_id = None
        episode_id = content_episode_id(canonical_path)
        # One normalized shape feeds every consumer below -- the run
        # fingerprint, the replay repair pass, and the dependent-table
        # inserts. Normalizing any later would let np.float32(0.4) and
        # float(0.4) hash as two outcomes while storing NULL columns.
        check_rows = [
            replace(
                row,
                measurements=_normalized_measurements(row.check_name, row.measurements),
                observations=_normalized_observations(row.check_name, row.observations),
                intervals=_normalized_intervals(row.check_name, row.intervals),
            )
            for row in check_rows
        ]
        _raise_if_measurement_keys_shadow_episode_columns(check_rows)
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
                "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    episode_id,
                    run_fingerprint,
                    orchestrator_run_id,
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
        (the idempotent-dedupe hot path) pays this pass's dependent file reads at
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
