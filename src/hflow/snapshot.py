"""Portable dataset snapshots over a curated catalog selection.

The export is intentionally made only of Parquet tables, ordinary media
files, and one JSON format marker.  HFlow-specific viewers and third-party
SDKs are not part of the contract: pandas, Polars, DuckDB, Spotlight, and
other tools can consume the same directory.
"""

import hashlib
import json
import mimetypes
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from uuid import uuid4

import duckdb

from hflow.app import ARTIFACT_MEASUREMENT_KEY_PREFIX, MEDIA_CONTACT_SHEET_STEP_NAME
from hflow.catalog import episode_status_case_sql
from hflow.curation import open_catalog_connection
from hflow.storage import StorageRoot, fetch_uri

DATASET_SNAPSHOT_FORMAT_NAME = "hflow-dataset-snapshot"
DATASET_SNAPSHOT_FORMAT_VERSION = "1"

_SAMPLES_TABLE_FILE_NAME = "samples.parquet"
_MEASUREMENTS_TABLE_FILE_NAME = "measurements.parquet"
_OBSERVATIONS_TABLE_FILE_NAME = "observations.parquet"
_MEDIA_TABLE_FILE_NAME = "media.parquet"
_CHECK_RUNS_TABLE_FILE_NAME = "check_runs.parquet"
_TAGS_TABLE_FILE_NAME = "tags.parquet"
_INTERVALS_TABLE_FILE_NAME = "intervals.parquet"
_FORMAT_MARKER_FILE_NAME = "format.json"
_REQUIRED_TABLE_FILES: dict[str, str] = {
    "samples": _SAMPLES_TABLE_FILE_NAME,
    "measurements": _MEASUREMENTS_TABLE_FILE_NAME,
    "observations": _OBSERVATIONS_TABLE_FILE_NAME,
    "media": _MEDIA_TABLE_FILE_NAME,
    "check_runs": _CHECK_RUNS_TABLE_FILE_NAME,
    "tags": _TAGS_TABLE_FILE_NAME,
    "intervals": _INTERVALS_TABLE_FILE_NAME,
}
_COPIED_ASSETS_DIRECTORY_NAME = "assets"


class SnapshotMediaMode(StrEnum):
    """How artifact URIs are represented in an exported dataset snapshot."""

    REFERENCES = "references"
    COPY = "copy"


class _SnapshotMediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OTHER = "other"


class _SnapshotMediaRole(StrEnum):
    CONTACT_SHEET = "contact_sheet"
    ARTIFACT = "artifact"


@dataclass(frozen=True)
class RetainedDatasetSnapshotBackup:
    """A prior snapshot retained because post-activation cleanup failed."""

    directory: Path
    cleanup_error: str


@dataclass(frozen=True)
class DatasetSnapshotReport:
    """Observable result of one completed dataset snapshot export."""

    output_directory: Path
    episode_count: int
    measurement_count: int
    observation_count: int
    media_count: int
    copied_media_count: int
    check_run_count: int
    tag_count: int
    interval_count: int
    media_mode: SnapshotMediaMode
    retained_backup: RetainedDatasetSnapshotBackup | None = None

    def summary(self) -> str:
        completed_summary = (
            f"dataset snapshot: {self.output_directory} "
            f"({self.episode_count} episodes, {self.measurement_count} measurements, "
            f"{self.observation_count} observation fields, "
            f"{self.media_count} media references, {self.copied_media_count} media files copied, "
            f"{self.tag_count} tags, {self.interval_count} intervals)"
        )
        if self.retained_backup is None:
            return completed_summary
        return (
            f"{completed_summary}; warning: previous snapshot backup retained at "
            f"{self.retained_backup.directory}: {self.retained_backup.cleanup_error}"
        )

    def __str__(self) -> str:
        return self.summary()


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256_hex(path: Path) -> str:
    """Full SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_integrity_record(relative_path: str, absolute_path: Path) -> dict[str, str | int]:
    """Receipt for one delivered snapshot file (table or copied asset)."""
    return {
        "path": relative_path,
        "size_bytes": absolute_path.stat().st_size,
        "sha256": _sha256_hex(absolute_path),
    }


def _inventory_content_id(entries: list[dict[str, str | int]]) -> str:
    """Full SHA-256 of the normalized integrity inventory.

    Entries are sorted by ``path`` and serialized with stable separators so the
    digest depends only on the delivered set, not write order. This is the
    delivery's integrity digest (corruption and deliberate set tampering), not
    an episode identity; the field name ``content_id`` matches prepared-manifest
    receipts (#389) but the value is full-length like the per-file hashes and
    Croissant's SHA-256 recommendation.
    """
    normalized = sorted(entries, key=lambda entry: str(entry["path"]))
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_snapshot_integrity_marker_fields(
    staging_directory: Path,
) -> dict[str, object]:
    """Additive ``integrity`` block for ``format.json`` under format version 1.

    Required Parquet tables and every regular file under ``assets/`` (copy mode)
    get ``path`` / ``size_bytes`` / ``sha256``. ``content_id`` digests that
    normalized inventory so a deleted member is detectable without a verifier
    product yet. References mode leaves ``assets`` empty: remote media are not
    fetched for hashing. Copy mode re-reads each copied asset once after the
    copy to compute its hash.
    """
    tables: dict[str, dict[str, str | int]] = {}
    for table_name, file_name in _REQUIRED_TABLE_FILES.items():
        absolute_path = staging_directory / file_name
        if not absolute_path.is_file():
            raise FileNotFoundError(
                f"snapshot staging is missing required table {file_name!r} "
                f"under {staging_directory}"
            )
        tables[table_name] = _file_integrity_record(file_name, absolute_path)

    assets: list[dict[str, str | int]] = []
    assets_directory = staging_directory / _COPIED_ASSETS_DIRECTORY_NAME
    if assets_directory.is_dir():
        for absolute_path in sorted(assets_directory.rglob("*")):
            if not absolute_path.is_file():
                continue
            relative_path = absolute_path.relative_to(staging_directory).as_posix()
            assets.append(_file_integrity_record(relative_path, absolute_path))

    inventory = [*tables.values(), *assets]
    return {
        "integrity": {
            "tables": tables,
            "assets": assets,
            "content_id": _inventory_content_id(inventory),
        }
    }


def _copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection, query: str, destination: Path
) -> int:
    connection.execute(
        f"COPY ({query}) TO {_quote_sql_string(str(destination))} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    (row_count,) = connection.execute(
        f"SELECT count(*) FROM read_parquet({_quote_sql_string(str(destination))})"
    ).fetchone() or (0,)
    return int(row_count)


def _register_selected_episode_ids(
    connection: duckdb.DuckDBPyConnection, manifest: Path | str | None
) -> int:
    if manifest is None:
        connection.execute(
            "CREATE TEMP TABLE snapshot_selected_episode_ids AS "
            "SELECT episode_id FROM episodes_latest"
        )
    else:
        local_manifest = fetch_uri(manifest)
        try:
            connection.execute(
                "CREATE TEMP TABLE snapshot_selected_episode_ids AS "
                "SELECT DISTINCT CAST(episode_id AS VARCHAR) AS episode_id "
                "FROM read_parquet(?)",
                [str(local_manifest)],
            )
        except duckdb.BinderException as error:
            raise ValueError(
                f"snapshot manifest {manifest!s} must contain an episode_id column"
            ) from error
        except duckdb.Error as error:
            raise ValueError(f"could not read snapshot manifest {manifest!s}: {error}") from error

        (null_episode_count_value,) = connection.execute(
            "SELECT count(*) FROM snapshot_selected_episode_ids WHERE episode_id IS NULL"
        ).fetchone() or (0,)
        null_episode_count = int(null_episode_count_value)
        if null_episode_count:
            raise ValueError(f"snapshot manifest {manifest!s} contains a null episode_id")

    missing_episode_ids = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT selected.episode_id
            FROM snapshot_selected_episode_ids selected
            LEFT JOIN episodes_latest cataloged USING (episode_id)
            WHERE cataloged.episode_id IS NULL
            ORDER BY selected.episode_id
            LIMIT 5
            """
        ).fetchall()
    ]
    if missing_episode_ids:
        examples = ", ".join(repr(episode_id) for episode_id in missing_episode_ids)
        raise ValueError(
            "snapshot manifest selects episode IDs absent from the catalog; "
            f"first missing values: {examples}"
        )

    (selected_episode_count,) = connection.execute(
        "SELECT count(*) FROM snapshot_selected_episode_ids"
    ).fetchone() or (0,)
    return int(selected_episode_count)


def _samples_snapshot_query(connection: duckdb.DuckDBPyConnection) -> str:
    numeric_measurement_keys = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT measurements.key
            FROM measurements_latest measurements
            JOIN snapshot_selected_episode_ids selected USING (episode_id)
            WHERE measurements.value_double IS NOT NULL
               OR measurements.value_bool IS NOT NULL
            ORDER BY measurements.key
            """
        ).fetchall()
    ]
    if not numeric_measurement_keys:
        return f"""
            SELECT
                episodes.* EXCLUDE (quarantined),
                {_SNAPSHOT_STATUS_CASE} AS status,
                primary_media.uri AS media_uri,
                primary_media.media_kind,
                primary_media.mime_type AS media_mime_type,
                primary_media.role AS media_role,
                primary_media.artifact_name AS media_artifact_name
            FROM episodes_latest episodes
            JOIN snapshot_selected_episode_ids selected USING (episode_id)
            LEFT JOIN snapshot_primary_media primary_media USING (episode_id)
            ORDER BY episodes.episode_id
        """

    quoted_measurement_keys = ", ".join(
        _quote_sql_string(measurement_key) for measurement_key in numeric_measurement_keys
    )
    return f"""
        SELECT
            episodes.* EXCLUDE (quarantined),
            {_SNAPSHOT_STATUS_CASE} AS status,
            measurements.* EXCLUDE (episode_id),
            primary_media.uri AS media_uri,
            primary_media.media_kind,
            primary_media.mime_type AS media_mime_type,
            primary_media.role AS media_role,
            primary_media.artifact_name AS media_artifact_name
        FROM episodes_latest episodes
        JOIN snapshot_selected_episode_ids selected USING (episode_id)
        LEFT JOIN (
            PIVOT (
                SELECT
                    latest.episode_id,
                    latest.key,
                    coalesce(
                        latest.value_double,
                        CASE
                            WHEN latest.value_bool THEN 1.0
                            WHEN latest.value_bool IS NOT NULL THEN 0.0
                        END
                    ) AS value
                FROM measurements_latest latest
                JOIN snapshot_selected_episode_ids selected USING (episode_id)
                WHERE latest.value_double IS NOT NULL OR latest.value_bool IS NOT NULL
            ) ON key IN ({quoted_measurement_keys}) USING first(value) GROUP BY episode_id
        ) measurements USING (episode_id)
        LEFT JOIN snapshot_primary_media primary_media USING (episode_id)
        ORDER BY episodes.episode_id
    """


def _register_latest_check_runs(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP VIEW snapshot_check_runs_latest AS
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT
                check_runs.* EXCLUDE (recorded_at),
                episodes.recorded_at,
                row_number() OVER (
                    PARTITION BY check_runs.episode_id, check_runs.check_name
                    ORDER BY episodes.recorded_at DESC, check_runs.run_fingerprint DESC
                ) AS row_rank
            FROM check_runs
            JOIN episodes_raw episodes USING (episode_id, run_fingerprint)
            JOIN snapshot_selected_episode_ids selected USING (episode_id)
        ) WHERE row_rank = 1
        """
    )


def _media_kind_for_mime_type(mime_type: str | None) -> _SnapshotMediaKind:
    if mime_type is None:
        return _SnapshotMediaKind.OTHER
    top_level_type, _, _subtype = mime_type.partition("/")
    match top_level_type:
        case "image":
            return _SnapshotMediaKind.IMAGE
        case "video":
            return _SnapshotMediaKind.VIDEO
        case "audio":
            return _SnapshotMediaKind.AUDIO
        case _:
            return _SnapshotMediaKind.OTHER


def _media_selection_rank(role: _SnapshotMediaRole, media_kind: _SnapshotMediaKind) -> int:
    match role, media_kind:
        case _SnapshotMediaRole.CONTACT_SHEET, _:
            return 0
        case _SnapshotMediaRole.ARTIFACT, _SnapshotMediaKind.IMAGE:
            return 1
        case _SnapshotMediaRole.ARTIFACT, _SnapshotMediaKind.VIDEO:
            return 2
        case _SnapshotMediaRole.ARTIFACT, _SnapshotMediaKind.AUDIO:
            return 3
        case _SnapshotMediaRole.ARTIFACT, _SnapshotMediaKind.OTHER:
            return 4


def _require_database_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"snapshot media {field_name} must be a string, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class _ExportedMediaRow:
    episode_id: str
    artifact_name: str
    producer: str
    producer_version: str
    role: _SnapshotMediaRole
    media_kind: _SnapshotMediaKind
    mime_type: str | None
    uri: str
    recorded_at: str

    def database_values(
        self,
    ) -> tuple[str, str, str, str, str, str, str | None, str, str, int]:
        return (
            self.episode_id,
            self.artifact_name,
            self.producer,
            self.producer_version,
            self.role.value,
            self.media_kind.value,
            self.mime_type,
            self.uri,
            self.recorded_at,
            _media_selection_rank(self.role, self.media_kind),
        )


def _safe_file_name(file_name: str) -> str:
    safe_file_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name).strip("._")
    return safe_file_name or "artifact"


def _copied_media_relative_path(*, episode_id: str, artifact_name: str, artifact_uri: str) -> Path:
    uri_path = PurePosixPath(artifact_uri.replace("\\", "/"))
    source_file_name = uri_path.name or PurePosixPath(artifact_name).name
    safe_source_file_name = _safe_file_name(source_file_name)
    identity_digest = hashlib.sha256(f"{artifact_name}\0{artifact_uri}".encode()).hexdigest()[:12]
    safe_episode_id = _safe_file_name(episode_id)
    return (
        Path(_COPIED_ASSETS_DIRECTORY_NAME)
        / safe_episode_id
        / f"{identity_digest}-{safe_source_file_name}"
    )


def _write_media_table(
    connection: duckdb.DuckDBPyConnection,
    staging_directory: Path,
    media_mode: SnapshotMediaMode,
) -> tuple[int, int]:
    artifact_rows = connection.execute(
        """
        SELECT
            measurements.episode_id,
            measurements.key,
            measurements.check_name,
            measurements.check_version,
            measurements.value_text,
            CAST(measurements.recorded_at AS VARCHAR) AS recorded_at
        FROM measurements_latest measurements
        JOIN snapshot_selected_episode_ids selected USING (episode_id)
        WHERE starts_with(measurements.key, ?)
          AND measurements.value_text IS NOT NULL
        ORDER BY measurements.episode_id, measurements.key
        """,
        [ARTIFACT_MEASUREMENT_KEY_PREFIX],
    ).fetchall()

    connection.execute(
        """
        CREATE TEMP TABLE snapshot_exported_media (
            episode_id VARCHAR,
            artifact_name VARCHAR,
            producer VARCHAR,
            producer_version VARCHAR,
            role VARCHAR,
            media_kind VARCHAR,
            mime_type VARCHAR,
            uri VARCHAR,
            recorded_at TIMESTAMPTZ,
            selection_rank INTEGER
        )
        """
    )
    exported_media_rows: list[_ExportedMediaRow] = []
    copied_media_count = 0
    for (
        episode_id_value,
        key_value,
        producer,
        producer_version,
        uri_value,
        recorded_at,
    ) in artifact_rows:
        episode_id = _require_database_string(episode_id_value, field_name="episode_id")
        measurement_key = _require_database_string(key_value, field_name="measurement key")
        artifact_name = measurement_key.removeprefix(ARTIFACT_MEASUREMENT_KEY_PREFIX)
        producer_name = _require_database_string(producer, field_name="producer")
        producer_version_name = _require_database_string(
            producer_version, field_name="producer version"
        )
        artifact_uri = _require_database_string(uri_value, field_name="URI")
        recorded_at_text = _require_database_string(recorded_at, field_name="recorded_at")
        mime_type, _encoding = mimetypes.guess_type(artifact_uri)
        media_kind = _media_kind_for_mime_type(mime_type)
        role = (
            _SnapshotMediaRole.CONTACT_SHEET
            if producer_name == MEDIA_CONTACT_SHEET_STEP_NAME
            else _SnapshotMediaRole.ARTIFACT
        )
        exported_uri = artifact_uri
        if media_mode is SnapshotMediaMode.COPY:
            source_file = fetch_uri(artifact_uri)
            copied_relative_path = _copied_media_relative_path(
                episode_id=episode_id,
                artifact_name=artifact_name,
                artifact_uri=artifact_uri,
            )
            copied_file = staging_directory / copied_relative_path
            copied_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, copied_file)
            exported_uri = copied_relative_path.as_posix()
            copied_media_count += 1
        exported_media_rows.append(
            _ExportedMediaRow(
                episode_id=episode_id,
                artifact_name=artifact_name,
                producer=producer_name,
                producer_version=producer_version_name,
                role=role,
                media_kind=media_kind,
                mime_type=mime_type,
                uri=exported_uri,
                recorded_at=recorded_at_text,
            )
        )

    if exported_media_rows:
        connection.executemany(
            "INSERT INTO snapshot_exported_media VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [exported_media_row.database_values() for exported_media_row in exported_media_rows],
        )
    media_count = _copy_query_to_parquet(
        connection,
        "SELECT * EXCLUDE (selection_rank) FROM snapshot_exported_media "
        "ORDER BY episode_id, artifact_name",
        staging_directory / _MEDIA_TABLE_FILE_NAME,
    )
    connection.execute(
        """
        CREATE TEMP VIEW snapshot_primary_media AS
        SELECT * EXCLUDE (selection_rank, media_rank)
        FROM (
            SELECT
                media.*,
                row_number() OVER (
                    PARTITION BY media.episode_id
                    ORDER BY media.selection_rank, media.artifact_name, media.uri
                ) AS media_rank
            FROM snapshot_exported_media media
        )
        WHERE media_rank = 1
        """
    )
    return media_count, copied_media_count


@dataclass(frozen=True)
class _NewDatasetSnapshotDestination:
    directory: Path


@dataclass(frozen=True)
class _ReplaceableDatasetSnapshotDestination:
    directory: Path


_DatasetSnapshotDestination = (
    _NewDatasetSnapshotDestination | _ReplaceableDatasetSnapshotDestination
)

# The snapshot export renders the same status vocabulary as the curation views,
# from the same builder, over its own selected-episode check-runs view.
_SNAPSHOT_STATUS_CASE = episode_status_case_sql(
    quarantined_column="episodes.quarantined",
    check_runs_relation="snapshot_check_runs_latest",
)


def _parse_dataset_snapshot_destination(
    output_directory: Path, *, overwrite: bool
) -> _DatasetSnapshotDestination:
    if output_directory.is_symlink():
        raise ValueError(f"snapshot export destination {output_directory} must not be a symlink")
    if not output_directory.exists():
        return _NewDatasetSnapshotDestination(output_directory)
    if not output_directory.is_dir():
        raise NotADirectoryError(
            f"snapshot export destination {output_directory} is not a directory"
        )
    if not overwrite:
        raise FileExistsError(
            f"snapshot export destination {output_directory} already exists; "
            "pass overwrite=True (CLI: --overwrite) to replace it"
        )

    format_marker_path = output_directory / _FORMAT_MARKER_FILE_NAME
    if format_marker_path.is_symlink() or not format_marker_path.is_file():
        raise ValueError(
            f"snapshot export destination {output_directory} cannot be replaced because it "
            f"does not contain a regular {_FORMAT_MARKER_FILE_NAME}"
        )
    try:
        format_marker = json.loads(format_marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"snapshot export destination {output_directory} cannot be replaced because "
            f"{_FORMAT_MARKER_FILE_NAME} is unreadable: {error}"
        ) from error
    if not isinstance(format_marker, dict):
        raise ValueError(
            f"snapshot export destination {output_directory} cannot be replaced because "
            f"{_FORMAT_MARKER_FILE_NAME} is not a JSON object"
        )
    if (
        format_marker.get("format") != DATASET_SNAPSHOT_FORMAT_NAME
        or format_marker.get("format_version") != DATASET_SNAPSHOT_FORMAT_VERSION
    ):
        raise ValueError(
            f"snapshot export destination {output_directory} cannot be replaced because "
            f"{_FORMAT_MARKER_FILE_NAME} does not identify supported "
            f"{DATASET_SNAPSHOT_FORMAT_NAME!r} format version "
            f"{DATASET_SNAPSHOT_FORMAT_VERSION!r}"
        )
    return _ReplaceableDatasetSnapshotDestination(output_directory)


def _activate_staged_export(
    staging_directory: Path, output_directory: Path, *, overwrite: bool
) -> RetainedDatasetSnapshotBackup | None:
    # The destination can change while the replacement is being staged, so
    # validate it again immediately before the first filesystem mutation.
    destination = _parse_dataset_snapshot_destination(output_directory, overwrite=overwrite)
    previous_directory: Path | None = None
    match destination:
        case _NewDatasetSnapshotDestination(directory=validated_output_directory):
            pass
        case _ReplaceableDatasetSnapshotDestination(directory=validated_output_directory):
            previous_directory = validated_output_directory.with_name(
                f".{validated_output_directory.name}.previous-{uuid4().hex}"
            )
            validated_output_directory.replace(previous_directory)
    try:
        staging_directory.replace(validated_output_directory)
    except Exception:
        if previous_directory is not None:
            previous_directory.replace(validated_output_directory)
        raise
    if previous_directory is not None:
        try:
            shutil.rmtree(previous_directory)
        except OSError as error:
            return RetainedDatasetSnapshotBackup(
                directory=previous_directory,
                cleanup_error=str(error),
            )
    return None


def export_dataset_snapshot(
    catalog_root: Path | str | StorageRoot,
    output_directory: Path | str,
    *,
    manifest: Path | str | None = None,
    media_mode: SnapshotMediaMode | str = SnapshotMediaMode.REFERENCES,
    overwrite: bool = False,
) -> DatasetSnapshotReport:
    """Export a portable snapshot selected by an optional Parquet manifest.

    The destination is a local directory containing ordinary Parquet tables
    and ``format.json``. ``manifest`` may be a local file or object-store URL
    and must contain ``episode_id``; without it, every latest catalog episode
    is exported. In ``references`` mode media URIs are preserved. In ``copy``
    mode every recorded artifact is materialized below ``assets/`` and the
    media table stores a path relative to the export directory.

    ``format.json`` stays format version ``1``. The published ``tables`` map
    remains a name-to-filename contract. An additive ``integrity`` block
    records per-file ``path`` / ``size_bytes`` / ``sha256`` for every required
    table and every copied asset, plus a full-length ``content_id`` over that
    normalized inventory. Copy mode re-reads each copied asset once after the
    copy to hash it. This is a delivery receipt only: this export does not
    verify the destination after transfer.

    The completed directory appears atomically. Existing destinations are
    refused unless ``overwrite=True`` and ``format.json`` identifies a
    supported HFlow dataset snapshot; even then, the prior export remains in
    place until the replacement is fully staged.
    """
    resolved_media_mode = SnapshotMediaMode(media_mode)
    resolved_output_directory = Path(output_directory)
    if resolved_output_directory.name in {"", ".", ".."}:
        raise ValueError("snapshot export destination must name a directory, not a filesystem root")
    resolved_output_directory.parent.mkdir(parents=True, exist_ok=True)
    _parse_dataset_snapshot_destination(resolved_output_directory, overwrite=overwrite)

    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_output_directory.name}.staging-",
            dir=resolved_output_directory.parent,
        )
    )
    try:
        connection = open_catalog_connection(catalog_root)
        try:
            episode_count = _register_selected_episode_ids(connection, manifest)
            _register_latest_check_runs(connection)

            measurement_count = _copy_query_to_parquet(
                connection,
                """
                SELECT measurements.*
                FROM measurements_latest measurements
                JOIN snapshot_selected_episode_ids selected USING (episode_id)
                ORDER BY measurements.episode_id, measurements.key
                """,
                staging_directory / _MEASUREMENTS_TABLE_FILE_NAME,
            )
            observation_count = _copy_query_to_parquet(
                connection,
                """
                SELECT observations.*
                FROM observations_latest observations
                JOIN snapshot_selected_episode_ids selected USING (episode_id)
                ORDER BY observations.episode_id, observations.check_name,
                         observations.observation_id, observations.key
                """,
                staging_directory / _OBSERVATIONS_TABLE_FILE_NAME,
            )
            media_count, copied_media_count = _write_media_table(
                connection, staging_directory, resolved_media_mode
            )
            _copy_query_to_parquet(
                connection,
                _samples_snapshot_query(connection),
                staging_directory / _SAMPLES_TABLE_FILE_NAME,
            )
            check_run_count = _copy_query_to_parquet(
                connection,
                "SELECT * FROM snapshot_check_runs_latest ORDER BY episode_id, check_name",
                staging_directory / _CHECK_RUNS_TABLE_FILE_NAME,
            )
            tag_count = _copy_query_to_parquet(
                connection,
                """
                SELECT tags.* EXCLUDE (recorded_at), latest.recorded_at
                FROM tags
                JOIN snapshot_check_runs_latest latest
                  USING (episode_id, run_fingerprint, check_name)
                ORDER BY tags.episode_id, tags.check_name, tags.tag
                """,
                staging_directory / _TAGS_TABLE_FILE_NAME,
            )
            interval_count = _copy_query_to_parquet(
                connection,
                """
                SELECT intervals.* EXCLUDE (recorded_at), latest.recorded_at
                FROM intervals
                JOIN snapshot_check_runs_latest latest
                  USING (episode_id, run_fingerprint, check_name)
                ORDER BY intervals.episode_id, intervals.check_name,
                         intervals.start_ns, intervals.end_ns
                """,
                staging_directory / _INTERVALS_TABLE_FILE_NAME,
            )
        finally:
            connection.close()

        format_marker = {
            "format": DATASET_SNAPSHOT_FORMAT_NAME,
            "format_version": DATASET_SNAPSHOT_FORMAT_VERSION,
            "media_mode": resolved_media_mode.value,
            "media_uri_base": (
                "export_directory" if resolved_media_mode is SnapshotMediaMode.COPY else None
            ),
            "tables": dict(_REQUIRED_TABLE_FILES),
            **_build_snapshot_integrity_marker_fields(staging_directory),
        }
        (staging_directory / _FORMAT_MARKER_FILE_NAME).write_text(
            json.dumps(format_marker, indent=2, sort_keys=True) + "\n"
        )
        retained_backup = _activate_staged_export(
            staging_directory, resolved_output_directory, overwrite=overwrite
        )
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)

    return DatasetSnapshotReport(
        output_directory=resolved_output_directory,
        episode_count=episode_count,
        measurement_count=measurement_count,
        observation_count=observation_count,
        media_count=media_count,
        copied_media_count=copied_media_count,
        check_run_count=check_run_count,
        tag_count=tag_count,
        interval_count=interval_count,
        media_mode=resolved_media_mode,
        retained_backup=retained_backup,
    )
