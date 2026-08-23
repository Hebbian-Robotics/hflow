"""Portable review-dataset exports over a curated catalog snapshot.

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
from hflow.curation import open_catalog_connection
from hflow.storage import StorageRoot, fetch_uri

REVIEW_DATASET_FORMAT_NAME = "hflow-review-dataset"
REVIEW_DATASET_FORMAT_VERSION = "1"

_EPISODES_TABLE_FILE_NAME = "episodes.parquet"
_MEASUREMENTS_TABLE_FILE_NAME = "measurements.parquet"
_MEDIA_TABLE_FILE_NAME = "media.parquet"
_CHECK_RUNS_TABLE_FILE_NAME = "check_runs.parquet"
_TAGS_TABLE_FILE_NAME = "tags.parquet"
_INTERVALS_TABLE_FILE_NAME = "intervals.parquet"
_FORMAT_MARKER_FILE_NAME = "format.json"


class ReviewMediaMode(StrEnum):
    """How artifact URIs are represented in an exported review dataset."""

    REFERENCES = "references"
    COPY = "copy"


@dataclass(frozen=True)
class ReviewDatasetReport:
    """Observable result of one completed review-dataset export."""

    output_directory: Path
    episode_count: int
    measurement_count: int
    media_count: int
    copied_media_count: int
    check_run_count: int
    tag_count: int
    interval_count: int
    media_mode: ReviewMediaMode

    def summary(self) -> str:
        return (
            f"review dataset: {self.output_directory} "
            f"({self.episode_count} episodes, {self.measurement_count} measurements, "
            f"{self.media_count} media references, {self.copied_media_count} media files copied, "
            f"{self.tag_count} tags, {self.interval_count} intervals)"
        )

    def __str__(self) -> str:
        return self.summary()


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
            "CREATE TEMP TABLE review_selected_episode_ids AS "
            "SELECT episode_id FROM episodes_latest"
        )
    else:
        local_manifest = fetch_uri(manifest)
        try:
            connection.execute(
                "CREATE TEMP TABLE review_selected_episode_ids AS "
                "SELECT DISTINCT CAST(episode_id AS VARCHAR) AS episode_id "
                "FROM read_parquet(?)",
                [str(local_manifest)],
            )
        except duckdb.BinderException as error:
            raise ValueError(
                f"review manifest {manifest!s} must contain an episode_id column"
            ) from error
        except duckdb.Error as error:
            raise ValueError(f"could not read review manifest {manifest!s}: {error}") from error

        (null_episode_count_value,) = connection.execute(
            "SELECT count(*) FROM review_selected_episode_ids WHERE episode_id IS NULL"
        ).fetchone() or (0,)
        null_episode_count = int(null_episode_count_value)
        if null_episode_count:
            raise ValueError(f"review manifest {manifest!s} contains a null episode_id")

    missing_episode_ids = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT selected.episode_id
            FROM review_selected_episode_ids selected
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
            "review manifest selects episode IDs absent from the catalog; "
            f"first missing values: {examples}"
        )

    (selected_episode_count,) = connection.execute(
        "SELECT count(*) FROM review_selected_episode_ids"
    ).fetchone() or (0,)
    return int(selected_episode_count)


def _episodes_snapshot_query(connection: duckdb.DuckDBPyConnection) -> str:
    numeric_measurement_keys = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT measurements.key
            FROM measurements_latest measurements
            JOIN review_selected_episode_ids selected USING (episode_id)
            WHERE measurements.value_double IS NOT NULL
               OR measurements.value_bool IS NOT NULL
            ORDER BY measurements.key
            """
        ).fetchall()
    ]
    if not numeric_measurement_keys:
        return """
            SELECT
                episodes.* EXCLUDE (quarantined),
                CASE WHEN episodes.quarantined THEN 'quarantined' ELSE 'ok' END AS status
            FROM episodes_latest episodes
            JOIN review_selected_episode_ids selected USING (episode_id)
            ORDER BY episodes.episode_id
        """

    quoted_measurement_keys = ", ".join(
        _quote_sql_string(measurement_key) for measurement_key in numeric_measurement_keys
    )
    return f"""
        SELECT
            episodes.* EXCLUDE (quarantined),
            CASE WHEN episodes.quarantined THEN 'quarantined' ELSE 'ok' END AS status,
            measurements.* EXCLUDE (episode_id)
        FROM episodes_latest episodes
        JOIN review_selected_episode_ids selected USING (episode_id)
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
                JOIN review_selected_episode_ids selected USING (episode_id)
                WHERE latest.value_double IS NOT NULL OR latest.value_bool IS NOT NULL
            ) ON key IN ({quoted_measurement_keys}) USING first(value) GROUP BY episode_id
        ) measurements USING (episode_id)
        ORDER BY episodes.episode_id
    """


def _register_latest_check_runs(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP VIEW review_check_runs_latest AS
        SELECT * EXCLUDE (row_rank) FROM (
            SELECT
                check_runs.*,
                row_number() OVER (
                    PARTITION BY check_runs.episode_id, check_runs.check_name
                    ORDER BY check_runs.recorded_at DESC, check_runs.run_fingerprint DESC
                ) AS row_rank
            FROM check_runs
            JOIN review_selected_episode_ids selected USING (episode_id)
        ) WHERE row_rank = 1
        """
    )


def _media_kind_for_mime_type(mime_type: str | None) -> str:
    if mime_type is None:
        return "other"
    top_level_type, _, _subtype = mime_type.partition("/")
    if top_level_type in {"image", "video", "audio"}:
        return top_level_type
    return "other"


def _safe_file_name(file_name: str) -> str:
    safe_file_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name).strip("._")
    return safe_file_name or "artifact"


def _copied_media_relative_path(*, episode_id: str, artifact_name: str, artifact_uri: str) -> Path:
    uri_path = PurePosixPath(artifact_uri.replace("\\", "/"))
    source_file_name = uri_path.name or PurePosixPath(artifact_name).name
    safe_source_file_name = _safe_file_name(source_file_name)
    identity_digest = hashlib.sha256(f"{artifact_name}\0{artifact_uri}".encode()).hexdigest()[:12]
    safe_episode_id = _safe_file_name(episode_id)
    return Path("assets") / safe_episode_id / f"{identity_digest}-{safe_source_file_name}"


def _write_media_table(
    connection: duckdb.DuckDBPyConnection,
    staging_directory: Path,
    media_mode: ReviewMediaMode,
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
        JOIN review_selected_episode_ids selected USING (episode_id)
        WHERE starts_with(measurements.key, ?)
          AND measurements.value_text IS NOT NULL
        ORDER BY measurements.episode_id, measurements.key
        """,
        [ARTIFACT_MEASUREMENT_KEY_PREFIX],
    ).fetchall()

    connection.execute(
        """
        CREATE TEMP TABLE review_exported_media (
            episode_id VARCHAR,
            artifact_name VARCHAR,
            producer VARCHAR,
            producer_version VARCHAR,
            role VARCHAR,
            media_kind VARCHAR,
            mime_type VARCHAR,
            uri VARCHAR,
            recorded_at TIMESTAMPTZ
        )
        """
    )
    exported_media_rows: list[tuple[object, ...]] = []
    copied_media_count = 0
    for (
        episode_id_value,
        key_value,
        producer,
        producer_version,
        uri_value,
        recorded_at,
    ) in artifact_rows:
        episode_id = str(episode_id_value)
        artifact_name = str(key_value).removeprefix(ARTIFACT_MEASUREMENT_KEY_PREFIX)
        artifact_uri = str(uri_value)
        mime_type, _encoding = mimetypes.guess_type(artifact_uri)
        exported_uri = artifact_uri
        if media_mode is ReviewMediaMode.COPY:
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
            (
                episode_id,
                artifact_name,
                str(producer),
                str(producer_version),
                "contact_sheet" if producer == MEDIA_CONTACT_SHEET_STEP_NAME else "artifact",
                _media_kind_for_mime_type(mime_type),
                mime_type,
                exported_uri,
                recorded_at,
            )
        )

    if exported_media_rows:
        connection.executemany(
            "INSERT INTO review_exported_media VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            exported_media_rows,
        )
    media_count = _copy_query_to_parquet(
        connection,
        "SELECT * FROM review_exported_media ORDER BY episode_id, artifact_name",
        staging_directory / _MEDIA_TABLE_FILE_NAME,
    )
    return media_count, copied_media_count


def _activate_staged_export(
    staging_directory: Path, output_directory: Path, *, overwrite: bool
) -> None:
    if output_directory.exists() and not overwrite:
        raise FileExistsError(
            f"review export destination {output_directory} already exists; "
            "pass overwrite=True (CLI: --overwrite) to replace it"
        )
    if output_directory.is_symlink():
        raise ValueError(f"review export destination {output_directory} must not be a symlink")
    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(f"review export destination {output_directory} is not a directory")

    previous_directory: Path | None = None
    if output_directory.exists():
        previous_directory = output_directory.with_name(
            f".{output_directory.name}.previous-{uuid4().hex}"
        )
        output_directory.replace(previous_directory)
    try:
        staging_directory.replace(output_directory)
    except Exception:
        if previous_directory is not None:
            previous_directory.replace(output_directory)
        raise
    if previous_directory is not None:
        shutil.rmtree(previous_directory)


def export_review_dataset(
    catalog_root: Path | str | StorageRoot,
    output_directory: Path | str,
    *,
    manifest: Path | str | None = None,
    media_mode: ReviewMediaMode | str = ReviewMediaMode.REFERENCES,
    overwrite: bool = False,
) -> ReviewDatasetReport:
    """Export a portable snapshot selected by an optional Parquet manifest.

    The destination is a local directory containing ordinary Parquet tables
    and ``format.json``. ``manifest`` may be a local file or object-store URL
    and must contain ``episode_id``; without it, every latest catalog episode
    is exported. In ``references`` mode media URIs are preserved. In ``copy``
    mode every recorded artifact is materialized below ``assets/`` and the
    media table stores a path relative to the export directory.

    The completed directory appears atomically. Existing destinations are
    refused unless ``overwrite=True``; even then, the prior export remains in
    place until the replacement is fully staged.
    """
    resolved_media_mode = ReviewMediaMode(media_mode)
    resolved_output_directory = Path(output_directory)
    if not resolved_output_directory.name:
        raise ValueError("review export destination must name a directory, not a filesystem root")
    resolved_output_directory.parent.mkdir(parents=True, exist_ok=True)
    if resolved_output_directory.exists() and not overwrite:
        raise FileExistsError(
            f"review export destination {resolved_output_directory} already exists; "
            "pass overwrite=True (CLI: --overwrite) to replace it"
        )

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

            _copy_query_to_parquet(
                connection,
                _episodes_snapshot_query(connection),
                staging_directory / _EPISODES_TABLE_FILE_NAME,
            )
            measurement_count = _copy_query_to_parquet(
                connection,
                """
                SELECT measurements.*
                FROM measurements_latest measurements
                JOIN review_selected_episode_ids selected USING (episode_id)
                ORDER BY measurements.episode_id, measurements.key
                """,
                staging_directory / _MEASUREMENTS_TABLE_FILE_NAME,
            )
            media_count, copied_media_count = _write_media_table(
                connection, staging_directory, resolved_media_mode
            )
            check_run_count = _copy_query_to_parquet(
                connection,
                "SELECT * FROM review_check_runs_latest ORDER BY episode_id, check_name",
                staging_directory / _CHECK_RUNS_TABLE_FILE_NAME,
            )
            tag_count = _copy_query_to_parquet(
                connection,
                """
                SELECT tags.*
                FROM tags
                JOIN review_check_runs_latest latest
                  USING (episode_id, run_fingerprint, check_name)
                ORDER BY tags.episode_id, tags.check_name, tags.tag
                """,
                staging_directory / _TAGS_TABLE_FILE_NAME,
            )
            interval_count = _copy_query_to_parquet(
                connection,
                """
                SELECT intervals.*
                FROM intervals
                JOIN review_check_runs_latest latest
                  USING (episode_id, run_fingerprint, check_name)
                ORDER BY intervals.episode_id, intervals.check_name,
                         intervals.start_ns, intervals.end_ns
                """,
                staging_directory / _INTERVALS_TABLE_FILE_NAME,
            )
        finally:
            connection.close()

        format_marker = {
            "format": REVIEW_DATASET_FORMAT_NAME,
            "format_version": REVIEW_DATASET_FORMAT_VERSION,
            "media_mode": resolved_media_mode.value,
            "media_uri_base": (
                "export_directory" if resolved_media_mode is ReviewMediaMode.COPY else None
            ),
            "tables": {
                "episodes": _EPISODES_TABLE_FILE_NAME,
                "measurements": _MEASUREMENTS_TABLE_FILE_NAME,
                "media": _MEDIA_TABLE_FILE_NAME,
                "check_runs": _CHECK_RUNS_TABLE_FILE_NAME,
                "tags": _TAGS_TABLE_FILE_NAME,
                "intervals": _INTERVALS_TABLE_FILE_NAME,
            },
        }
        (staging_directory / _FORMAT_MARKER_FILE_NAME).write_text(
            json.dumps(format_marker, indent=2, sort_keys=True) + "\n"
        )
        _activate_staged_export(staging_directory, resolved_output_directory, overwrite=overwrite)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)

    return ReviewDatasetReport(
        output_directory=resolved_output_directory,
        episode_count=episode_count,
        measurement_count=measurement_count,
        media_count=media_count,
        copied_media_count=copied_media_count,
        check_run_count=check_run_count,
        tag_count=tag_count,
        interval_count=interval_count,
        media_mode=resolved_media_mode,
    )
