"""Shared fakes for catalog, curation, and dataset-snapshot tests."""

from __future__ import annotations

from pathlib import Path

import duckdb

import hflow
from hflow.catalog import TABLE_COLUMN_DDL, Catalog, CheckRunRow
from hflow.transform import EpisodeStamps

FAKE_STAMPS = EpisodeStamps(
    schema_version="1",
    pipeline_version="abc123def456",
    ffmpeg_version="ffmpeg version test",
    robot_software_version="sim-0.1.0",
)

SNAPSHOT_EPISODE_STAMPS = EpisodeStamps(
    schema_version="1",
    pipeline_version="snapshot-pipeline-v1",
    ffmpeg_version="ffmpeg test",
    robot_software_version="robot test",
)


def write_fake_canonical(directory: Path, content: bytes = b"fake canonical bytes") -> Path:
    """A stand-in canonical episode: the catalog only hashes and references it."""
    path = directory / "episode.canonical.mcap"
    path.write_bytes(content)
    return path


def example_check_row(version: str = "v1", value: float = 1.0) -> CheckRunRow:
    """A measured check run touching every dependent table."""
    return CheckRunRow(
        check_name="example_check",
        check_version=version,
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.01,
        measurements={"example_metric": value, "note": "text", "flag": True},
        observations=[
            hflow.Observation(
                observation_id="frame:3",
                timestamp_ns=30,
                values={"score": value, "reviewed": True, "note": "clear"},
            )
        ],
        tags=["seen"],
        intervals=[hflow.Interval(start_ns=0, end_ns=10, label="span")],
    )


def recorded_at_values(catalog_directory: Path, stem: str) -> set[str]:
    """Every distinct ``recorded_at`` across one append's table files.

    One outcome must carry one timestamp in every table, or the per-key
    'latest' views attribute another run's rows to it. Cast to VARCHAR so
    TIMESTAMPTZ materialization needs no pytz.
    """
    timestamps: set[str] = set()
    with duckdb.connect() as connection:
        for table_name in TABLE_COLUMN_DDL:
            table_file = catalog_directory / table_name / f"{stem}.parquet"
            assert table_file.is_file(), f"{table_name} file missing for {stem}"
            rows = connection.execute(
                "SELECT DISTINCT CAST(recorded_at AS VARCHAR) FROM read_parquet(?)",
                [str(table_file)],
            ).fetchall()
            timestamps.update(recorded_at for (recorded_at,) in rows)
    return timestamps


def append_snapshot_episode(
    catalog: Catalog,
    working_directory: Path,
    *,
    name: str,
    score: float,
    with_media: bool,
    score_key: str = "quality/score",
) -> tuple[str, Path | None]:
    """Append one scored episode, optionally with a preview artifact to copy."""
    canonical_episode = working_directory / f"{name}.canonical.mcap"
    canonical_episode.write_bytes(f"canonical bytes for {name}".encode())
    preview_file: Path | None = None
    media_measurements: dict[str, hflow.MeasurementValue] = {}
    if with_media:
        preview_file = working_directory / f"{name}-preview.jpg"
        preview_file.write_bytes(b"portable preview bytes")
        media_measurements["artifact//wrist_cam/compressed"] = str(preview_file.resolve())

    append_result = catalog.append_episode(
        canonical_path=canonical_episode,
        stamps=SNAPSHOT_EPISODE_STAMPS,
        episode_metadata={"task": name, "operator": "robot-01"},
        check_rows=[
            CheckRunRow(
                check_name="quality",
                check_version="quality-v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.1,
                measurements={score_key: score, "caption": f"sample {name}"},
                observations=[
                    hflow.Observation(
                        observation_id="frame:1",
                        timestamp_ns=10,
                        values={"score": score, "reviewed": True},
                    )
                ],
                tags=["needs-inspection"],
                intervals=[hflow.Interval(start_ns=10, end_ns=20, label="inspect")],
            ),
            CheckRunRow(
                check_name="media/contact_sheet",
                check_version="media-v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.2,
                measurements=media_measurements,
            ),
        ],
    )
    return append_result.episode_id, preview_file
