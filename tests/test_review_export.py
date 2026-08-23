import json
from pathlib import Path

import duckdb
import pytest

import hflow
from hflow.catalog import Catalog, CheckRunRow
from hflow.cli import main as cli_main
from hflow.transform import EpisodeStamps

TEST_EPISODE_STAMPS = EpisodeStamps(
    schema_version="1",
    pipeline_version="review-pipeline-v1",
    ffmpeg_version="ffmpeg test",
    robot_software_version="robot test",
)


def _append_review_episode(
    catalog: Catalog,
    working_directory: Path,
    *,
    name: str,
    score: float,
    with_media: bool,
) -> tuple[str, Path | None]:
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
        stamps=TEST_EPISODE_STAMPS,
        episode_metadata={"task": name, "operator": "robot-01"},
        check_rows=[
            CheckRunRow(
                check_name="quality",
                check_version="quality-v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.1,
                measurements={"quality/score": score, "caption": f"review {name}"},
                tags=["needs-review"],
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


def test_review_export_is_a_tool_neutral_snapshot_selected_by_manifest(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    selected_episode_id, preview_file = _append_review_episode(
        catalog,
        tmp_path,
        name="fold-shirt",
        score=0.75,
        with_media=True,
    )
    assert preview_file is not None
    _append_review_episode(
        catalog,
        tmp_path,
        name="pour-water",
        score=0.25,
        with_media=False,
    )
    manifest = tmp_path / "manifest.parquet"
    hflow.curate(
        catalog.location,
        f"SELECT episode_id FROM episodes WHERE episode_id = '{selected_episode_id}'",
        output=manifest,
    )

    output_directory = tmp_path / "review-dataset"
    report = hflow.export_review_dataset(
        catalog.location,
        output_directory,
        manifest=manifest,
    )

    assert report.episode_count == 1
    assert report.media_count == 1
    assert report.copied_media_count == 0
    assert {path.name for path in output_directory.iterdir() if path.is_file()} == {
        "format.json",
        "episodes.parquet",
        "measurements.parquet",
        "media.parquet",
        "check_runs.parquet",
        "tags.parquet",
        "intervals.parquet",
    }
    format_marker = json.loads((output_directory / "format.json").read_text())
    assert format_marker["format"] == "hflow-review-dataset"
    assert format_marker["format_version"] == "1"
    assert format_marker["media_mode"] == "references"

    episode_row = duckdb.execute(
        'SELECT episode_id, task, status, "quality/score" FROM read_parquet(?)',
        [str(output_directory / "episodes.parquet")],
    ).fetchone()
    assert episode_row == (selected_episode_id, "fold-shirt", "ok", 0.75)
    measurement_rows = duckdb.execute(
        "SELECT key, value_double, value_text FROM read_parquet(?) ORDER BY key",
        [str(output_directory / "measurements.parquet")],
    ).fetchall()
    assert measurement_rows == [
        ("artifact//wrist_cam/compressed", None, str(preview_file.resolve())),
        ("caption", None, "review fold-shirt"),
        ("quality/score", 0.75, None),
    ]
    media_row = duckdb.execute(
        "SELECT artifact_name, role, media_kind, mime_type, uri FROM read_parquet(?)",
        [str(output_directory / "media.parquet")],
    ).fetchone()
    assert media_row == (
        "/wrist_cam/compressed",
        "contact_sheet",
        "image",
        "image/jpeg",
        str(preview_file.resolve()),
    )
    assert duckdb.execute(
        "SELECT tag FROM read_parquet(?)",
        [str(output_directory / "tags.parquet")],
    ).fetchall() == [("needs-review",)]
    assert duckdb.execute(
        "SELECT label, start_ns, end_ns FROM read_parquet(?)",
        [str(output_directory / "intervals.parquet")],
    ).fetchall() == [("inspect", 10, 20)]


def test_cli_review_export_copy_mode_materializes_media_and_refuses_implicit_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = Catalog(tmp_path / "catalog")
    selected_episode_id, preview_file = _append_review_episode(
        catalog,
        tmp_path,
        name="stack-blocks",
        score=1.0,
        with_media=True,
    )
    assert preview_file is not None
    output_directory = tmp_path / "portable-review"

    exit_code = cli_main(
        [
            "export",
            "review",
            "--catalog",
            str(catalog.location),
            "--output",
            str(output_directory),
            "--media",
            "copy",
        ]
    )

    assert exit_code == 0
    assert "1 episodes" in capsys.readouterr().out
    media_row = duckdb.execute(
        "SELECT uri FROM read_parquet(?)",
        [str(output_directory / "media.parquet")],
    ).fetchone()
    assert media_row is not None
    media_uri = str(media_row[0])
    assert not Path(media_uri).is_absolute()
    copied_media = output_directory / media_uri
    assert copied_media.is_file()
    assert copied_media.read_bytes() == preview_file.read_bytes()
    assert selected_episode_id in copied_media.parts

    repeated_exit_code = cli_main(
        [
            "export",
            "review",
            "--catalog",
            str(catalog.location),
            "--output",
            str(output_directory),
        ]
    )
    assert repeated_exit_code == 2
    assert "already exists" in capsys.readouterr().err

    overwrite_exit_code = cli_main(
        [
            "export",
            "review",
            "--catalog",
            str(catalog.location),
            "--output",
            str(output_directory),
            "--overwrite",
        ]
    )
    assert overwrite_exit_code == 0
    assert json.loads((output_directory / "format.json").read_text())["media_mode"] == "references"
    assert not copied_media.exists()


def test_review_export_rejects_manifest_episode_ids_absent_from_catalog(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    manifest = tmp_path / "manifest.parquet"
    duckdb.execute("CREATE TABLE selected (episode_id VARCHAR)")
    duckdb.execute("INSERT INTO selected VALUES ('missing-episode')")
    duckdb.execute(f"COPY selected TO '{manifest}' (FORMAT PARQUET)")

    with pytest.raises(ValueError, match="missing-episode"):
        hflow.export_review_dataset(
            catalog.location,
            tmp_path / "review-dataset",
            manifest=manifest,
        )

    assert not (tmp_path / "review-dataset").exists()
