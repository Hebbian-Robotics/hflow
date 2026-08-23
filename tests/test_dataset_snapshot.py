import json
from pathlib import Path

import duckdb
import pytest

import hflow
import hflow.snapshot as snapshot_module
from hflow.catalog import Catalog, CheckRunRow
from hflow.cli import main as cli_main
from hflow.transform import EpisodeStamps

TEST_EPISODE_STAMPS = EpisodeStamps(
    schema_version="1",
    pipeline_version="snapshot-pipeline-v1",
    ffmpeg_version="ffmpeg test",
    robot_software_version="robot test",
)


def _append_snapshot_episode(
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
                measurements={"quality/score": score, "caption": f"sample {name}"},
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


def _append_media_priority_episode(
    catalog: Catalog,
    working_directory: Path,
    *,
    name: str,
    include_contact_sheet: bool,
) -> str:
    canonical_episode = working_directory / f"{name}.canonical.mcap"
    canonical_episode.write_bytes(f"canonical bytes for {name}".encode())

    artifact_paths = {
        "artifact/a-image.png": working_directory / f"{name}-a-image.png",
        "artifact/z-image.jpg": working_directory / f"{name}-z-image.jpg",
        "artifact/clip.mp4": working_directory / f"{name}-clip.mp4",
        "artifact/sound.wav": working_directory / f"{name}-sound.wav",
        "artifact/data.bin": working_directory / f"{name}-data.bin",
    }
    for artifact_path in artifact_paths.values():
        artifact_path.write_bytes(f"media bytes for {artifact_path.name}".encode())

    check_rows = [
        CheckRunRow(
            check_name="media/artifacts",
            check_version="media-artifacts-v1",
            critical=False,
            status=hflow.CheckStatus.MEASURED,
            duration_s=0.1,
            measurements={
                artifact_name: str(artifact_path.resolve())
                for artifact_name, artifact_path in artifact_paths.items()
            },
        )
    ]
    if include_contact_sheet:
        contact_sheet_path = working_directory / f"{name}-contact-sheet.jpg"
        contact_sheet_path.write_bytes(b"contact sheet bytes")
        check_rows.append(
            CheckRunRow(
                check_name="media/contact_sheet",
                check_version="contact-sheet-v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.1,
                measurements={"artifact/contact-sheet.jpg": str(contact_sheet_path.resolve())},
            )
        )

    append_result = catalog.append_episode(
        canonical_path=canonical_episode,
        stamps=TEST_EPISODE_STAMPS,
        episode_metadata={"task": name},
        check_rows=check_rows,
    )
    return append_result.episode_id


def test_dataset_snapshot_is_tool_neutral_and_selected_by_manifest(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    selected_episode_id, preview_file = _append_snapshot_episode(
        catalog,
        tmp_path,
        name="fold-shirt",
        score=0.75,
        with_media=True,
    )
    assert preview_file is not None
    _append_snapshot_episode(
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

    output_directory = tmp_path / "dataset-snapshot"
    report = hflow.export_dataset_snapshot(
        catalog.location,
        output_directory,
        manifest=manifest,
    )

    assert report.episode_count == 1
    assert report.media_count == 1
    assert report.copied_media_count == 0
    assert {path.name for path in output_directory.iterdir() if path.is_file()} == {
        "format.json",
        "samples.parquet",
        "measurements.parquet",
        "media.parquet",
        "check_runs.parquet",
        "tags.parquet",
        "intervals.parquet",
    }
    format_marker = json.loads((output_directory / "format.json").read_text())
    assert format_marker["format"] == "hflow-dataset-snapshot"
    assert format_marker["format_version"] == "1"
    assert format_marker["media_mode"] == "references"

    sample_row = duckdb.execute(
        """
        SELECT episode_id, task, status, "quality/score", media_uri,
               media_kind, media_mime_type, media_role, media_artifact_name
        FROM read_parquet(?)
        """,
        [str(output_directory / "samples.parquet")],
    ).fetchone()
    assert sample_row == (
        selected_episode_id,
        "fold-shirt",
        "ok",
        0.75,
        str(preview_file.resolve()),
        "image",
        "image/jpeg",
        "contact_sheet",
        "/wrist_cam/compressed",
    )
    measurement_rows = duckdb.execute(
        "SELECT key, value_double, value_text FROM read_parquet(?) ORDER BY key",
        [str(output_directory / "measurements.parquet")],
    ).fetchall()
    assert measurement_rows == [
        ("artifact//wrist_cam/compressed", None, str(preview_file.resolve())),
        ("caption", None, "sample fold-shirt"),
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
    ).fetchall() == [("needs-inspection",)]
    assert duckdb.execute(
        "SELECT label, start_ns, end_ns FROM read_parquet(?)",
        [str(output_directory / "intervals.parquet")],
    ).fetchall() == [("inspect", 10, 20)]


def test_cli_snapshot_copy_mode_materializes_media_and_refuses_implicit_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = Catalog(tmp_path / "catalog")
    selected_episode_id, preview_file = _append_snapshot_episode(
        catalog,
        tmp_path,
        name="stack-blocks",
        score=1.0,
        with_media=True,
    )
    assert preview_file is not None
    output_directory = tmp_path / "portable-snapshot"

    exit_code = cli_main(
        [
            "export",
            "snapshot",
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
    assert duckdb.execute(
        "SELECT media_uri FROM read_parquet(?)",
        [str(output_directory / "samples.parquet")],
    ).fetchone() == (media_uri,)

    repeated_exit_code = cli_main(
        [
            "export",
            "snapshot",
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
            "snapshot",
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


def test_dataset_snapshot_rejects_manifest_episode_ids_absent_from_catalog(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    manifest = tmp_path / "manifest.parquet"
    duckdb.execute("CREATE TABLE selected (episode_id VARCHAR)")
    duckdb.execute("INSERT INTO selected VALUES ('missing-episode')")
    duckdb.execute(f"COPY selected TO '{manifest}' (FORMAT PARQUET)")

    with pytest.raises(ValueError, match="missing-episode"):
        hflow.export_dataset_snapshot(
            catalog.location,
            tmp_path / "dataset-snapshot",
            manifest=manifest,
        )

    assert not (tmp_path / "dataset-snapshot").exists()


def test_dataset_snapshot_overwrite_refuses_unmarked_directory_without_removing_it(
    tmp_path: Path,
) -> None:
    catalog = Catalog(tmp_path / "catalog")
    output_directory = tmp_path / "unrelated-directory"
    output_directory.mkdir()
    sentinel_file = output_directory / "keep-me.txt"
    sentinel_file.write_text("unrelated user data")

    with pytest.raises(ValueError, match=r"regular format\.json"):
        hflow.export_dataset_snapshot(
            catalog.location,
            output_directory,
            overwrite=True,
        )

    assert sentinel_file.read_text() == "unrelated user data"


def test_dataset_snapshot_excludes_check_runs_without_a_committed_episode(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical_episode = tmp_path / "committed.canonical.mcap"
    canonical_episode.write_bytes(b"committed canonical bytes")
    append_result = catalog.append_episode(
        canonical_path=canonical_episode,
        stamps=TEST_EPISODE_STAMPS,
        episode_metadata={"task": "committed"},
        check_rows=[
            CheckRunRow(
                check_name="quality",
                check_version="committed-v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.1,
            )
        ],
    )

    orphan_check_runs_file = catalog.table_dir("check_runs") / "orphan-check-run.parquet"
    orphan_connection = duckdb.connect()
    try:
        orphan_connection.execute(
            """
            CREATE TABLE orphan_check_run AS SELECT
                ?::VARCHAR AS episode_id,
                'orphan-run'::VARCHAR AS run_fingerprint,
                'quality'::VARCHAR AS check_name,
                'uncommitted-v2'::VARCHAR AS check_version,
                false::BOOLEAN AS critical,
                'failed'::VARCHAR AS status,
                0.2::DOUBLE AS duration_s,
                'orphaned before episode commit'::VARCHAR AS error,
                '2999-01-01T00:00:00Z'::TIMESTAMPTZ AS recorded_at
            """,
            [append_result.episode_id],
        )
        orphan_connection.execute(
            f"COPY orphan_check_run TO '{orphan_check_runs_file}' (FORMAT PARQUET)"
        )
    finally:
        orphan_connection.close()

    output_directory = tmp_path / "dataset-snapshot"
    hflow.export_dataset_snapshot(catalog.location, output_directory)

    exported_check_runs = duckdb.execute(
        """
        SELECT run_fingerprint, check_version, status
        FROM read_parquet(?)
        WHERE episode_id = ? AND check_name = 'quality'
        """,
        [str(output_directory / "check_runs.parquet"), append_result.episode_id],
    ).fetchall()
    assert exported_check_runs == [
        (append_result.run_fingerprint, "committed-v1", hflow.CheckStatus.MEASURED.value)
    ]


def test_dataset_snapshot_reports_retained_backup_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = Catalog(tmp_path / "catalog")
    _append_snapshot_episode(
        catalog,
        tmp_path,
        name="cleanup-outcome",
        score=1.0,
        with_media=False,
    )
    output_directory = tmp_path / "dataset-snapshot"
    hflow.export_dataset_snapshot(catalog.location, output_directory)
    previous_generation_file = output_directory / "previous-generation.txt"
    previous_generation_file.write_text("old snapshot")

    real_remove_tree = snapshot_module.shutil.rmtree

    def refuse_previous_snapshot_cleanup(directory: Path | str) -> None:
        if ".previous-" in Path(directory).name:
            raise OSError("simulated cleanup failure")
        real_remove_tree(directory)

    monkeypatch.setattr(snapshot_module.shutil, "rmtree", refuse_previous_snapshot_cleanup)

    report = hflow.export_dataset_snapshot(
        catalog.location,
        output_directory,
        media_mode=hflow.SnapshotMediaMode.COPY,
        overwrite=True,
    )

    assert json.loads((output_directory / "format.json").read_text())["media_mode"] == "copy"
    assert not previous_generation_file.exists()
    assert report.retained_backup is not None
    assert report.retained_backup.directory.is_dir()
    assert (report.retained_backup.directory / previous_generation_file.name).read_text() == (
        "old snapshot"
    )
    assert report.retained_backup.cleanup_error == "simulated cleanup failure"
    assert "previous snapshot backup retained" in report.summary()


def test_dataset_snapshot_selects_primary_media_by_typed_precedence(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    episode_with_contact_sheet = _append_media_priority_episode(
        catalog,
        tmp_path,
        name="with-contact-sheet",
        include_contact_sheet=True,
    )
    episode_without_contact_sheet = _append_media_priority_episode(
        catalog,
        tmp_path,
        name="without-contact-sheet",
        include_contact_sheet=False,
    )
    output_directory = tmp_path / "dataset-snapshot"

    report = hflow.export_dataset_snapshot(catalog.location, output_directory)

    assert report.media_count == 11
    primary_media_rows = duckdb.execute(
        """
        SELECT episode_id, media_artifact_name, media_kind, media_role
        FROM read_parquet(?)
        ORDER BY episode_id
        """,
        [str(output_directory / "samples.parquet")],
    ).fetchall()
    assert primary_media_rows == sorted(
        [
            (episode_with_contact_sheet, "contact-sheet.jpg", "image", "contact_sheet"),
            (episode_without_contact_sheet, "a-image.png", "image", "artifact"),
        ]
    )
    media_contract_rows = duckdb.execute(
        """
        SELECT artifact_name, media_kind
        FROM read_parquet(?)
        WHERE episode_id = ?
        ORDER BY artifact_name
        """,
        [str(output_directory / "media.parquet"), episode_without_contact_sheet],
    ).fetchall()
    assert media_contract_rows == [
        ("a-image.png", "image"),
        ("clip.mp4", "video"),
        ("data.bin", "other"),
        ("sound.wav", "audio"),
        ("z-image.jpg", "image"),
    ]
