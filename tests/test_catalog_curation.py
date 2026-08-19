"""Catalog appends and curation queries (issues #16/#17)."""

from pathlib import Path
from typing import cast

import duckdb
import pytest

import hflow
from hflow.catalog import Catalog, CheckRunRow, content_episode_id
from hflow.cli import main as cli_main
from hflow.curation import curate, open_catalog_connection
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import EpisodeStamps

FAKE_STAMPS = EpisodeStamps(
    schema_version="1",
    pipeline_version="abc123def456",
    ffmpeg_version="ffmpeg version test",
    robot_software_version="sim-0.1.0",
)


def _fake_canonical(tmp_path: Path, content: bytes = b"fake canonical bytes") -> Path:
    path = tmp_path / "episode.canonical.mcap"
    path.write_bytes(content)
    return path


def _check_row(version: str = "v1", value: float = 1.0) -> CheckRunRow:
    return CheckRunRow(
        check_name="example_check",
        check_version=version,
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.01,
        measurements={"example_metric": value, "note": "text", "flag": True},
        tags=["seen"],
        intervals=[hflow.Interval(start_ns=0, end_ns=10, label="span")],
    )


def test_append_is_idempotent_for_the_same_content_versions_and_outcome(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = _fake_canonical(tmp_path)
    first = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[_check_row()],
    )
    second = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[_check_row()],
    )
    assert first.written and not second.written
    assert first.episode_id == second.episode_id == content_episode_id(canonical)
    parquet_files = list((tmp_path / "catalog" / "measurements").glob("*.parquet"))
    assert len(parquet_files) == 1


def test_rerunning_a_changed_check_appends_new_version_rows(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = _fake_canonical(tmp_path)
    catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row(version="v1", value=1.0)],
    )
    result = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row(version="v2", value=2.0)],
    )
    assert result.written

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        rows = connection.execute(
            "SELECT check_version, value_double FROM measurements "
            "WHERE key = 'example_metric' ORDER BY check_version"
        ).fetchall()
        assert rows == [("v1", 1.0), ("v2", 2.0)]
        latest_row = connection.execute(
            "SELECT value_double FROM measurements_latest WHERE key = 'example_metric'"
        ).fetchone()
        assert latest_row == (2.0,)
        wide_row = connection.execute("SELECT example_metric FROM episodes").fetchone()
        assert wide_row == (2.0,)
    finally:
        connection.close()


def test_successful_retry_after_error_appends_repaired_outcome(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = _fake_canonical(tmp_path)
    failed_row = CheckRunRow(
        check_name="remote_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.ERROR,
        duration_s=0.1,
        error="temporary timeout",
    )
    successful_row = CheckRunRow(
        check_name="remote_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        measurements={"score": 1.0},
    )

    first = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[failed_row],
    )
    second = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[successful_row],
    )

    assert first.written and second.written
    assert first.run_fingerprint != second.run_fingerprint
    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        assert connection.execute(
            "SELECT status FROM check_runs ORDER BY recorded_at"
        ).fetchall() == [("error",), ("measured",)]
        assert connection.execute("SELECT score FROM episodes").fetchone() == (1.0,)
    finally:
        connection.close()


def test_catalog_refuses_unknown_format_version(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "format_version").write_text("999\n")
    with pytest.raises(ValueError, match="format version '999'"):
        Catalog(root)


def test_curate_on_empty_catalog(tmp_path: Path) -> None:
    Catalog(tmp_path / "catalog")
    report = curate(tmp_path / "catalog", "SELECT 1 AS one")
    assert report.total_episodes == 0
    assert report.coverage == []
    assert report.row_count == 1

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        assert connection.execute("SELECT count(*) FROM episodes").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM episodes_latest").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM measurements_latest").fetchone() == (0,)
    finally:
        connection.close()


@pytest.fixture(scope="module")
def recorded_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sources_dir = tmp_path_factory.mktemp("sources")
    data_root = tmp_path_factory.mktemp("data-root")
    episode_specs = {
        # ~12.5% blackout: passes the gate below.
        "fold_napkin": SyntheticEpisodeSpec(
            duration_s=4.0, task="fold_napkin", black_segment=(1.0, 1.5)
        ),
        # 75% blackout: quarantined by the gate below.
        "pour_water": SyntheticEpisodeSpec(
            duration_s=4.0, task="pour_water", black_segment=(0.5, 3.5)
        ),
    }

    app = hflow.App("catalog-pipeline", data_root=data_root)

    @app.check()
    def joints(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.joint_discontinuity(ep)

    @app.check(critical=True)
    def camera_blackout(ep: hflow.Episode) -> hflow.CheckResult:
        stats = hflow.ffmpeg.frame_stats(ep.video("wrist_cam"))
        return hflow.CheckResult(
            measurements={"black_pct": stats.black_frame_pct},
            verdict=stats.black_frame_pct < 50.0,
        )

    @app.check()
    def late_check(ep: hflow.Episode) -> hflow.CheckResult:
        # Registered after the gate: skipped on quarantined episodes, so its
        # coverage must come out below 100%.
        return hflow.CheckResult(measurements={"late_metric": 1.0})

    for task_name, spec in episode_specs.items():
        source = synthesize_episode(sources_dir / f"{task_name}.mcap", spec)
        report = app.test(source, verbose=False, record=True)
        assert report.catalog_entry is not None and report.catalog_entry.written

    rerun_report = app.test(sources_dir / "fold_napkin.mcap", verbose=False, record=True)
    assert rerun_report.catalog_entry is not None
    return data_root


def test_readme_style_curation_query(recorded_data_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.parquet"
    report = curate(
        recorded_data_root / "catalog",
        """
        SELECT episode_id, uri FROM episodes
        WHERE task = 'fold_napkin'
          AND status != 'quarantined'
          AND black_pct < 50.0
        """,
        output=manifest,
    )
    assert report.row_count == 1
    assert manifest.is_file()
    manifest_row = duckdb.execute("SELECT uri FROM read_parquet(?)", [str(manifest)]).fetchone()
    assert manifest_row is not None
    assert str(manifest_row[0]).endswith("fold_napkin.canonical.mcap")


def test_quarantined_episode_is_filtered_by_status(recorded_data_root: Path) -> None:
    connection = open_catalog_connection(recorded_data_root / "catalog")
    try:
        rows = dict(
            connection.execute("SELECT task, status FROM episodes ORDER BY task").fetchall()
        )
    finally:
        connection.close()
    assert rows == {"fold_napkin": "ok", "pour_water": "quarantined"}


def test_coverage_denominators(recorded_data_root: Path) -> None:
    report = curate(recorded_data_root / "catalog", "SELECT episode_id FROM episodes")
    coverage_by_check = {entry.check_name: entry for entry in report.coverage}
    assert report.total_episodes == 2
    assert coverage_by_check["joints"].fraction == 1.0
    assert coverage_by_check["camera_blackout"].fraction == 1.0  # failed still ran
    assert coverage_by_check["late_check"].fraction == 0.5  # skipped when quarantined
    assert "late_check: 1/2 (50%)" in report.summary()


def test_cli_curate(
    recorded_data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "cli-manifest.parquet"
    exit_code = cli_main(
        [
            "curate",
            "SELECT episode_id, task FROM episodes WHERE status != 'quarantined'",
            "--catalog",
            str(recorded_data_root / "catalog"),
            "--output",
            str(manifest),
        ]
    )
    assert exit_code == 0
    assert manifest.is_file()
    printed = capsys.readouterr().out
    assert "1 rows" in printed
    assert "coverage" in printed


def test_cli_curate_requires_exactly_one_sql_source(tmp_path: Path) -> None:
    assert cli_main(["curate", "--catalog", str(tmp_path)]) == 2


def test_cli_curate_dry_run_does_not_write_manifest(
    recorded_data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "should-not-exist.parquet"
    exit_code = cli_main(
        [
            "curate",
            "SELECT episode_id, task FROM episodes WHERE status != 'quarantined'",
            "--catalog",
            str(recorded_data_root / "catalog"),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not manifest.is_file()
    printed = capsys.readouterr().out
    # Dry-run renders as a human message, not the literal None.
    assert "(not written; dry run)" in printed
    assert "None" not in printed
    assert "1 rows" in printed


def test_cli_curate_dry_run_rejects_output_flag(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli_main(["curate", "SELECT 1", "--dry-run", "--output", "x.parquet"])


def test_stale_episodes_lists_only_episodes_behind_the_current_versions(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    stale_stamps = FAKE_STAMPS
    current_stamps = EpisodeStamps(
        schema_version="1",
        pipeline_version="fresh00000001",
        ffmpeg_version="ffmpeg version test",
        robot_software_version="sim-0.1.0",
    )
    behind = _fake_canonical(tmp_path, content=b"behind the current pipeline")
    current = tmp_path / "current.canonical.mcap"
    current.write_bytes(b"already reprocessed")
    catalog.append_episode(
        canonical_path=behind,
        stamps=stale_stamps,
        episode_metadata={},
        check_rows=[],
        source_uri="episodes-in/behind.mcap",
    )
    catalog.append_episode(
        canonical_path=current,
        stamps=current_stamps,
        episode_metadata={},
        check_rows=[],
        source_uri="episodes-in/current.mcap",
    )

    stale = hflow.stale_episodes(
        tmp_path / "catalog", pipeline_version=current_stamps.pipeline_version
    )
    assert [episode.source_uri for episode in stale] == ["episodes-in/behind.mcap"]
    assert stale[0].pipeline_version == stale_stamps.pipeline_version

    # Staleness follows the SOURCE: reprocessing behind.mcap mints a new
    # content-addressed episode_id, and the source's latest run now carries
    # the current stamps, so the source stops being stale.
    reprocessed = tmp_path / "behind-reprocessed.canonical.mcap"
    reprocessed.write_bytes(b"behind, reprocessed to current")
    catalog.append_episode(
        canonical_path=reprocessed,
        stamps=current_stamps,
        episode_metadata={},
        check_rows=[],
        source_uri="episodes-in/behind.mcap",
    )
    remaining_stale_source_uris = {
        episode.source_uri
        for episode in hflow.stale_episodes(
            tmp_path / "catalog", pipeline_version=current_stamps.pipeline_version
        )
    }
    assert "episodes-in/behind.mcap" not in remaining_stale_source_uris

    # A schema bump makes every source stale regardless of pipeline version.
    all_stale = hflow.stale_episodes(
        tmp_path / "catalog",
        pipeline_version=current_stamps.pipeline_version,
        schema_version="2",
    )
    assert {episode.source_uri for episode in all_stale} == {
        "episodes-in/behind.mcap",
        "episodes-in/current.mcap",
    }


def test_cli_stale_prints_source_uris_for_ingest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[],
        source_uri="episodes-in/run_0001.mcap",
    )
    exit_code = cli_main(
        [
            "stale",
            "--catalog",
            str(tmp_path / "catalog"),
            "--pipeline-version",
            "somethingnewer",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    # stdout is exactly the pipeable URI list; the summary goes to stderr.
    assert captured.out.splitlines() == ["episodes-in/run_0001.mcap"]
    assert "1 episode(s)" in captured.err


def test_cli_stale_reports_a_broken_pipeline_file_instead_of_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Catalog(tmp_path / "catalog")
    broken_pipeline = tmp_path / "pipeline.py"
    broken_pipeline.write_text("raise RuntimeError('boom at import time')\n")
    exit_code = cli_main(
        ["stale", "--catalog", str(tmp_path / "catalog"), "--pipeline", str(broken_pipeline)]
    )
    assert exit_code == 2
    assert "boom at import time" in capsys.readouterr().err


def test_append_accepts_non_json_measurement_scalars(tmp_path: Path) -> None:
    """numpy scalars are user data: they must fingerprint, not crash the append."""
    import numpy as np

    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    catalog = Catalog(tmp_path / "catalog")
    row = CheckRunRow(
        check_name="numpy_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        # cast: deliberately smuggling user-supplied non-JSON scalars past the
        # declared MeasurementValue type -- exactly what real check code does.
        measurements=cast(dict, {"count": np.int64(3), "flag": np.bool_(True)}),
    )
    result = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[row],
    )
    assert result.written is True
    replay = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[row],
    )
    assert replay.written is False
    assert replay.run_fingerprint == result.run_fingerprint


def test_crash_repaired_append_keeps_one_recorded_at_across_tables(tmp_path: Path) -> None:
    """A retry after a crashed append must not mix timestamps across tables.

    Mixed recorded_at would let the per-key 'latest' views attribute another
    run's rows to this one.
    """
    import time

    import duckdb

    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    catalog = Catalog(tmp_path / "catalog")
    row = CheckRunRow(
        check_name="smoothness",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        measurements={"score": 1.0},
        tags=["reviewed"],
    )

    def append_same_outcome() -> "hflow.catalog.AppendResult":
        return catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )

    first = append_same_outcome()
    stem = f"{first.episode_id}-{first.run_fingerprint}"
    # Simulate the crash: the episodes file (written last) and one dependent
    # never landed; two dependents from the first attempt survive.
    (tmp_path / "catalog" / "episodes" / f"{stem}.parquet").unlink()
    (tmp_path / "catalog" / "intervals" / f"{stem}.parquet").unlink()
    time.sleep(0.01)  # a retry strictly later than the first attempt

    repaired = append_same_outcome()
    assert repaired.written is True
    assert repaired.run_fingerprint == first.run_fingerprint

    connection = duckdb.connect()
    try:
        timestamps = set()
        for table_name in ("episodes", "check_runs", "measurements", "tags", "intervals"):
            table_file = tmp_path / "catalog" / table_name / f"{stem}.parquet"
            assert table_file.is_file(), f"{table_name} file missing after repair"
            rows = connection.execute(
                # VARCHAR cast: comparing timestamps as text avoids a pytz
                # dependency for TIMESTAMPTZ materialization.
                f"SELECT DISTINCT CAST(recorded_at AS VARCHAR) FROM read_parquet('{table_file}')"
            ).fetchall()
            timestamps.update(value for (value,) in rows)
        assert len(timestamps) == 1, f"mixed recorded_at across tables: {timestamps}"
    finally:
        connection.close()


def test_curate_accepts_file_url_output(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    catalog.append_episode(
        canonical_path=canonical, stamps=FAKE_STAMPS, episode_metadata={}, check_rows=[]
    )
    manifest_target = tmp_path / "out" / "manifest.parquet"
    report = curate(
        tmp_path / "catalog",
        "SELECT episode_id FROM episodes",
        output=f"file://{manifest_target}",
    )
    assert report.row_count == 1
    assert manifest_target.is_file()
    assert not manifest_target.with_name(manifest_target.name + ".tmp").exists()
