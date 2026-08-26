"""Catalog appends and curation queries (issues #16/#17)."""

import tempfile
from pathlib import Path
from typing import cast

import duckdb
import pytest

import hflow
from hflow.catalog import (
    _EPISODES_VIEW_RESERVED_COLUMNS,
    TABLE_COLUMN_DDL,
    Catalog,
    CheckRunRow,
    content_episode_id,
    episode_status_case_sql,
)
from hflow.checks import camera_frame_stats
from hflow.cli import main as cli_main
from hflow.curation import CurationReport, curate, open_catalog_connection
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


def test_the_orchestrator_run_id_is_recorded_without_entering_the_fingerprint(
    tmp_path: Path,
) -> None:
    """Provenance, not identity.

    The run fingerprint exists so replaying an identical outcome is a no-op.
    If the orchestrated run's id reached that hash, every rerun would append a
    duplicate of data already stored, which is the property this asserts is
    still intact: same outcome under a different run id is still one append.
    """
    catalog = Catalog(tmp_path / "catalog")
    canonical = _fake_canonical(tmp_path)
    first = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row()],
        orchestrator_run_id="manual__2026-08-23T00:00:00+00:00",
    )
    replayed_under_another_run = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row()],
        orchestrator_run_id="scheduled__2026-08-24T00:00:00+00:00",
    )

    assert first.written and not replayed_under_another_run.written
    assert first.run_fingerprint == replayed_under_another_run.run_fingerprint
    assert len(list((tmp_path / "catalog" / "episodes").glob("*.parquet"))) == 1

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        # The row keeps the run that FIRST recorded the outcome: the second
        # append did nothing, so claiming it as that run's output would be a
        # fiction. Documented on append_episode.
        assert connection.execute("SELECT orchestrator_run_id FROM episodes").fetchall() == [
            ("manual__2026-08-23T00:00:00+00:00",)
        ]
    finally:
        connection.close()


def test_an_unorchestrated_append_records_no_run(tmp_path: Path) -> None:
    """The dev loop and any non-runtime caller pass nothing and stay valid."""
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row()],
    )

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        assert connection.execute("SELECT orchestrator_run_id FROM episodes").fetchall() == [
            (None,)
        ]
    finally:
        connection.close()


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_run_id_records_as_absent_rather_than_as_a_value(
    tmp_path: Path, blank: str
) -> None:
    """One stored representation of "no orchestrator".

    A blank falls out of an adapter reading its own environment. Stored
    verbatim it would leave NULL and '' both meaning unorchestrated, so an
    IS NULL query would miss rows and a filter could match a value that names
    no run.
    """
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row()],
        orchestrator_run_id=blank,
    )

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        assert connection.execute(
            "SELECT count(*) FROM episodes WHERE orchestrator_run_id IS NULL"
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_a_real_run_id_is_stored_verbatim(tmp_path: Path) -> None:
    """Never normalized beyond blankness: it has to compare equal to the id
    the orchestrator's own API reports, or the join it exists for breaks."""
    run_id = "scheduled__2026-08-23T00:00:00+00:00"
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row()],
        orchestrator_run_id=run_id,
    )

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        assert connection.execute("SELECT orchestrator_run_id FROM episodes").fetchall() == [
            (run_id,)
        ]
    finally:
        connection.close()


def test_a_corpus_written_before_the_run_id_column_still_reads(tmp_path: Path) -> None:
    """No migration: an older episodes file reads back with NULL.

    Every glob reader passes ``union_by_name=true`` and the views select ``*``,
    so adding a column is backward compatible. This pins that rather than
    trusting it, because the alternative to it being true is a corpus that
    stops opening after an upgrade.
    """
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path, b"new bytes"),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row()],
        orchestrator_run_id="manual__2026-08-23T00:00:00+00:00",
    )

    # An episodes file in the pre-column shape, beside the new one.
    legacy_columns = TABLE_COLUMN_DDL["episodes"].replace("orchestrator_run_id VARCHAR, ", "")
    legacy_file = str(catalog_root / "episodes" / "legacy-episode-000000000000.parquet")
    writer = duckdb.connect()
    try:
        writer.execute(f"CREATE TABLE legacy ({legacy_columns})")
        writer.execute(
            "INSERT INTO legacy VALUES "
            "('legacyepisode', 'legacyrun000', 'file:///legacy.mcap', NULL, '1', "
            "'abc123def456', NULL, NULL, NULL, NULL, NULL, NULL, '{}', false, '[]', now())"
        )
        writer.execute(f"COPY legacy TO '{legacy_file}' (FORMAT PARQUET)")
    finally:
        writer.close()

    connection = open_catalog_connection(catalog_root)
    try:
        rows = connection.execute(
            "SELECT episode_id, orchestrator_run_id FROM episodes ORDER BY episode_id"
        ).fetchall()
    finally:
        connection.close()
    assert ("legacyepisode", None) in rows
    assert any(run_id == "manual__2026-08-23T00:00:00+00:00" for _episode_id, run_id in rows)


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
            duration_s=1.0,
            cameras=("wrist_cam",),
            task="fold_napkin",
            black_segment=(0.2, 0.35),
        ),
        # ~80% blackout: quarantined by the gate below.
        "pour_water": SyntheticEpisodeSpec(
            duration_s=1.0,
            cameras=("wrist_cam",),
            task="pour_water",
            black_segment=(0.1, 0.9),
        ),
    }

    app = hflow.App("catalog-pipeline", data_root=data_root)

    @app.check(version="1")
    def joints(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.joint_discontinuity(ep)

    @app.check(version="1", critical=True)
    def camera_blackout(ep: hflow.Episode) -> hflow.CheckResult:
        camera_topic = next(topic for topic in ep.cameras if "wrist_cam" in topic)
        camera_evidence = camera_frame_stats(ep, cameras=[camera_topic])
        black_frame_percent = camera_evidence.measurements[f"{camera_topic}/black_frame_pct"]
        assert isinstance(black_frame_percent, float)
        return hflow.CheckResult(
            measurements={"black_pct": black_frame_percent},
            verdict=black_frame_percent < 50.0,
        )

    @app.check(version="1")
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


def test_cli_curate_dry_run_reports_without_writing_a_manifest(
    recorded_data_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data-root"
    monkeypatch.setenv("HFLOW_DATA_ROOT", str(data_root))
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
    printed = capsys.readouterr().out
    assert "1 rows" in printed
    assert "manifest: (not written; dry run)" in printed
    assert "manifest: None" not in printed
    assert not (data_root / "manifest.parquet").exists()


def test_cli_curate_rejects_dry_run_with_an_explicit_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main(
            [
                "curate",
                "SELECT 1",
                "--catalog",
                str(tmp_path),
                "--dry-run",
                "--output",
                str(tmp_path / "manifest.parquet"),
            ]
        )
    assert exc_info.value.code == 2


def test_curation_report_summary_renders_the_no_output_case() -> None:
    report = CurationReport(
        manifest_path=None,
        row_count=3,
        total_episodes=10,
        coverage=[],
    )
    summary = report.summary()
    assert "manifest: (not written; dry run)" in summary
    assert "manifest: None" not in summary


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


def test_reprocessing_a_source_supersedes_its_previous_generation(tmp_path: Path) -> None:
    """One row per source recording, everywhere the corpus is counted.

    Reprocessing mints a new content-addressed episode_id, so a per-episode
    ranking would leave both generations in ``episodes_latest`` -- and both
    carry the SAME ``uri``, because publication overwrites in place, so the
    duplicate is not even distinguishable by address.
    """
    catalog = Catalog(tmp_path / "catalog")
    first_generation = _fake_canonical(tmp_path, content=b"first canonical bytes")
    catalog.append_episode(
        canonical_path=first_generation,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[_check_row(value=1.0)],
        source_uri="episodes-in/fold.mcap",
        uri="/data/episodes/fold.canonical.mcap",
    )
    reprocessed = tmp_path / "fold-reprocessed.canonical.mcap"
    reprocessed.write_bytes(b"reprocessed canonical bytes")
    second_append = catalog.append_episode(
        canonical_path=reprocessed,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[_check_row(value=2.0)],
        source_uri="episodes-in/fold.mcap",
        uri="/data/episodes/fold.canonical.mcap",
    )

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        assert connection.execute("SELECT count(*) FROM episodes_raw").fetchone() == (2,)
        latest = connection.execute("SELECT episode_id FROM episodes_latest").fetchall()
        assert latest == [(second_append.episode_id,)]
        # The wide view is the documented raw-SQL surface, so it must agree:
        # otherwise every user query counts the recording twice.
        wide = connection.execute("SELECT task, example_metric FROM episodes").fetchall()
        assert wide == [("fold_napkin", 2.0)]
    finally:
        connection.close()

    # Coverage is the honesty feature: the denominator is the corpus, not the
    # corpus plus its history.
    report = curate(tmp_path / "catalog", "SELECT episode_id FROM episodes")
    assert report.total_episodes == 1
    assert report.row_count == 1
    assert {entry.check_name: entry.fraction for entry in report.coverage} == {"example_check": 1.0}


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


def test_cli_stale_exit_code_returns_one_when_episodes_are_behind(
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
            "--exit-code",
        ]
    )
    assert exit_code == 1
    # The flag only changes the exit code; the pipeable URI list is unchanged.
    assert capsys.readouterr().out.splitlines() == ["episodes-in/run_0001.mcap"]


def test_cli_stale_exit_code_returns_zero_when_nothing_is_behind(
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
            FAKE_STAMPS.pipeline_version,
            "--exit-code",
        ]
    )
    assert exit_code == 0
    assert capsys.readouterr().out == ""


def test_constrained_connection_confines_sql_to_the_catalog(tmp_path: Path) -> None:
    """The service posture for tenant-supplied SQL: catalog views stay
    queryable, but file access outside the catalog and configuration changes
    are refused -- arbitrary SQL must not become arbitrary file access on a
    shared host.
    """
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[_check_row()],
    )

    connection = open_catalog_connection(tmp_path / "catalog", constrained=True)
    try:
        assert connection.execute("SELECT count(*) FROM episodes").fetchone() == (1,)
        assert connection.execute("SELECT example_metric FROM episodes").fetchone() == (1.0,)
        with pytest.raises(duckdb.Error, match=r"allowed_directories|[Pp]ermission"):
            connection.execute("SELECT * FROM read_csv('/etc/hosts')")
        with pytest.raises(duckdb.Error, match=r"lock|configuration"):
            connection.execute("SET enable_external_access = true")
        # The catalog itself must stay unwritable: DuckDB's directory
        # allowlist permits writes, so the catalog root is never on it --
        # tenant SQL cannot forge rows or clobber committed files.
        forged_row_file = tmp_path / "catalog" / "episodes" / "forged.parquet"
        with pytest.raises(duckdb.Error, match=r"allowed_directories|[Pp]ermission"):
            connection.execute(f"COPY (SELECT 1 AS x) TO '{forged_row_file}' (FORMAT PARQUET)")
        assert not forged_row_file.exists()
        # ...and it cannot READ the catalog's raw files either; the data
        # arrives through the materialized tables alone.
        episodes_glob = tmp_path / "catalog" / "episodes" / "*.parquet"
        with pytest.raises(duckdb.Error, match=r"allowed_directories|[Pp]ermission"):
            connection.execute(f"SELECT * FROM read_parquet('{episodes_glob}')")
    finally:
        connection.close()


def test_constrained_curate_writes_the_manifest_but_refuses_outside_reads(
    tmp_path: Path,
) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog = Catalog(catalog_dir)
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row()],
    )
    manifest_path = tmp_path / "out" / "manifest.parquet"

    report = curate(
        catalog_dir,
        "SELECT episode_id, uri FROM episodes",
        output=manifest_path,
        constrained=True,
    )
    assert report.row_count == 1
    assert manifest_path.is_file()

    with pytest.raises(duckdb.Error, match=r"allowed_directories|[Pp]ermission"):
        curate(
            catalog_dir,
            "SELECT * FROM read_csv('/etc/hosts')",
            output=tmp_path / "out" / "evil.parquet",
            constrained=True,
        )

    # The output's parent directory is NOT on the allowlist (only a private
    # staging subdirectory is), so tenant SQL cannot read what happens to
    # live beside its own manifest.
    sibling_secret = tmp_path / "out" / "sibling-secret.csv"
    sibling_secret.write_text("secret\n")
    with pytest.raises(duckdb.Error, match=r"allowed_directories|[Pp]ermission"):
        curate(
            catalog_dir,
            f"SELECT * FROM read_csv('{sibling_secret}')",
            output=tmp_path / "out" / "second.parquet",
            constrained=True,
        )


def test_numpy_scalar_measurements_round_trip(tmp_path: Path) -> None:
    """NumPy scalars from real check code store readable values, not NULLs."""
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
        # cast: deliberately passing user-supplied NumPy scalars past the
        # declared MeasurementValue type -- exactly what real check code does.
        measurements=cast(
            dict,
            {"ratio": np.float32(0.4), "frames": np.int64(3), "flag": np.bool_(True)},
        ),
    )
    result = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[row],
    )
    assert result.written is True
    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        raw_rows = connection.execute(
            "SELECT key, value_double, value_text, value_bool FROM measurements"
        ).fetchall()
        raw = {
            key: (value_double, value_text, value_bool)
            for key, value_double, value_text, value_bool in raw_rows
        }
        wide = connection.execute(
            "SELECT ratio, frames FROM episodes WHERE episode_id = ?",
            [result.episode_id],
        ).fetchone()
    finally:
        connection.close()
    # The measurements table holds one typed column per value.
    assert raw["ratio"] == (np.float32(0.4).item(), None, None)
    assert raw["frames"] == (3.0, None, None)
    assert raw["flag"] == (None, None, True)
    # The wide view exposes them to threshold predicates.
    assert wide == pytest.approx((np.float32(0.4).item(), 3.0))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_measurements_are_refused(tmp_path: Path, bad_value: float) -> None:
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="camera_blackout",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        measurements={"black_pct": bad_value},
    )
    with pytest.raises(
        ValueError,
        match=r"camera_blackout.*black_pct.*omit the key.*no finite value",
    ):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )


def test_numpy_nan_measurements_are_refused(tmp_path: Path) -> None:
    import numpy as np

    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="numpy_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        measurements=cast(dict, {"ratio": np.float64("nan")}),
    )
    with pytest.raises(ValueError, match=r"numpy_check.*ratio.*omit the key"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )


def test_zero_and_non_float_measurements_are_still_accepted(tmp_path: Path) -> None:
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="mixed_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        measurements={"zero": 0.0, "count": 0, "label": "ok", "flag": False},
    )
    result = Catalog(tmp_path / "catalog").append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[row],
    )
    assert result.written is True


def test_numpy_measured_episode_survives_a_manifest_filter(tmp_path: Path) -> None:
    """A NumPy-measured episode must not vanish from a threshold-filtered manifest."""
    import numpy as np

    catalog_dir = tmp_path / "catalog"
    for name, black_pct in (("numpy", np.float32(0.4)), ("python", 0.4)):
        canonical = tmp_path / f"{name}.canonical.mcap"
        canonical.write_bytes(name.encode())
        Catalog(catalog_dir).append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[
                CheckRunRow(
                    check_name="camera_blackout",
                    check_version="v1",
                    critical=False,
                    status=hflow.CheckStatus.MEASURED,
                    duration_s=0.1,
                    measurements=cast(dict, {"black_pct": black_pct}),
                )
            ],
        )
    connection = open_catalog_connection(catalog_dir)
    try:
        kept = connection.execute(
            "SELECT episode_id FROM episodes WHERE black_pct < 1.0 AND status != 'quarantined'"
        ).fetchall()
    finally:
        connection.close()
    assert len(kept) == 2


def test_same_value_in_a_numpy_or_python_scalar_replays_as_one_run(
    tmp_path: Path,
) -> None:
    """Equal values across scalar flavors fingerprint identically.

    Idempotence must not depend on which scalar flavor a check happened to
    return that day. (A genuinely different value -- float32 rounding of 0.4
    versus the float64 literal, say -- stays a distinct outcome by design.)
    """
    import numpy as np

    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    catalog = Catalog(tmp_path / "catalog")

    def measure(value: object) -> CheckRunRow:
        return CheckRunRow(
            check_name="camera_blackout",
            check_version="v1",
            critical=False,
            status=hflow.CheckStatus.MEASURED,
            duration_s=0.1,
            measurements=cast(dict, {"black_pct": value}),
        )

    float_attempts = [
        catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[measure(value)],
        )
        for value in (np.float64(0.5), 0.5, np.float64(0.5))
    ]
    assert [attempt.written for attempt in float_attempts] == [True, False, False]
    assert len({attempt.run_fingerprint for attempt in float_attempts}) == 1

    int_canonical = tmp_path / "int.canonical.mcap"
    int_canonical.write_bytes(b"int-episode-bytes")
    int_attempts = [
        catalog.append_episode(
            canonical_path=int_canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[measure(value)],
        )
        for value in (np.int64(7), 7)
    ]
    assert [attempt.written for attempt in int_attempts] == [True, False]
    assert len({attempt.run_fingerprint for attempt in int_attempts}) == 1


def test_numpy_scalar_interval_bounds_round_trip(tmp_path: Path) -> None:
    """A user check building interval bounds from a NumPy array stores real ints."""
    import numpy as np

    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="segment_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        intervals=[
            # cast: real check code indexes channel.to_numpy() and gets np.int64.
            hflow.Interval(
                start_ns=cast(int, np.int64(0)),
                end_ns=cast(int, np.int64(5)),
                label="segment",
            )
        ],
    )
    result = Catalog(tmp_path / "catalog").append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[row],
    )
    assert result.written is True
    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        stored = connection.execute(
            "SELECT start_ns, end_ns, label FROM intervals WHERE start_ns = 0 AND end_ns = 5"
        ).fetchall()
    finally:
        connection.close()
    assert stored == [(0, 5, "segment")]


def test_a_float_interval_bound_is_refused_naming_the_check_and_bound(
    tmp_path: Path,
) -> None:
    """A float bound is not a nanosecond timestamp, so it raises rather than coercing."""
    import numpy as np

    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="segment_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        # cast: the misuse this test exists to refuse.
        intervals=[hflow.Interval(start_ns=cast(int, np.float64(5.0)), end_ns=10)],
    )
    with pytest.raises(ValueError, match=r"segment_check.*start_ns.*float"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )


def test_a_bool_interval_bound_is_refused_rather_than_stored_as_one_nanosecond(
    tmp_path: Path,
) -> None:
    """``bool`` subclasses ``int``, so ``True`` passes an isinstance test and
    would store as 1 ns. It is a mistake, not a timestamp."""
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="segment_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        # cast: the misuse this test exists to refuse.
        intervals=[hflow.Interval(start_ns=cast(int, True), end_ns=10)],
    )
    with pytest.raises(ValueError, match=r"segment_check.*start_ns.*bool"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )


def _appended_with_interval(tmp_path: Path, interval: hflow.Interval) -> None:
    """Append one episode carrying ``interval``, for the bound-rule tests."""
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    Catalog(tmp_path / "catalog").append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[
            CheckRunRow(
                check_name="segment_check",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.1,
                intervals=[interval],
            )
        ],
    )


def test_an_inverted_interval_is_refused_naming_the_check_and_the_label(
    tmp_path: Path,
) -> None:
    """An end before its start is a negative duration to everything downstream.

    The label is in the message because a check emitting many intervals gives
    the reader no other way to tell which one is wrong (#161).
    """
    with pytest.raises(ValueError, match=r"segment_check.*'peak_velocity'.*end must be >= start"):
        _appended_with_interval(
            tmp_path, hflow.Interval(start_ns=10, end_ns=5, label="peak_velocity")
        )
    assert list((tmp_path / "catalog" / "episodes").glob("*.parquet")) == []


def test_a_negative_interval_bound_is_refused(tmp_path: Path) -> None:
    """Log time is nanoseconds since the epoch, so a negative bound is a bug."""
    with pytest.raises(ValueError, match=r"segment_check.*'gap'.*non-negative"):
        _appended_with_interval(tmp_path, hflow.Interval(start_ns=-5, end_ns=-1, label="gap"))


def test_a_zero_length_interval_is_allowed(tmp_path: Path) -> None:
    """An instant is a real thing to record, so only end < start is refused.

    Pinned rather than left implicit: the ordering rule is one ``<`` away from
    rejecting every event with no duration.
    """
    _appended_with_interval(tmp_path, hflow.Interval(start_ns=5, end_ns=5, label="touchdown"))

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        stored = connection.execute(
            "SELECT start_ns, end_ns, label FROM intervals WHERE label = 'touchdown'"
        ).fetchall()
    finally:
        connection.close()
    assert stored == [(5, 5, "touchdown")]


def test_same_interval_bound_in_a_numpy_or_python_scalar_replays_as_one_run(
    tmp_path: Path,
) -> None:
    """Equal interval bounds across scalar flavors fingerprint identically."""
    import numpy as np

    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    catalog = Catalog(tmp_path / "catalog")

    def segment(start: object, end: object) -> CheckRunRow:
        return CheckRunRow(
            check_name="segment_check",
            check_version="v1",
            critical=False,
            status=hflow.CheckStatus.MEASURED,
            duration_s=0.1,
            intervals=[
                hflow.Interval(start_ns=cast(int, start), end_ns=cast(int, end), label="segment")
            ],
        )

    attempts = [
        catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[segment(start, end)],
        )
        for start, end in ((np.int64(0), np.int64(5)), (0, 5))
    ]
    assert [attempt.written for attempt in attempts] == [True, False]
    assert len({attempt.run_fingerprint for attempt in attempts}) == 1


def test_non_scalar_measurement_is_refused_naming_the_check_and_key(
    tmp_path: Path,
) -> None:
    """An unstoreable measurement raises loudly instead of writing NULLs."""
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="broken_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        # cast: the misuse this test exists to refuse.
        measurements=cast(dict, {"bad": {"nested": 1}}),
    )
    with pytest.raises(ValueError, match=r"broken_check.*'bad'.*dict"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )


def test_measurement_key_claiming_an_episode_column_is_refused(tmp_path: Path) -> None:
    """A key named like an episodes column would pivot into <key>_1 beside it."""
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="claims_task",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        measurements={"task": 99.0},
    )
    with pytest.raises(ValueError, match=r"'claims_task'.*'task'"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )
    assert list((tmp_path / "catalog" / "episodes").glob("*.parquet")) == []


def test_episodes_view_reserved_columns_match_queryable_episode_columns(
    tmp_path: Path,
) -> None:
    """Keep the shadow guard aligned with the curated episodes view."""
    Catalog(tmp_path / "catalog").append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[],
    )

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        view_columns = [
            column[0] for column in connection.execute("SELECT * FROM episodes").description
        ]
    finally:
        connection.close()

    view_columns_by_lookup_key = {column.lower(): column for column in view_columns}
    reserved_columns_by_lookup_key = dict(_EPISODES_VIEW_RESERVED_COLUMNS)
    # The wide episodes view deliberately excludes the stored quarantine flag, but
    # it stays reserved because the derived status column is computed from it.
    reserved_only_exemptions = {"quarantined"}

    reserved_not_in_view = sorted(
        reserved_columns_by_lookup_key[key]
        for key in reserved_columns_by_lookup_key.keys()
        - view_columns_by_lookup_key.keys()
        - reserved_only_exemptions
    )
    view_not_reserved = sorted(
        view_columns_by_lookup_key[key]
        for key in view_columns_by_lookup_key.keys() - reserved_columns_by_lookup_key.keys()
    )

    assert not reserved_not_in_view and not view_not_reserved, (
        "episodes view reserved-column drift: "
        f"reserved_not_in_view={reserved_not_in_view}, "
        f"view_not_reserved={view_not_reserved}"
    )


def test_measurement_key_shadowing_is_case_insensitive(tmp_path: Path) -> None:
    """DuckDB identifiers are case-insensitive, so 'Task' shadows 'task' too."""
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="claims_task",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        measurements={"Task": 1.0},
    )
    with pytest.raises(ValueError, match=r"'Task'.*shadows 'task'"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )


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


def test_replaying_an_append_heals_dependents_left_stale_by_a_crashed_repair(
    tmp_path: Path,
) -> None:
    """#51's residual window: a winner that created the episodes file but
    crashed before force-aligning the dependents leaves them carrying a stale
    recorded_at. A later replay of the same outcome (the normal retry lane)
    must reconcile every dependent to the episodes file's recorded_at instead
    of early-returning past the damage forever.
    """
    import duckdb

    canonical = _fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")
    row = _check_row()

    def append_same_outcome() -> "hflow.catalog.AppendResult":
        return catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )

    first = append_same_outcome()
    stem = f"{first.episode_id}-{first.run_fingerprint}"

    # Simulate the crash debris: two dependents still carry an earlier
    # attempt's recorded_at (one hour older), exactly what a repair pass that
    # died mid-publish leaves behind.
    connection = duckdb.connect()
    try:
        for table_name in ("check_runs", "tags"):
            table_file = tmp_path / "catalog" / table_name / f"{stem}.parquet"
            stale_copy = tmp_path / f"stale-{table_name}.parquet"
            connection.execute(
                f"COPY (SELECT * REPLACE (recorded_at - INTERVAL 1 HOUR AS recorded_at) "
                f"FROM read_parquet('{table_file}')) TO '{stale_copy}' (FORMAT PARQUET)"
            )
            table_file.write_bytes(stale_copy.read_bytes())
    finally:
        connection.close()

    # The replay happens in a fresh worker process in reality; the crashed
    # repairer's process-local memo of "already aligned" does not carry over.
    hflow.catalog._reconciled_append_stems.clear()

    replay = append_same_outcome()
    assert replay.written is False
    assert replay.run_fingerprint == first.run_fingerprint

    connection = duckdb.connect()
    try:
        timestamps = set()
        for table_name in ("episodes", "check_runs", "measurements", "tags", "intervals"):
            table_file = tmp_path / "catalog" / table_name / f"{stem}.parquet"
            rows = connection.execute(
                f"SELECT DISTINCT CAST(recorded_at AS VARCHAR) FROM read_parquet('{table_file}')"
            ).fetchall()
            timestamps.update(value for (value,) in rows)
        assert len(timestamps) == 1, f"replay left mixed recorded_at across tables: {timestamps}"
    finally:
        connection.close()


def test_replaying_an_append_refuses_a_corrupt_empty_commit_marker(tmp_path: Path) -> None:
    """append_episode always inserts exactly one episodes row, so a zero-row
    episodes file is corruption -- a replay must refuse loudly instead of
    silently skipping reconciliation against it."""
    import duckdb

    canonical = _fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")
    row = _check_row()

    def append_same_outcome() -> "hflow.catalog.AppendResult":
        return catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )

    first = append_same_outcome()
    stem = f"{first.episode_id}-{first.run_fingerprint}"
    episodes_file = tmp_path / "catalog" / "episodes" / f"{stem}.parquet"
    connection = duckdb.connect()
    try:
        empty_copy = tmp_path / "empty-episodes.parquet"
        connection.execute(
            f"COPY (SELECT * FROM read_parquet('{episodes_file}') WHERE false) "
            f"TO '{empty_copy}' (FORMAT PARQUET)"
        )
        episodes_file.write_bytes(empty_copy.read_bytes())
    finally:
        connection.close()
    hflow.catalog._reconciled_append_stems.clear()  # a replay is a fresh process

    with pytest.raises(ValueError, match="holds no rows"):
        append_same_outcome()


def test_concurrent_append_of_the_identical_outcome_keeps_one_recorded_at(
    tmp_path: Path,
) -> None:
    """Two callers racing ``append_episode`` for the identical outcome (a
    retried or duplicate-dispatched batch task, not a crash) must not split
    one outcome's tables across two different recorded_at values.

    Before the fix, a dependent table that lost its create-if-absent race
    was unconditionally deleted and rewritten with the losing caller's OWN
    recorded_at, independently per table -- so the caller that ultimately
    won ``episodes`` (the durability marker) could end up with dependents
    stamped by the OTHER caller, and different dependent tables could even
    disagree with each other.
    """
    import threading

    import duckdb

    canonical = _fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")
    row = _check_row()

    # A storage-boundary test double: gate the first two dependent-table
    # writes on a barrier so both threads are guaranteed to reach
    # append_episode's dependent-write loop at the same time, forcing the
    # real interleaving a natural race only produces intermittently.
    barrier = threading.Barrier(2)
    released = 0
    release_lock = threading.Lock()
    real_location = catalog.location

    class GatedLocation:
        def __getattr__(self, name: str) -> object:
            return getattr(real_location, name)

        def store_file_if_absent(self, local_file: Path, relative: str) -> bool:
            nonlocal released
            with release_lock:
                released += 1
                should_wait = released <= 2
            if should_wait:
                barrier.wait(timeout=5)
            return real_location.store_file_if_absent(local_file, relative)

    catalog.location = cast("hflow.storage.StorageRoot", GatedLocation())

    results: dict[str, hflow.catalog.AppendResult] = {}

    def append_same_outcome(label: str) -> None:
        results[label] = catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )

    threads = [threading.Thread(target=append_same_outcome, args=(label,)) for label in "AB"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert {result.written for result in results.values()} == {True, False}
    stem = f"{results['A'].episode_id}-{results['A'].run_fingerprint}"

    connection = duckdb.connect()
    try:
        timestamps = set()
        for table_name in ("episodes", "check_runs", "measurements", "tags", "intervals"):
            table_file = tmp_path / "catalog" / table_name / f"{stem}.parquet"
            assert table_file.is_file(), f"{table_name} file missing after the race"
            rows = connection.execute(
                f"SELECT DISTINCT CAST(recorded_at AS VARCHAR) FROM read_parquet('{table_file}')"
            ).fetchall()
            timestamps.update(value for (value,) in rows)
        assert len(timestamps) == 1, f"mixed recorded_at across tables: {timestamps}"
    finally:
        connection.close()


def test_measurements_latest_ranks_by_the_owning_episodes_recorded_at(tmp_path: Path) -> None:
    """A dependent table's own recorded_at can go permanently stale: a repair
    pass that wins the episodes race can crash before reaching measurements
    (#51), and a later retry early-returns on exists(episodes) and never
    revisits it. measurements_latest must still agree with episodes_latest
    on which run is newest -- ranking (and reporting) off the episode's own
    recorded_at, the one column create-if-absent guarantees a single writer
    for, rather than this table's own, possibly-stale, column.
    """
    import time

    import duckdb

    canonical = _fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")

    older = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row(version="v1", value=1.0)],
    )
    time.sleep(0.01)
    newer = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[_check_row(version="v2", value=2.0)],
    )
    assert older.written and newer.written
    assert older.run_fingerprint != newer.run_fingerprint

    # Simulate a repair pass that won `episodes` for the NEWER run but
    # crashed before repairing `measurements`: that table's file is left
    # with a stale recorded_at older than even the OLDER run's, despite
    # `episodes` (the source of truth) correctly carrying the newest one.
    stale_file = (
        tmp_path
        / "catalog"
        / "measurements"
        / f"{newer.episode_id}-{newer.run_fingerprint}.parquet"
    )
    connection = duckdb.connect()
    try:
        with tempfile.TemporaryDirectory(prefix="hflow-test-corrupt-") as staging_name:
            staged = Path(staging_name) / "measurements.parquet"
            connection.execute(
                f"""
                COPY (
                    SELECT * REPLACE ('2000-01-01 00:00:00+00'::TIMESTAMPTZ AS recorded_at)
                    FROM read_parquet('{stale_file}')
                ) TO '{staged}' (FORMAT PARQUET)
                """
            )
            staged.replace(stale_file)
    finally:
        connection.close()

    connection = open_catalog_connection(tmp_path / "catalog")
    try:
        # episodes_latest ranks off its own always-authoritative recorded_at,
        # so it is unaffected and still correctly calls the newer run latest.
        assert connection.execute("SELECT run_fingerprint FROM episodes_latest").fetchone() == (
            newer.run_fingerprint,
        )
        # measurements_latest must agree, despite its own corrupted column --
        # before the fix, it picked the OLDER run's value (1.0) here.
        assert connection.execute(
            "SELECT value_double FROM measurements_latest WHERE key = 'example_metric'"
        ).fetchone() == (2.0,)
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


def _episode_with_check(
    tmp_path: Path,
    *,
    critical: bool,
    status: hflow.CheckStatus,
    quarantined: bool = False,
) -> Path:
    """Append one episode carrying a single check run with the given shape."""
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    check_row = CheckRunRow(
        check_name="blur",
        check_version="v1",
        critical=critical,
        status=status,
        duration_s=0.1,
        error="boom" if status is hflow.CheckStatus.ERROR else None,
    )
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[check_row],
        quarantine_tags=["quarantined:blur"] if quarantined else [],
    )
    return catalog_root


def _status_of_only_episode(catalog_root: Path, *, constrained: bool = False) -> str:
    connection = open_catalog_connection(catalog_root, constrained=constrained)
    try:
        row = connection.execute("SELECT status FROM episodes").fetchone()
    finally:
        connection.close()
    assert row is not None
    return str(row[0])


def test_errored_critical_check_reports_unverified(tmp_path: Path) -> None:
    """#164 item 1: a crashed critical check no longer reads as a pass."""
    catalog_root = _episode_with_check(tmp_path, critical=True, status=hflow.CheckStatus.ERROR)
    assert _status_of_only_episode(catalog_root) == "unverified"


def test_passing_critical_check_still_reports_ok(tmp_path: Path) -> None:
    catalog_root = _episode_with_check(tmp_path, critical=True, status=hflow.CheckStatus.PASSED)
    assert _status_of_only_episode(catalog_root) == "ok"


def test_quarantine_outranks_a_critical_error(tmp_path: Path) -> None:
    """#164 item 2: quarantined wins when an episode is both."""
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    catalog.append_episode(
        canonical_path=_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[
            CheckRunRow(
                check_name="blur",
                check_version="v1",
                critical=True,
                status=hflow.CheckStatus.FAILED,
                duration_s=0.1,
            ),
            CheckRunRow(
                check_name="exposure",
                check_version="v1",
                critical=True,
                status=hflow.CheckStatus.ERROR,
                duration_s=0.1,
                error="boom",
            ),
        ],
        quarantine_tags=["quarantined:blur"],
    )
    assert _status_of_only_episode(catalog_root) == "quarantined"


def test_non_critical_error_still_reports_ok(tmp_path: Path) -> None:
    """#164 item 3: only a CRITICAL crash leaves an episode unverified."""
    catalog_root = _episode_with_check(tmp_path, critical=False, status=hflow.CheckStatus.ERROR)
    assert _status_of_only_episode(catalog_root) == "ok"


@pytest.mark.parametrize("status", [hflow.CheckStatus.SKIPPED, hflow.CheckStatus.SUPERSEDED])
def test_critical_check_that_did_not_crash_reports_ok(
    tmp_path: Path, status: hflow.CheckStatus
) -> None:
    """Skipped and superseded are not crashes, so neither reads unverified."""
    catalog_root = _episode_with_check(tmp_path, critical=True, status=status)
    assert _status_of_only_episode(catalog_root) == "ok"


def test_successful_rerun_clears_unverified(tmp_path: Path) -> None:
    """#164 item 4: a later good run of the same check reports ok again."""
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    canonical = _fake_canonical(tmp_path)
    errored = CheckRunRow(
        check_name="blur",
        check_version="v1",
        critical=True,
        status=hflow.CheckStatus.ERROR,
        duration_s=0.1,
        error="boom",
    )
    catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[errored],
    )
    assert _status_of_only_episode(catalog_root) == "unverified"

    passed = CheckRunRow(
        check_name="blur",
        check_version="v1",
        critical=True,
        status=hflow.CheckStatus.PASSED,
        duration_s=0.1,
    )
    catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={"attempt": "2"},
        check_rows=[passed],
    )
    assert _status_of_only_episode(catalog_root) == "ok"


def test_both_view_definitions_agree_on_unverified(tmp_path: Path) -> None:
    """#164 item 5: the narrow and wide paths answer identically.

    The wide path exists only once a measurement key is recorded, so this
    appends an episode with a measurement and one without, and reads both
    through a plain and a constrained connection.
    """
    narrow_root = _episode_with_check(
        tmp_path / "narrow", critical=True, status=hflow.CheckStatus.ERROR
    )

    wide_base = tmp_path / "wide"
    wide_base.mkdir()
    wide_root = wide_base / "catalog"
    catalog = Catalog(wide_root)
    catalog.append_episode(
        canonical_path=_fake_canonical(wide_base),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[
            CheckRunRow(
                check_name="blur",
                check_version="v1",
                critical=True,
                status=hflow.CheckStatus.ERROR,
                duration_s=0.1,
                error="boom",
                measurements={"example_metric": 1.0},
            )
        ],
    )

    for root in (narrow_root, wide_root):
        assert _status_of_only_episode(root) == "unverified"
        assert _status_of_only_episode(root, constrained=True) == "unverified"


def test_one_errored_episode_does_not_mark_its_neighbours_unverified(
    tmp_path: Path,
) -> None:
    """The status subquery correlates per episode, not across the catalog.

    Every other test here uses a single-episode catalog, which cannot tell a
    correctly correlated EXISTS apart from one that binds to its own relation
    and is therefore true for every row.
    """
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    canonicals = {}
    for name, status in (
        ("crashed", hflow.CheckStatus.ERROR),
        ("healthy", hflow.CheckStatus.PASSED),
    ):
        canonical = tmp_path / f"{name}.canonical.mcap"
        canonical.write_bytes(f"canonical for {name}".encode())
        append = catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={"task": name},
            check_rows=[
                CheckRunRow(
                    check_name="blur",
                    check_version="v1",
                    critical=True,
                    status=status,
                    duration_s=0.1,
                    error="boom" if status is hflow.CheckStatus.ERROR else None,
                )
            ],
        )
        canonicals[name] = append.episode_id

    connection = open_catalog_connection(catalog_root)
    try:
        statuses = dict(connection.execute("SELECT episode_id, status FROM episodes").fetchall())
    finally:
        connection.close()

    assert statuses[canonicals["crashed"]] == "unverified"
    assert statuses[canonicals["healthy"]] == "ok"


def test_status_builder_refuses_an_unqualified_column() -> None:
    """An unqualified column would correlate the subquery against itself."""
    with pytest.raises(ValueError, match="qualified"):
        episode_status_case_sql(
            quarantined_column="quarantined", check_runs_relation="check_runs_latest"
        )
