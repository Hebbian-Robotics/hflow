"""Catalog appends and curation queries (issues #16/#17)."""

import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pytest
from catalog_test_helpers import (
    FAKE_STAMPS,
    example_check_row,
    recorded_at_values,
    write_fake_canonical,
)

import hflow
from hflow.catalog import (
    _EPISODES_VIEW_RESERVED_COLUMNS,
    TABLE_COLUMN_DDL,
    Catalog,
    CheckRunRow,
    _run_fingerprint,
    content_episode_id,
    episode_status_case_sql,
)
from hflow.checks import camera_frame_stats
from hflow.cli import main as cli_main
from hflow.curation import (
    CurationReport,
    NonSingleSelectQueryError,
    curate,
    open_catalog_connection,
    reject_non_single_select,
)
from hflow.format import CATALOG_FORMAT_VERSION
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import EpisodeStamps


def test_append_is_idempotent_for_the_same_content_versions_and_outcome(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = write_fake_canonical(tmp_path)
    first = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[example_check_row()],
    )
    second = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[example_check_row()],
    )
    assert first.written and not second.written
    assert first.episode_id == second.episode_id == content_episode_id(canonical)
    parquet_files = list((tmp_path / "catalog" / "measurements").glob("*.parquet"))
    assert len(parquet_files) == 1


def test_checks_without_observations_keep_the_pre_observation_fingerprint() -> None:
    check_row = CheckRunRow(
        check_name="compat",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=1.0,
        measurements={"score": 1.0},
        tags=["seen"],
        intervals=[hflow.Interval(start_ns=0, end_ns=10, label="span")],
    )

    assert _run_fingerprint("episode-id", "pipeline-v1", [check_row], []) == "b47ee98776b1"


def test_selective_appends_do_not_replay_an_obsolete_full_outcome(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = write_fake_canonical(tmp_path)
    measured = replace(example_check_row(), critical=True)
    other = replace(example_check_row(), check_name="other", measurements={"other_score": 1.0})
    errored = CheckRunRow(
        check_name=measured.check_name,
        check_version=measured.check_version,
        critical=True,
        status=hflow.CheckStatus.ERROR,
        duration_s=0.1,
        error="temporary service failure",
    )
    full_outcome = [measured, other]
    for rows, expected_status in [
        (full_outcome, "ok"),
        ([errored], "unverified"),
        ([other], "unverified"),
        (full_outcome, "ok"),
    ]:
        result = catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=rows,
        )
        assert result.written
        assert _status_of_only_episode(catalog.root) == expected_status

    with open_catalog_connection(catalog.root) as connection:
        assert connection.execute("SELECT count(*) FROM episodes_raw").fetchone() == (4,)
        assert connection.execute(
            "SELECT DISTINCT run_fingerprint FROM check_runs_latest"
        ).fetchall() == [(result.run_fingerprint,)]
        assert connection.execute(
            "SELECT DISTINCT run_fingerprint FROM observations_latest"
        ).fetchall() == [(result.run_fingerprint,)]


@pytest.mark.parametrize("execution_id", ["", "  ", "\t\n"])
def test_append_rejects_blank_execution_id(tmp_path: Path, execution_id: str) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = write_fake_canonical(tmp_path)

    with pytest.raises(ValueError, match=r"^execution_id must be non-empty when supplied$"):
        catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[example_check_row()],
            execution_id=execution_id,
        )

    assert not list((catalog.root / "episodes").glob("*.parquet"))


@pytest.mark.parametrize("bucket", [False, True])
def test_execution_identity_replays_history_without_reordering_it(
    tmp_path: Path,
    bucket_over_tmp: tuple[hflow.storage.BucketStorageRoot, Path],
    bucket: bool,
) -> None:
    root = bucket_over_tmp[0].child("catalog") if bucket else tmp_path / "catalog"
    canonical = write_fake_canonical(tmp_path)

    def append(execution_id: str, value: float = 1.0) -> hflow.catalog.AppendResult:
        return Catalog(root).append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[example_check_row(value=value)],
            execution_id=execution_id,
        )

    first = append("first")
    assert first.written
    newer = append("second", value=2.0)
    assert newer.written
    location = Catalog(root).location
    original_files = {
        table: location.read_bytes(f"{table}/{first.episode_id}-{first.run_fingerprint}.parquet")
        for table in TABLE_COLUMN_DDL
    }
    hflow.catalog._reconciled_append_stems.clear()
    replay = append("first")
    assert not replay.written
    assert replay.run_fingerprint == first.run_fingerprint
    with open_catalog_connection(root) as connection:
        assert connection.execute("SELECT example_metric FROM episodes").fetchone() == (2.0,)
    returned = append("third")
    assert returned.written
    assert returned.run_fingerprint != first.run_fingerprint
    assert not append("third").written
    for table, content in original_files.items():
        assert (
            location.read_bytes(f"{table}/{first.episode_id}-{first.run_fingerprint}.parquet")
            == content
        )
    with open_catalog_connection(root) as connection:
        assert connection.execute("SELECT example_metric FROM episodes").fetchone() == (1.0,)
        assert connection.execute("SELECT count(*) FROM episodes_raw").fetchone() == (3,)
        for table in TABLE_COLUMN_DDL:
            relation = "episodes_raw" if table == "episodes" else table
            assert connection.execute(
                f"SELECT count(DISTINCT run_fingerprint) FROM {relation}"
            ).fetchone() == (3,)


def test_timestamped_observations_round_trip_as_typed_long_rows(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    check_result = hflow.CheckResult(
        observations=[
            hflow.Observation(
                observation_id="frame:3",
                timestamp_ns=30,
                values={"score": 1.0, "reviewed": True, "note": "clear"},
            )
        ]
    )
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[
            CheckRunRow.from_result(
                check_name="example_check",
                check_version="v1",
                critical=False,
                status=hflow.CheckStatus.MEASURED,
                duration_s=0.01,
                error=None,
                result=check_result,
            )
        ],
    )

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute(
            """
            SELECT observation_id, timestamp_ns, key, value_double, value_text, value_bool
            FROM observations_latest
            ORDER BY key
            """
        ).fetchall() == [
            ("frame:3", 30, "note", None, "clear", None),
            ("frame:3", 30, "reviewed", None, None, True),
            ("frame:3", 30, "score", 1.0, None, None),
        ]
        assert connection.execute(
            """
            SELECT observation_id, timestamp_ns, note, reviewed, score
            FROM (
                PIVOT observations_latest
                ON key IN ('note', 'reviewed', 'score')
                USING first(coalesce(CAST(value_double AS VARCHAR), value_text,
                                     CAST(value_bool AS VARCHAR)))
                GROUP BY episode_id, check_name, check_version,
                         observation_id, timestamp_ns
            )
            """
        ).fetchall() == [("frame:3", 30, "clear", "true", "1.0")]


def test_observations_latest_switches_a_check_result_as_one_unit(tmp_path: Path) -> None:
    import time

    catalog = Catalog(tmp_path / "catalog")
    canonical = write_fake_canonical(tmp_path)
    catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row(version="model-a")],
    )
    time.sleep(0.01)
    replacement_row = example_check_row(version="model-b")
    replacement_row = hflow.CheckRunRow(
        check_name=replacement_row.check_name,
        check_version=replacement_row.check_version,
        critical=replacement_row.critical,
        status=replacement_row.status,
        duration_s=replacement_row.duration_s,
        observations=[
            hflow.Observation(
                observation_id="frame:3",
                timestamp_ns=30,
                values={"score": 0.5},
            )
        ],
    )
    catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[replacement_row],
    )

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute(
            "SELECT check_version, key, value_double FROM observations_latest"
        ).fetchall() == [("model-b", "score", 0.5)]
        assert connection.execute("SELECT count(*) FROM observations").fetchone() == (4,)


def test_duplicate_observation_ids_are_refused_before_catalog_writes(tmp_path: Path) -> None:
    duplicate_observations = [
        hflow.Observation(observation_id="frame:3", timestamp_ns=30, values={"score": 1.0}),
        hflow.Observation(observation_id="frame:3", timestamp_ns=30, values={"score": 0.5}),
    ]
    check_row = example_check_row()
    check_row = hflow.CheckRunRow(
        check_name=check_row.check_name,
        check_version=check_row.check_version,
        critical=check_row.critical,
        status=check_row.status,
        duration_s=check_row.duration_s,
        observations=duplicate_observations,
    )

    with pytest.raises(ValueError, match="duplicate observation id 'frame:3'"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=write_fake_canonical(tmp_path),
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[check_row],
        )

    assert not list((tmp_path / "catalog" / "episodes").glob("*.parquet"))


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
    canonical = write_fake_canonical(tmp_path)
    first = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
        orchestrator_run_id="manual__2026-08-23T00:00:00+00:00",
    )
    replayed_under_another_run = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
        orchestrator_run_id="scheduled__2026-08-24T00:00:00+00:00",
    )

    assert first.written and not replayed_under_another_run.written
    assert first.run_fingerprint == replayed_under_another_run.run_fingerprint
    assert len(list((tmp_path / "catalog" / "episodes").glob("*.parquet"))) == 1

    with open_catalog_connection(tmp_path / "catalog") as connection:
        # The row keeps the run that FIRST recorded the outcome: the second
        # append did nothing, so claiming it as that run's output would be a
        # fiction. Documented on append_episode. The id is stored verbatim,
        # never normalized beyond blankness: it has to compare equal to the id
        # the orchestrator's own API reports, or the join it exists for breaks.
        assert connection.execute("SELECT orchestrator_run_id FROM episodes").fetchall() == [
            ("manual__2026-08-23T00:00:00+00:00",)
        ]


def test_episode_time_bounds_are_recorded_as_the_episode_axis(tmp_path: Path) -> None:
    """A synthesized episode's first and last message stamps land as
    ``start_ns``/``end_ns``; the bounds describe the canonical bytes the
    episode id already hashes, so they never enter the run fingerprint."""
    raw = synthesize_episode(tmp_path / "raw.mcap", SyntheticEpisodeSpec(duration_s=2.0))
    canonical = tmp_path / "episode.canonical.mcap"
    hflow.write_canonical_episode(raw, canonical)
    with hflow.Episode(canonical) as episode:
        time_bounds = episode.time_bounds
        assert time_bounds is not None
        stamps = np.concatenate([episode.channel(topic).timestamps for topic in episode.topics])
    assert time_bounds.start_ns == int(stamps.min())
    assert time_bounds.end_ns == int(stamps.max())
    assert time_bounds.duration_s == pytest.approx(2.0, abs=0.2)

    catalog = Catalog(tmp_path / "catalog")
    with_bounds = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
        time_bounds=time_bounds,
    )
    replayed_without_bounds = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
    )
    assert with_bounds.written and not replayed_without_bounds.written
    assert with_bounds.run_fingerprint == replayed_without_bounds.run_fingerprint

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute("SELECT start_ns, end_ns FROM episodes_latest").fetchall() == [
            (time_bounds.start_ns, time_bounds.end_ns)
        ]


def test_a_catalog_written_before_time_bounds_still_reads_beside_new_rows(
    tmp_path: Path,
) -> None:
    """The episodes files are unioned by column name, so a row an older hflow
    wrote (no ``start_ns``/``end_ns`` columns at all) reads as NULL bounds
    next to a row that carries them."""
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    older = catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path, b"older canonical"),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
        time_bounds=hflow.EpisodeTimeBounds(start_ns=5, end_ns=50),
    )
    # Rewrite the older row's file the way a pre-bounds hflow laid it out.
    (older_file,) = (catalog_root / "episodes").glob(f"{older.episode_id}-*.parquet")
    older_layout = tmp_path / "older-layout.parquet"
    rewrite = duckdb.connect()
    try:
        rewrite.execute(
            f"COPY (SELECT * EXCLUDE (start_ns, end_ns) FROM read_parquet('{older_file}')) "
            f"TO '{older_layout}' (FORMAT PARQUET)"
        )
    finally:
        rewrite.close()
    older_layout.replace(older_file)

    newer = catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path, b"newer canonical"),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
        time_bounds=hflow.EpisodeTimeBounds(start_ns=100, end_ns=900),
    )

    with open_catalog_connection(catalog_root) as connection:
        bounds_by_episode = dict(
            connection.execute(
                "SELECT episode_id, (start_ns, end_ns) FROM episodes_latest"
            ).fetchall()
        )
    assert bounds_by_episode[older.episode_id] == (None, None)
    assert bounds_by_episode[newer.episode_id] == (100, 900)


@pytest.mark.parametrize(
    "blank",
    [
        # The dev loop and any non-runtime caller pass nothing (the None
        # default) and stay valid.
        pytest.param(None, id="unorchestrated"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param("\t\n", id="whitespace"),
    ],
)
def test_a_blank_run_id_records_as_absent_rather_than_as_a_value(
    tmp_path: Path, blank: str | None
) -> None:
    """One stored representation of "no orchestrator".

    A blank falls out of an adapter reading its own environment. Stored
    verbatim it would leave NULL and '' both meaning unorchestrated, so an
    IS NULL query would miss rows and a filter could match a value that names
    no run.
    """
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
        orchestrator_run_id=blank,
    )

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute("SELECT orchestrator_run_id FROM episodes").fetchall() == [
            (None,)
        ]


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
        canonical_path=write_fake_canonical(tmp_path, b"new bytes"),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
        orchestrator_run_id="manual__2026-08-23T00:00:00+00:00",
    )

    # An episodes file in the pre-column shape, beside the new one. A corpus
    # that predates the run id column predates the time-bounds columns too.
    legacy_columns = (
        TABLE_COLUMN_DDL["episodes"]
        .replace("orchestrator_run_id VARCHAR, ", "")
        .replace(", start_ns BIGINT, end_ns BIGINT", "")
    )
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

    with open_catalog_connection(catalog_root) as connection:
        rows = connection.execute(
            "SELECT episode_id, orchestrator_run_id FROM episodes ORDER BY episode_id"
        ).fetchall()
    assert ("legacyepisode", None) in rows
    assert any(run_id == "manual__2026-08-23T00:00:00+00:00" for _episode_id, run_id in rows)


def test_rerunning_a_changed_check_appends_new_version_rows(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = write_fake_canonical(tmp_path)
    catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row(version="v1", value=1.0)],
    )
    result = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row(version="v2", value=2.0)],
    )
    assert result.written

    with open_catalog_connection(tmp_path / "catalog") as connection:
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


def test_successful_retry_after_error_appends_repaired_outcome(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    canonical = write_fake_canonical(tmp_path)
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
    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute(
            "SELECT status FROM check_runs ORDER BY recorded_at"
        ).fetchall() == [("error",), ("measured",)]
        assert connection.execute("SELECT score FROM episodes").fetchone() == (1.0,)


@pytest.mark.parametrize(
    ("open_catalog", "expected_message"),
    [
        pytest.param(
            Catalog,
            f"has format version '999'.*this build reads/writes version '{CATALOG_FORMAT_VERSION}'",
            id="catalog-writer",
        ),
        pytest.param(
            open_catalog_connection,
            f"has format version '999'.*this build reads version '{CATALOG_FORMAT_VERSION}'",
            id="curation-reader",
        ),
    ],
)
def test_every_catalog_entry_point_refuses_an_unknown_format_version(
    tmp_path: Path, open_catalog: Callable[[Path], object], expected_message: str
) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "format_version").write_text("999\n")
    with pytest.raises(ValueError, match=expected_message):
        open_catalog(root)


def test_curate_on_empty_catalog(tmp_path: Path) -> None:
    Catalog(tmp_path / "catalog")
    report = curate(tmp_path / "catalog", "SELECT 1 AS one")
    assert report.total_episodes == 0
    assert report.coverage == []
    assert report.row_count == 1

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute("SELECT count(*) FROM episodes").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM episodes_latest").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM measurements_latest").fetchone() == (0,)


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
    async def joints(ep: hflow.Episode) -> hflow.CheckResult:
        return await hflow.checks.joint_discontinuity(ep)

    @app.check(version="1", critical=True)
    async def camera_blackout(ep: hflow.Episode) -> hflow.CheckResult:
        camera_topic = next(topic for topic in ep.cameras if "wrist_cam" in topic)
        camera_evidence = await camera_frame_stats(ep, cameras=[camera_topic])
        black_frame_percent = camera_evidence.measurements[f"{camera_topic}/black_frame_pct"]
        assert isinstance(black_frame_percent, float)
        return hflow.CheckResult(
            measurements={"black_pct": black_frame_percent},
            verdict=black_frame_percent < 50.0,
        )

    @app.check(version="1")
    async def late_check(ep: hflow.Episode) -> hflow.CheckResult:
        # Registered after the gate: skipped on quarantined episodes, so its
        # coverage must come out below 100%.
        return hflow.CheckResult(measurements={"late_metric": 1.0})

    for task_name, spec in episode_specs.items():
        source = synthesize_episode(sources_dir / f"{task_name}.mcap", spec)
        report = asyncio.run(app.test(source, verbose=False, record=True))
        assert report.catalog_entry is not None and report.catalog_entry.written

    rerun_report = asyncio.run(
        app.test(sources_dir / "fold_napkin.mcap", verbose=False, record=True)
    )
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
    with open_catalog_connection(recorded_data_root / "catalog") as connection:
        rows = dict(
            connection.execute("SELECT task, status FROM episodes ORDER BY task").fetchall()
        )
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
    behind = write_fake_canonical(tmp_path, content=b"behind the current pipeline")
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
    first_generation = write_fake_canonical(tmp_path, content=b"first canonical bytes")
    catalog.append_episode(
        canonical_path=first_generation,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[example_check_row(value=1.0)],
        source_uri="episodes-in/fold.mcap",
        uri="/data/episodes/fold.canonical.mcap",
    )
    reprocessed = tmp_path / "fold-reprocessed.canonical.mcap"
    reprocessed.write_bytes(b"reprocessed canonical bytes")
    second_append = catalog.append_episode(
        canonical_path=reprocessed,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[example_check_row(value=2.0)],
        source_uri="episodes-in/fold.mcap",
        uri="/data/episodes/fold.canonical.mcap",
    )

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute("SELECT count(*) FROM episodes_raw").fetchone() == (2,)
        latest = connection.execute("SELECT episode_id FROM episodes_latest").fetchall()
        assert latest == [(second_append.episode_id,)]
        # The wide view is the documented raw-SQL surface, so it must agree:
        # otherwise every user query counts the recording twice.
        wide = connection.execute("SELECT task, example_metric FROM episodes").fetchall()
        assert wide == [("fold_napkin", 2.0)]

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
        canonical_path=write_fake_canonical(tmp_path),
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


@pytest.mark.parametrize(
    ("pipeline_source", "app_selector", "expected_stderr_fragments"),
    [
        pytest.param(
            "raise RuntimeError('boom at import time')\n",
            "",
            ("boom at import time",),
            id="broken-pipeline-file",
        ),
        pytest.param(
            "value = 42\n",
            ":custom_name",
            ("has no hflow.App named 'custom_name'",),
            id="named-app-missing",
        ),
        pytest.param("value = 42\n", "", ("defines no hflow.App",), id="no-apps"),
        pytest.param(
            "import hflow\n\nkitchen = hflow.App('kitchen')\ngarage = hflow.App('garage')\n",
            "",
            ("defines 2 hflow.App objects", "'kitchen'", "'garage'"),
            id="ambiguous-apps",
        ),
    ],
)
def test_cli_stale_reports_an_unusable_pipeline_instead_of_crashing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pipeline_source: str,
    app_selector: str,
    expected_stderr_fragments: tuple[str, ...],
) -> None:
    Catalog(tmp_path / "catalog")
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text(pipeline_source)

    exit_code = cli_main(
        [
            "stale",
            "--catalog",
            str(tmp_path / "catalog"),
            "--pipeline",
            f"{pipeline}{app_selector}",
        ]
    )

    assert exit_code == 2
    stderr = capsys.readouterr().err
    for fragment in expected_stderr_fragments:
        assert fragment in stderr


def test_cli_stale_exit_code_returns_one_when_episodes_are_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
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
        canonical_path=write_fake_canonical(tmp_path),
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
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[example_check_row()],
    )

    with open_catalog_connection(tmp_path / "catalog", constrained=True) as connection:
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


# The SQL surface curate() accepts, written down (#279). Each row is
# (sql_template, expected_refused, expected_rows). ``{decoy}`` is replaced at
# runtime with a path inside the constrained connection's writable directory.
#
# This table is the one statement of the surface. When the gate changes, this
# is the thing that has to change with it, deliberately and in one place.
_SQL_SURFACE_PAYLOADS = [
    # Injection shapes, from the #271 review. Refused.
    pytest.param(
        "SELECT 1) TO {decoy} ...; CREATE TABLE p(x TEXT); --",
        True,
        0,
        id="copy-escape-old-pr-shape",
    ),
    pytest.param(
        "SELECT 1; CREATE TABLE pwned AS SELECT 1 AS x",
        True,
        0,
        id="direct-multistatement-select-plus-create",
    ),
    pytest.param("CREATE TABLE t(x INT)", True, 0, id="ddl-only-no-select"),
    # PIVOT looks like a false positive and is not. DuckDB rewrites it, and
    # extract_statements reports two statements, [CREATE, SELECT], so a
    # user-written PIVOT really does run a CREATE first. The count check is
    # what catches it. Do not "fix" this row by relaxing that check.
    pytest.param(
        "PIVOT episodes ON status USING count(*)",
        True,
        0,
        id="pivot-expands-to-create-then-select",
    ),
    # Table functions DuckDB labels SELECT. Refused since #453: they parse as
    # SELECT but produce a description of columns, not a set of episodes, so a
    # manifest from one has no episode_id and every consumer refuses it.
    pytest.param("DESCRIBE SELECT episode_id FROM episodes", True, 0, id="describe"),
    pytest.param("SUMMARIZE SELECT episode_id FROM episodes", True, 0, id="summarize"),
    pytest.param("PRAGMA database_list", True, 0, id="pragma"),
    pytest.param("SHOW TABLES", True, 0, id="show"),
    # Accepted. The catalog holds exactly one episode, so a query selecting
    # from it yields one row.
    pytest.param("SELECT episode_id FROM episodes", False, 1, id="legitimate-single-select"),
    pytest.param("TABLE episodes", False, 1, id="table-episodes"),
    pytest.param(
        "WITH picked AS (SELECT episode_id FROM episodes) SELECT * FROM picked",
        False,
        1,
        id="cte",
    ),
    # Legal SELECTs that do not start with the word SELECT. #453 accepts these;
    # its first attempt refused them on a text prefix, which is the regression
    # these three rows exist to catch.
    pytest.param("FROM episodes", False, 1, id="from-first"),
    pytest.param("(SELECT episode_id FROM episodes)", False, 1, id="parenthesized-select"),
    pytest.param("VALUES (1), (2)", False, 2, id="values"),
]


@pytest.mark.parametrize(
    ("sql_template", "expected_refused", "expected_rows"), _SQL_SURFACE_PAYLOADS
)
def test_curate_sql_surface_is_pinned(
    tmp_path: Path,
    sql_template: str,
    expected_refused: bool,
    expected_rows: int,
) -> None:
    """The SQL `curate()` accepts and refuses, stated rather than discovered.

    _stage_manifest_and_count must refuse anything that is not exactly one
    read-only SELECT statement.

    ``connection.sql()`` and ``connection.execute()`` both silently execute
    every semicolon-separated statement in their input (verified on duckdb
    1.5.5: ``connection.sql("SELECT 1; CREATE TABLE pwned(x TEXT)")`` returns
    None and creates the table).  The guard uses
    ``connection.extract_statements`` to parse WITHOUT executing; it requires
    exactly one statement whose type is SELECT.

    The payload table above is the surface. For refused cases, ValueError is
    raised BEFORE any table is created and BEFORE any file is written. For
    accepted cases, the manifest is produced with the expected row count.
    """
    from hflow.curation import _open_connection_over_root, _stage_manifest_and_count

    catalog_dir = tmp_path / "catalog"
    catalog = Catalog(catalog_dir)
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
    )
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    staged_manifest = staging_dir / "manifest.parquet"
    decoy_path = staging_dir / "decoy.parquet"

    sql = sql_template.replace("{decoy}", str(decoy_path))

    connection = _open_connection_over_root(
        catalog_dir, constrained=True, writable_directories=(staging_dir,)
    )
    try:
        if expected_refused:
            # #453 gave the table-function refusals their own wording, so the
            # pattern has to admit both spellings of the same rule.
            with pytest.raises(ValueError, match=r"exactly one (read-only )?SELECT"):
                _stage_manifest_and_count(connection, sql, staged_manifest)
            # Guard fires before any file is written or any table is created.
            assert not staged_manifest.is_file()
            assert not decoy_path.exists()
            with pytest.raises(duckdb.Error):
                connection.execute("SELECT * FROM pwned")
            with pytest.raises(duckdb.Error):
                connection.execute("SELECT * FROM t")
            with pytest.raises(duckdb.Error):
                connection.execute("SELECT * FROM p")
        else:
            # Accepted query: manifest written, with the row count this shape
            # yields over a one-episode catalog.
            row_count = _stage_manifest_and_count(connection, sql, staged_manifest)
            assert row_count == expected_rows
            assert staged_manifest.is_file()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("sql_template", "expected_refused", "expected_rows"), _SQL_SURFACE_PAYLOADS
)
def test_curate_output_none_sql_surface_matches_manifest_gate(
    tmp_path: Path,
    sql_template: str,
    expected_refused: bool,
    expected_rows: int,
) -> None:
    """``curate(..., output=None)`` must refuse the same SQL surface as manifest write.

    Dry-run wraps tenant SQL in ``SELECT count(*) FROM ({sql})``, which can
    turn a multi-statement injection into a parser error — but DESCRIBE,
    SHOW, SUMMARIZE, and PIVOT still execute cleanly inside that wrapper.
    Without ``reject_non_single_select`` on the ``output=None`` branch those
    shapes return a row count instead of ``NonSingleSelectQueryError``.
    """
    catalog_dir = tmp_path / "catalog"
    catalog = Catalog(catalog_dir)
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
    )
    decoy_path = tmp_path / "decoy.parquet"
    sql = sql_template.replace("{decoy}", str(decoy_path))

    if expected_refused:
        with pytest.raises(ValueError, match=r"exactly one (read-only )?SELECT"):
            curate(catalog_dir, sql, output=None, constrained=True)
        assert not decoy_path.exists()
    else:
        report = curate(catalog_dir, sql, output=None, constrained=True)
        assert report.row_count == expected_rows
        assert report.manifest_path is None


def test_cli_curate_dry_run_refuses_non_single_select(
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
            "DESCRIBE SELECT episode_id FROM episodes",
            "--catalog",
            str(recorded_data_root / "catalog"),
            "--dry-run",
        ]
    )
    assert exit_code == 2
    printed = capsys.readouterr()
    assert "exactly one" in printed.err
    assert not (data_root / "manifest.parquet").exists()


_SQL_REFUSED_AS_NOT_ONE_SELECT = [
    # A lone well-formed CREATE TABLE parses cleanly but is not a SELECT:
    # a rule rejection, never a parser error.
    pytest.param("CREATE TABLE t(x INT)", id="lone-create-table"),
    pytest.param("SELECT 1 AS one; CREATE TABLE pwned AS SELECT 1", id="select-then-create"),
    pytest.param("SELECT 1 AS one; DROP TABLE episodes_raw", id="select-then-drop"),
    # DuckDB labels PRAGMA, DESCRIBE, SHOW, SUMMARIZE as StatementType.SELECT
    # because they are table functions, but the curation endpoints advertise
    # exactly one read-only SELECT. Preview interpolated the SQL as
    # ``SELECT * FROM (<sql>)`` where PRAGMA is a syntax error, so the
    # wrapper's parse failure leaked as the caller's error and preview/pin
    # disagreed. The gate now refuses these four by leading keyword (#450).
    pytest.param("PRAGMA database_list", id="pragma-database-list"),
    pytest.param("PRAGMA show_tables", id="pragma-show-tables"),
    pytest.param("PRAGMA version", id="pragma-version"),
    pytest.param("DESCRIBE SELECT 1", id="describe"),
    pytest.param("SHOW TABLES", id="show"),
    pytest.param("SUMMARIZE SELECT 1", id="summarize"),
    # Case-insensitive and comment-prefixed forms are likewise rejected.
    pytest.param("pragma database_list", id="lowercase-pragma"),
    pytest.param("-- comment\nPRAGMA database_list", id="line-comment-then-pragma"),
    pytest.param("/* block */ DESCRIBE SELECT 1", id="block-comment-then-describe"),
]

_SQL_ACCEPTED_AS_ONE_SELECT = [
    pytest.param("SELECT 1 AS one", id="select-literal"),
    pytest.param("SELECT episode_id FROM episodes", id="select-from-catalog"),
    # Leading -- and /* */ comments do not change the statement type.
    pytest.param("-- comment\nSELECT 1", id="line-comment-then-select"),
    pytest.param("/* block comment */ SELECT 1", id="block-comment-then-select"),
    pytest.param("/*c*/--line\n  SELECT 1", id="mixed-comments-then-select"),
    pytest.param("WITH c AS (SELECT 1) SELECT * FROM c", id="cte"),
    pytest.param("with c as (select 1) select * from c", id="lowercase-cte"),
    # A SELECT that reads a pragma as a table function is still SELECT text.
    pytest.param("SELECT * FROM pragma_version()", id="pragma-table-function"),
    # Review of #453: the previous text-prefix heuristic rejected every query
    # that didn't start with SELECT or WITH, but DuckDB labels FROM-first
    # queries, parenthesized selects, and VALUES clauses as
    # StatementType.SELECT and accepts them inside ``FROM (<sql>)``, the
    # shape preview interpolates. The gate must accept them too.
    pytest.param("FROM range(3)", id="from-first"),
    pytest.param("FROM range(3) WHERE range > 0", id="from-first-where"),
    pytest.param("(SELECT 1)", id="parenthesized-select"),
    pytest.param("(SELECT 1 AS one)", id="parenthesized-select-alias"),
    pytest.param("VALUES (1), (2)", id="values"),
    pytest.param("VALUES (1, 'a'), (2, 'b')", id="values-tuples"),
    # FROM-first against a real catalog column. A SELECT previewed from this
    # is the case the reviewer flagged as "the one that matters".
    pytest.param("FROM episodes WHERE status = 'ok'", id="from-first-catalog"),
    # Without the fix, ``SELECT * FROM (SELECT 1 -- trailing comment)`` is a
    # parser error: the trailing line comment swallows the wrapper's closing
    # paren. Appending a newline before wrapping ends the comment first.
    pytest.param("SELECT 1 -- trailing comment without newline", id="trailing-line-comment"),
    # The PRAGMA/DESCRIBE/SHOW/SUMMARIZE refusal reads the first identifier,
    # so a parenthesized DESCRIBE is not headed by the keyword and is
    # accepted. Pinned deliberately: preview and pin both run that form and
    # agree on it, which is the #450 requirement. Read-only introspection of
    # an in-memory catalog was never what the gate was keeping out.
    pytest.param("(DESCRIBE SELECT 1)", id="parenthesized-describe"),
    pytest.param("SELECT * FROM (SHOW TABLES)", id="show-inside-select"),
]


@pytest.mark.parametrize("sql", _SQL_REFUSED_AS_NOT_ONE_SELECT)
def test_reject_non_single_select_refuses_anything_but_one_select(sql: str) -> None:
    with pytest.raises(NonSingleSelectQueryError, match=r"exactly one.*SELECT"):
        reject_non_single_select(sql)


@pytest.mark.parametrize("sql", _SQL_ACCEPTED_AS_ONE_SELECT)
def test_reject_non_single_select_accepts_one_select(sql: str) -> None:
    reject_non_single_select(sql)


def test_reject_non_single_select_distinguishes_parse_failure_from_rule_rejection() -> None:
    # Syntactically invalid SQL surfaces DuckDB's own parser error (which a
    # service renders as the diagnostic message); a well-formed non-SELECT is
    # the rule's ValueError instead, as the refusal table above pins. The
    # server's 400 detail depends on the two staying apart.
    with pytest.raises(duckdb.Error, match="Parser Error"):
        reject_non_single_select("SELEC 1")


def test_constrained_curate_writes_the_manifest_but_refuses_outside_reads(
    tmp_path: Path,
) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog = Catalog(catalog_dir)
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row()],
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
    with open_catalog_connection(tmp_path / "catalog") as connection:
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
    # The measurements table holds one typed column per value.
    assert raw["ratio"] == (np.float32(0.4).item(), None, None)
    assert raw["frames"] == (3.0, None, None)
    assert raw["flag"] == (None, None, True)
    # The wide view exposes them to threshold predicates.
    assert wide == pytest.approx((np.float32(0.4).item(), 3.0))


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="negative-inf"),
        pytest.param(np.float64("nan"), id="numpy-nan"),
    ],
)
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
    with open_catalog_connection(catalog_dir) as connection:
        kept = connection.execute(
            "SELECT episode_id FROM episodes WHERE black_pct < 1.0 AND status != 'quarantined'"
        ).fetchall()
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
    with open_catalog_connection(tmp_path / "catalog") as connection:
        stored = connection.execute(
            "SELECT start_ns, end_ns, label FROM intervals WHERE start_ns = 0 AND end_ns = 5"
        ).fetchall()
    assert stored == [(0, 5, "segment")]


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


# Every interval refusal names the check, and the label where one exists,
# because a check emitting many intervals gives the reader no other way to
# tell which one is wrong (#161). The casts are the misuse each case refuses.
_REFUSED_INTERVALS = [
    # A float bound is not a nanosecond timestamp, so it raises rather than coercing.
    pytest.param(
        hflow.Interval(start_ns=cast(int, np.float64(5.0)), end_ns=10),
        r"segment_check.*start_ns.*float",
        id="float-bound",
    ),
    # ``bool`` subclasses ``int``, so ``True`` passes an isinstance test and
    # would store as 1 ns. It is a mistake, not a timestamp.
    pytest.param(
        hflow.Interval(start_ns=cast(int, True), end_ns=10),
        r"segment_check.*start_ns.*bool",
        id="bool-bound",
    ),
    # An end before its start is a negative duration to everything downstream.
    pytest.param(
        hflow.Interval(start_ns=10, end_ns=5, label="peak_velocity"),
        r"segment_check.*'peak_velocity'.*end must be >= start",
        id="inverted",
    ),
    # Log time is nanoseconds since the epoch, so a negative bound is a bug.
    pytest.param(
        hflow.Interval(start_ns=-5, end_ns=-1, label="gap"),
        r"segment_check.*'gap'.*non-negative",
        id="negative-bound",
    ),
    # A label that is not a string reaches the run fingerprint and breaks its
    # sort. The failure it replaces was data dependent: the fingerprint sorts
    # ``(start_ns, end_ns, label)`` tuples, so a bad label is only ever
    # compared against another when two intervals share both bounds. One
    # interval stored fine, two colliding ones raised a ``TypeError`` several
    # frames inside ``append_episode`` naming neither the check nor the field
    # (#392).
    pytest.param(
        hflow.Interval(start_ns=0, end_ns=10, label=cast(str, None)),
        r"segment_check.*NoneType.*labels are strings",
        id="non-string-label",
    ),
    # An unserializable label used to surface as a json.dumps TypeError. That
    # one fired whatever the bounds were, because the fingerprint payload is
    # serialized after the sort. It is now refused where the bounds are, so
    # the message names the check.
    pytest.param(
        hflow.Interval(start_ns=0, end_ns=10, label=cast(str, object())),
        r"segment_check.*object.*labels are strings",
        id="non-serializable-label",
    ),
    # Labels are stored verbatim, so " freeze " and "freeze" would be two
    # names for one thing. The same rule an observation id already carries.
    pytest.param(
        hflow.Interval(start_ns=0, end_ns=10, label=" freeze "),
        r"segment_check.*' freeze '.*stored verbatim",
        id="padded-label",
    ),
]


@pytest.mark.parametrize(("interval", "expected_message"), _REFUSED_INTERVALS)
def test_an_unstoreable_interval_is_refused_before_any_write(
    tmp_path: Path, interval: hflow.Interval, expected_message: str
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _appended_with_interval(tmp_path, interval)
    assert list((tmp_path / "catalog" / "episodes").glob("*.parquet")) == []


def test_colliding_intervals_with_a_bad_label_are_refused_not_crashed(tmp_path: Path) -> None:
    """The case that used to raise TypeError from inside the fingerprint sort.

    Two intervals sharing both bounds are the only shape that reaches the label
    comparison, so this is the arrangement the old code died on rather than
    stored.
    """
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name="segment_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        intervals=[
            hflow.Interval(start_ns=0, end_ns=10, label="freeze"),
            # cast: the misuse this test exists to refuse.
            hflow.Interval(start_ns=0, end_ns=10, label=cast(str, None)),
        ],
    )
    with pytest.raises(ValueError, match=r"segment_check.*NoneType.*labels are strings"):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )


def test_an_empty_interval_label_is_allowed(tmp_path: Path) -> None:
    """``label: str = ""`` is the dataclass default, so empty is a real value
    rather than an omission. Pinned so the padding rule above cannot grow into
    refusing it."""
    _appended_with_interval(tmp_path, hflow.Interval(start_ns=0, end_ns=10))

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute("SELECT label FROM intervals").fetchall() == [("",)]


def test_a_zero_length_interval_is_allowed(tmp_path: Path) -> None:
    """An instant is a real thing to record, so only end < start is refused.

    Pinned rather than left implicit: the ordering rule is one ``<`` away from
    rejecting every event with no duration.
    """
    _appended_with_interval(tmp_path, hflow.Interval(start_ns=5, end_ns=5, label="touchdown"))

    with open_catalog_connection(tmp_path / "catalog") as connection:
        stored = connection.execute(
            "SELECT start_ns, end_ns, label FROM intervals WHERE label = 'touchdown'"
        ).fetchall()
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


def test_catalog_timestamp_bigint_boundaries_round_trip(tmp_path: Path) -> None:
    """The complete non-negative BIGINT domain survives catalog storage."""
    maximum = 2**63 - 1
    row = CheckRunRow(
        check_name="range_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        observations=[
            hflow.Observation("zero", 0, {"score": 0}),
            hflow.Observation("maximum", cast(int, np.int64(maximum)), {"score": 1}),
        ],
        intervals=[
            hflow.Interval(start_ns=0, end_ns=0, label="zero"),
            hflow.Interval(start_ns=cast(int, np.int64(maximum)), end_ns=maximum, label="maximum"),
        ],
    )
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[row],
    )

    with open_catalog_connection(tmp_path / "catalog") as connection:
        assert connection.execute(
            "SELECT observation_id, timestamp_ns FROM observations_latest ORDER BY timestamp_ns"
        ).fetchall() == [("zero", 0), ("maximum", maximum)]
        assert connection.execute(
            "SELECT label, start_ns, end_ns FROM intervals ORDER BY start_ns"
        ).fetchall() == [("zero", 0, 0), ("maximum", maximum, maximum)]


@pytest.mark.parametrize("timestamp_ns", [2**63, np.uint64(2**63)])
def test_out_of_range_observation_timestamp_is_refused_before_catalog_writes(
    tmp_path: Path, timestamp_ns: object
) -> None:
    row = CheckRunRow(
        check_name="range_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        observations=[hflow.Observation("frame:1", cast(int, timestamp_ns), {"score": 1})],
    )

    with pytest.raises(
        ValueError,
        match=r"range_check.*'frame:1'.*timestamp_ns=9223372036854775808.*BIGINT",
    ):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=write_fake_canonical(tmp_path),
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )

    assert not list((tmp_path / "catalog").rglob("*.parquet"))


@pytest.mark.parametrize("bound_name", ["start_ns", "end_ns"])
@pytest.mark.parametrize("bound", [2**63, np.uint64(2**63)])
def test_out_of_range_interval_bound_is_refused_before_catalog_writes(
    tmp_path: Path, bound_name: str, bound: object
) -> None:
    bounds = {"start_ns": 0, "end_ns": 10}
    bounds[bound_name] = cast(int, bound)
    row = CheckRunRow(
        check_name="range_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        intervals=[hflow.Interval(**bounds, label="segment")],
    )

    with pytest.raises(
        ValueError,
        match=rf"range_check.*'segment'.*{bound_name}=9223372036854775808.*BIGINT",
    ):
        Catalog(tmp_path / "catalog").append_episode(
            canonical_path=write_fake_canonical(tmp_path),
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )

    assert not list((tmp_path / "catalog").rglob("*.parquet"))


@pytest.mark.parametrize(
    ("check_name", "measurement_key", "measurement_value", "expected_message"),
    [
        # An unstoreable measurement raises loudly instead of writing NULLs.
        pytest.param(
            "broken_check", "bad", {"nested": 1}, r"broken_check.*'bad'.*dict", id="non-scalar"
        ),
        pytest.param(
            "claims_task", "task", 99.0, r"'claims_task'.*'task'", id="claims-an-episodes-column"
        ),
        # DuckDB identifiers are case-insensitive, so 'Task' shadows 'task' too.
        pytest.param(
            "claims_task", "Task", 1.0, r"'Task'.*shadows 'task'", id="shadows-case-insensitively"
        ),
        pytest.param("empty_key_check", "", 99.0, r"'empty_key_check'.*''", id="empty"),
        pytest.param("blank_key_check", "   ", 99.0, r"'blank_key_check'", id="whitespace-only"),
    ],
)
def test_a_measurement_that_cannot_become_a_column_is_refused(
    check_name: str,
    measurement_key: str,
    measurement_value: object,
    expected_message: str,
    tmp_path: Path,
) -> None:
    """Every measurement has to survive the pivot into a wide-view column (#160).

    A key named like an episodes column pivots into ``<key>_1`` beside it. An
    empty or whitespace-only key pivots into a column named for the SQL
    expression that produced it, which is a queryable surface with no name a
    person would write and no rename path (docs/CATALOG.md, "Naming
    measurement keys").
    """
    canonical = tmp_path / "e.canonical.mcap"
    canonical.write_bytes(b"episode-bytes")
    row = CheckRunRow(
        check_name=check_name,
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.1,
        # cast: the non-scalar case is the misuse this test exists to refuse.
        measurements=cast(dict, {measurement_key: measurement_value}),
    )
    with pytest.raises(ValueError, match=expected_message):
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
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[],
    )

    with open_catalog_connection(tmp_path / "catalog") as connection:
        view_columns = [
            column[0] for column in connection.execute("SELECT * FROM episodes").description
        ]

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


def test_append_refuses_case_colliding_keys_before_writing(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    row = replace(example_check_row(), measurements={"/Camera/score": 0.1, "/camera/score": 0.9})
    with pytest.raises(ValueError, match=r"'/Camera/score'.*'/camera/score'.*collide"):
        catalog.append_episode(
            canonical_path=write_fake_canonical(tmp_path),
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )
    assert list(catalog.root.rglob("*.parquet")) == []


@pytest.mark.parametrize("same_episode", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_case_collisions_across_appends_refuse_curation_without_replacing_output(
    tmp_path: Path, same_episode: bool, constrained: bool
) -> None:
    catalog = Catalog(tmp_path / "catalog")
    manifest = tmp_path / "manifest.parquet"
    for index, key in enumerate(("/Camera/score", "/camera/score")):
        catalog.append_episode(
            canonical_path=write_fake_canonical(
                tmp_path, b"same episode" if same_episode else f"episode {index}".encode()
            ),
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[replace(example_check_row(), measurements={key: 0.1 + index * 0.8})],
        )
        if index == 0:
            curate(catalog.root, "SELECT episode_id FROM episodes", output=manifest)
    previous_manifest = manifest.read_bytes()
    previous_catalog = {path: path.read_bytes() for path in catalog.root.rglob("*.parquet")}

    with pytest.raises(ValueError, match=r"'/Camera/score'.*'/camera/score'.*collide") as failure:
        open_catalog_connection(catalog.root, constrained=constrained)
    assert "Rename" in str(failure.value)
    assert "measurements/*.parquet" in str(failure.value)
    with pytest.raises(ValueError, match=r"measurement keys.*collide"):
        curate(
            catalog.root,
            'SELECT episode_id FROM episodes WHERE "/camera/score" < 0.5',
            output=manifest,
            constrained=constrained,
        )
    assert manifest.read_bytes() == previous_manifest
    assert not list(tmp_path.glob(".hflow-manifest-*"))
    assert {path: path.read_bytes() for path in catalog.root.rglob("*.parquet")} == previous_catalog
    with duckdb.connect() as connection:
        assert connection.execute(
            "SELECT key, value_double FROM read_parquet(?) ORDER BY key",
            [str(catalog.root / "measurements" / "*.parquet")],
        ).fetchall() == [("/Camera/score", 0.1), ("/camera/score", 0.9)]


def test_existing_catalog_with_case_colliding_keys_is_refused(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[
            replace(example_check_row(), measurements={"/Camera/score": 0.1, "other": 0.9})
        ],
    )
    # Model evidence written before validation existed, without bypassing the
    # reader under test or relying on today's append accepting invalid input.
    (measurements,) = (catalog.root / "measurements").glob("*.parquet")
    legacy = tmp_path / "legacy.parquet"
    with duckdb.connect() as connection:
        connection.execute(
            "COPY (SELECT * REPLACE (CASE WHEN key = 'other' THEN '/camera/score' "
            "ELSE key END AS key) FROM read_parquet($source)) TO $destination (FORMAT PARQUET)",
            {"source": str(measurements), "destination": str(legacy)},
        )
    legacy.replace(measurements)
    previous = measurements.read_bytes()
    with pytest.raises(ValueError, match=r"'/Camera/score'.*'/camera/score'.*collide"):
        open_catalog_connection(catalog.root)
    assert measurements.read_bytes() == previous


def test_distinct_unicode_keys_and_exact_key_reuse_remain_queryable(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    # lower() would merge Ä/ä and Kelvin sign/k; casefold() also merges ß/ss.
    keys = [
        "/Ä/score",
        "/ä/score",
        "/\N{KELVIN SIGN}/score",
        "/k/score",
        "/ß/score",
        "/ss/score",
        "tas\N{KELVIN SIGN}",
    ]
    for index in range(2):
        catalog.append_episode(
            canonical_path=write_fake_canonical(tmp_path, f"episode {index}".encode()),
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[
                replace(
                    example_check_row(),
                    measurements={key: index + n / 10 for n, key in enumerate(keys)},
                )
            ],
        )
    columns = ", ".join(f'"{key}"' for key in keys)
    expected = [tuple(index + n / 10 for n in range(len(keys))) for index in range(2)]
    with open_catalog_connection(catalog.root) as connection:
        assert (
            connection.execute(f'SELECT {columns} FROM episodes ORDER BY "{keys[0]}"').fetchall()
            == expected
        )
        assert connection.execute(
            "SELECT DISTINCT key FROM measurements ORDER BY key"
        ).fetchall() == [(key,) for key in sorted(keys)]
    snapshot = tmp_path / "snapshot"
    hflow.export_dataset_snapshot(catalog.root, snapshot)
    with duckdb.connect() as connection:
        assert (
            connection.execute(
                f'SELECT {columns} FROM read_parquet(?) ORDER BY "{keys[0]}"',
                [str(snapshot / "samples.parquet")],
            ).fetchall()
            == expected
        )


@pytest.mark.parametrize("recurring", [False, True])
def test_crash_repaired_append_keeps_one_recorded_at_across_tables(
    tmp_path: Path, recurring: bool
) -> None:
    """A retry after a crashed append must not mix timestamps across tables.

    Mixed recorded_at would let the per-key 'latest' views attribute another
    run's rows to this one.
    """
    import time

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

    if recurring:
        append_same_outcome()
        catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[replace(row, measurements={"score": 2.0})],
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

    timestamps = recorded_at_values(tmp_path / "catalog", stem)
    assert len(timestamps) == 1, f"mixed recorded_at across tables: {timestamps}"


@pytest.mark.parametrize("recurring", [False, True])
def test_replaying_an_append_heals_dependents_left_stale_by_a_crashed_repair(
    tmp_path: Path,
    recurring: bool,
) -> None:
    """#51's residual window: a winner that created the episodes file but
    crashed before force-aligning the dependents leaves them carrying a stale
    recorded_at. A later replay of the same outcome (the normal retry lane)
    must reconcile every dependent to the episodes file's recorded_at instead
    of early-returning past the damage forever.
    """
    import duckdb

    canonical = write_fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")
    row = example_check_row()

    def append_same_outcome() -> "hflow.catalog.AppendResult":
        return catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[row],
        )

    if recurring:
        append_same_outcome()
        catalog.append_episode(
            canonical_path=canonical,
            stamps=FAKE_STAMPS,
            episode_metadata={},
            check_rows=[example_check_row(value=2.0)],
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

    timestamps = recorded_at_values(tmp_path / "catalog", stem)
    assert len(timestamps) == 1, f"replay left mixed recorded_at across tables: {timestamps}"


def test_replaying_an_append_refuses_a_corrupt_empty_commit_marker(tmp_path: Path) -> None:
    """append_episode always inserts exactly one episodes row, so a zero-row
    episodes file is corruption -- a replay must refuse loudly instead of
    silently skipping reconciliation against it."""
    import duckdb

    canonical = write_fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")
    row = example_check_row()

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


@pytest.mark.parametrize("recurring", [False, True])
def test_concurrent_append_of_the_identical_outcome_keeps_one_recorded_at(
    tmp_path: Path,
    recurring: bool,
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

    canonical = write_fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")
    row = example_check_row()

    if recurring:
        for prior_row in [row, example_check_row(value=2.0)]:
            catalog.append_episode(
                canonical_path=canonical,
                stamps=FAKE_STAMPS,
                episode_metadata={},
                check_rows=[prior_row],
            )

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

    timestamps = recorded_at_values(tmp_path / "catalog", stem)
    assert len(timestamps) == 1, f"mixed recorded_at across tables: {timestamps}"


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

    canonical = write_fake_canonical(tmp_path)
    catalog = Catalog(tmp_path / "catalog")

    older = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row(version="v1", value=1.0)],
    )
    time.sleep(0.01)
    newer = catalog.append_episode(
        canonical_path=canonical,
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[example_check_row(version="v2", value=2.0)],
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

    with open_catalog_connection(tmp_path / "catalog") as connection:
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
        canonical_path=write_fake_canonical(tmp_path),
        stamps=FAKE_STAMPS,
        episode_metadata={},
        check_rows=[check_row],
        quarantine_tags=["quarantined:blur"] if quarantined else [],
    )
    return catalog_root


def _status_of_only_episode(catalog_root: Path, *, constrained: bool = False) -> str:
    with open_catalog_connection(catalog_root, constrained=constrained) as connection:
        row = connection.execute("SELECT status FROM episodes").fetchone()
    assert row is not None
    return str(row[0])


@pytest.mark.parametrize(
    ("critical", "status", "expected_episode_status"),
    [
        # #164 item 1: a crashed critical check no longer reads as a pass.
        pytest.param(True, hflow.CheckStatus.ERROR, "unverified", id="critical-error"),
        pytest.param(True, hflow.CheckStatus.PASSED, "ok", id="critical-pass"),
        # #164 item 3: only a CRITICAL crash leaves an episode unverified.
        pytest.param(False, hflow.CheckStatus.ERROR, "ok", id="non-critical-error"),
        # Skipped and superseded are not crashes, so neither reads unverified.
        pytest.param(True, hflow.CheckStatus.SKIPPED, "ok", id="critical-skipped"),
        pytest.param(True, hflow.CheckStatus.SUPERSEDED, "ok", id="critical-superseded"),
    ],
)
def test_episode_status_reflects_its_single_check(
    tmp_path: Path, critical: bool, status: hflow.CheckStatus, expected_episode_status: str
) -> None:
    catalog_root = _episode_with_check(tmp_path, critical=critical, status=status)
    assert _status_of_only_episode(catalog_root) == expected_episode_status


def test_quarantine_outranks_a_critical_error(tmp_path: Path) -> None:
    """#164 item 2: quarantined wins when an episode is both."""
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    catalog.append_episode(
        canonical_path=write_fake_canonical(tmp_path),
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


def test_successful_rerun_clears_unverified(tmp_path: Path) -> None:
    """#164 item 4: a later good run of the same check reports ok again."""
    catalog_root = tmp_path / "catalog"
    catalog = Catalog(catalog_root)
    canonical = write_fake_canonical(tmp_path)
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
        canonical_path=write_fake_canonical(wide_base),
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

    with open_catalog_connection(catalog_root) as connection:
        statuses = dict(connection.execute("SELECT episode_id, status FROM episodes").fetchall())

    assert statuses[canonicals["crashed"]] == "unverified"
    assert statuses[canonicals["healthy"]] == "ok"


def test_status_builder_refuses_an_unqualified_column() -> None:
    """An unqualified column would correlate the subquery against itself."""
    with pytest.raises(ValueError, match="qualified"):
        episode_status_case_sql(
            quarantined_column="quarantined", check_runs_relation="check_runs_latest"
        )
