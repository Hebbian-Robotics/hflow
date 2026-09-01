"""Run profiles: the ingest stage graph's toggleable stages driven through App.process."""

from pathlib import Path

import duckdb
import pytest

import hflow
from hflow.curation import open_catalog_connection
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

FAST_SPEC = SyntheticEpisodeSpec(duration_s=2.0, cameras=())
CAMERA_SPEC = SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",))


@pytest.fixture()
def source_episode(tmp_path: Path) -> Path:
    return synthesize_episode(tmp_path / "episode.mcap", FAST_SPEC)


def _app_with_check_and_enrichment(data_root: Path) -> hflow.App:
    app = hflow.App("profiles", data_root=data_root, default_checks=())

    @app.check(version="1")
    def joints(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"joint_count": 7})

    @app.enrich(version="1")
    def caption(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(labels={"caption": "a robot arm moves"})

    return app


def _count(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = connection.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def test_full_then_relabel_appends_labels_without_rewriting_canonical(
    source_episode: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    app = _app_with_check_and_enrichment(data_root)

    full_report = app.process(source_episode, stages="full")
    assert full_report.stages_run == frozenset(hflow.Stage)
    canonical_mtime_ns = full_report.canonical_path.stat().st_mtime_ns

    connection = open_catalog_connection(data_root / "catalog")
    try:
        measurements_after_full = _count(connection, "SELECT count(*) FROM measurements")
        check_rows_after_full = _count(
            connection, "SELECT count(*) FROM check_runs WHERE check_name = 'joints'"
        )
    finally:
        connection.close()

    relabel_report = app.process(source_episode, stages="relabel")
    assert relabel_report.stages_run == frozenset({hflow.Stage.LABELS})
    # The canonical file is untouched: relabel never re-runs the transform.
    assert full_report.canonical_path.stat().st_mtime_ns == canonical_mtime_ns
    # Stamps come back from the file's own provenance record.
    assert relabel_report.stamps == full_report.stamps
    assert not relabel_report.checks
    assert [run.enrichment.name for run in relabel_report.enrichments] == ["caption"]
    assert relabel_report.enrichments[0].status == hflow.CheckStatus.MEASURED

    connection = open_catalog_connection(data_root / "catalog")
    try:
        # New enrichment measurement rows landed; check rows did not grow.
        assert _count(connection, "SELECT count(*) FROM measurements") > measurements_after_full
        assert (
            _count(connection, "SELECT count(*) FROM check_runs WHERE check_name = 'joints'")
            == check_rows_after_full
        )
    finally:
        connection.close()


def test_metadata_backfill_runs_checks_only(source_episode: Path, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app = _app_with_check_and_enrichment(data_root)
    app.process(source_episode, stages="full")

    backfill_report = app.process(source_episode, stages="metadata_backfill")
    assert backfill_report.stages_run == frozenset({hflow.Stage.META})
    assert [run.check.name for run in backfill_report.checks] == ["joints"]
    assert not backfill_report.enrichments
    assert backfill_report.catalog_entry is not None

    connection = open_catalog_connection(data_root / "catalog")
    try:
        backfill_row_names = connection.execute(
            "SELECT DISTINCT check_name FROM check_runs WHERE run_fingerprint = ?",
            [backfill_report.catalog_entry.run_fingerprint],
        ).fetchall()
    finally:
        connection.close()
    assert backfill_row_names == [("joints",)]


def test_step_selection_records_only_the_named_step(source_episode: Path, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app = hflow.App("selected-steps", data_root=data_root, default_checks=())

    @app.check(version="1")
    def selected_check(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"selected": 1.0})

    @app.check(version="1")
    def unselected_check(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"unselected": 1.0})

    @app.enrich(version="1")
    def unselected_enrichment(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(labels={"caption": "not requested"})

    report = app.process(
        source_episode,
        stages={hflow.Stage.SYNC, hflow.Stage.META, hflow.Stage.LABELS},
        step_names={"selected_check"},
    )

    assert [run.check.name for run in report.checks] == ["selected_check"]
    assert not report.enrichments
    assert report.catalog_entry is not None
    connection = open_catalog_connection(data_root / "catalog")
    try:
        recorded_step_names = connection.execute(
            "SELECT check_name FROM check_runs WHERE run_fingerprint = ?",
            [report.catalog_entry.run_fingerprint],
        ).fetchall()
    finally:
        connection.close()
    assert recorded_step_names == [("selected_check",)]


def test_step_selection_runs_only_the_named_enrichment(
    source_episode: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    app = hflow.App("selected-enrichment", data_root=data_root, default_checks=())

    @app.enrich(version="1")
    def selected_caption(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(labels={"caption": "selected"})

    @app.enrich(version="1")
    def unselected_embedding(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(labels={"embedding": "not requested"})

    report = app.process(
        source_episode,
        stages={hflow.Stage.SYNC, hflow.Stage.LABELS},
        step_names={"selected_caption"},
    )

    assert [run.enrichment.name for run in report.enrichments] == ["selected_caption"]
    assert report.enrichments[0].result is not None
    assert report.enrichments[0].result.labels == {"caption": "selected"}
    assert report.catalog_entry is not None
    connection = open_catalog_connection(data_root / "catalog")
    try:
        recorded_step_names = connection.execute(
            "SELECT check_name FROM check_runs WHERE run_fingerprint = ?",
            [report.catalog_entry.run_fingerprint],
        ).fetchall()
    finally:
        connection.close()
    assert recorded_step_names == [("selected_caption",)]


def test_step_selection_validates_names_and_enabled_stages_before_episode_io(
    tmp_path: Path,
) -> None:
    missing_episode = tmp_path / "missing.mcap"
    app = _app_with_check_and_enrichment(tmp_path / "data")

    with pytest.raises(ValueError, match=r"unknown step names.*not_registered"):
        app.process(missing_episode, step_names={"not_registered"})

    with pytest.raises(ValueError, match=r"caption \(labels\).+enabled stages: \['meta'\]"):
        app.process(
            missing_episode,
            stages={hflow.Stage.META},
            step_names={"caption"},
        )


def test_unselected_endpoint_does_not_need_configuration(
    source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App("selected-endpoint", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1")
    def local_check(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"local": 1.0})

    @app.enrich(version="1", uses="remote-model")
    def remote_enrichment(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(labels={"remote": "unused"})

    report = app.process(
        source_episode,
        stages={hflow.Stage.SYNC, hflow.Stage.META},
        step_names={"local_check"},
        record=False,
    )

    assert [run.check.name for run in report.checks] == ["local_check"]


def test_partial_metadata_run_preserves_unselected_quarantine_and_rechecked_gate_replaces_it(
    source_episode: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    original_application = hflow.App("selected-quarantine", data_root=data_root, default_checks=())

    @original_application.check(version="1", name="safety_gate", critical=True)
    def rejecting_safety_gate(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(verdict=False)

    @original_application.check(version="1")
    def refreshed_evidence(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"refreshed": 1.0})

    assert original_application.process(source_episode).quarantined

    partial_report = original_application.process(
        source_episode,
        stages={hflow.Stage.META},
        step_names={"refreshed_evidence"},
    )
    assert partial_report.quarantined
    assert partial_report.quarantine_tags == ["quarantined:safety_gate"]

    revised_application = hflow.App("selected-quarantine", data_root=data_root, default_checks=())

    @revised_application.check(version="2", name="safety_gate", critical=True)
    def accepting_safety_gate(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(verdict=True)

    revised_report = revised_application.process(
        source_episode,
        stages={hflow.Stage.META},
        step_names={"safety_gate"},
    )
    assert not revised_report.quarantined
    assert revised_report.quarantine_tags == []


def test_relabel_without_canonical_errors_helpfully(source_episode: Path, tmp_path: Path) -> None:
    app = _app_with_check_and_enrichment(tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="run the full or sync profile first"):
        app.process(source_episode, stages="relabel")


def test_quarantine_is_honored_across_invocations_via_the_catalog(
    source_episode: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    app = hflow.App("profiles-quarantine", data_root=data_root, default_checks=())
    enrichment_ran = False

    @app.check(version="1", critical=True)
    def dead_camera(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(verdict=False)

    @app.enrich(version="1")
    def expensive_labeling(ep: hflow.Episode) -> hflow.EnrichmentResult:
        nonlocal enrichment_ran
        enrichment_ran = True
        return hflow.EnrichmentResult()

    full_report = app.process(source_episode, stages="full")
    assert full_report.quarantined

    relabel_report = app.process(source_episode, stages="relabel")
    assert not enrichment_ran
    assert relabel_report.enrichments[0].status == hflow.CheckStatus.SKIPPED
    not_run = relabel_report.enrichments[0].not_run
    assert isinstance(not_run, hflow.SkippedByQuarantine)
    assert "quarantined:dead_camera" in not_run.reason


def test_no_catalog_means_no_known_quarantine(source_episode: Path, tmp_path: Path) -> None:
    app = _app_with_check_and_enrichment(tmp_path / "data")
    # sync writes the canonical without recording anything to consult ...
    app.process(source_episode, stages={hflow.Stage.SYNC}, record=False)
    # ... so a relabel run proceeds: no catalog = no known quarantine.
    relabel_report = app.process(source_episode, stages="relabel", record=False)
    assert relabel_report.enrichments[0].status == hflow.CheckStatus.MEASURED


def test_media_stage_records_a_contact_sheet_artifact(tmp_path: Path) -> None:
    source_episode = synthesize_episode(tmp_path / "camera_episode.mcap", CAMERA_SPEC)
    data_root = tmp_path / "data"
    app = hflow.App("profiles-media", data_root=data_root, default_checks=())

    report = app.process(source_episode, stages={hflow.Stage.SYNC, hflow.Stage.MEDIA})
    media_runs = [run for run in report.enrichments if run.enrichment.name == "media/contact_sheet"]
    assert len(media_runs) == 1
    assert media_runs[0].status == hflow.CheckStatus.MEASURED
    assert media_runs[0].result is not None
    sheet_path = media_runs[0].result.artifacts["/wrist_cam/compressed"]
    assert sheet_path.is_file()
    assert sheet_path.parent.name == "media"
    assert sheet_path.name == "wrist_cam_compressed.jpg"

    connection = open_catalog_connection(data_root / "catalog")
    try:
        artifact_row = connection.execute(
            "SELECT value_text FROM measurements "
            "WHERE check_name = 'media/contact_sheet' AND key = 'artifact//wrist_cam/compressed'"
        ).fetchone()
    finally:
        connection.close()
    assert artifact_row is not None
    assert str(artifact_row[0]).endswith("wrist_cam_compressed.jpg")


def test_media_step_selection_distinguishes_unselected_from_requested(tmp_path: Path) -> None:
    source_episode = synthesize_episode(tmp_path / "camera_episode.mcap", CAMERA_SPEC)
    data_root = tmp_path / "data"
    app = hflow.App("selected-media", data_root=data_root, default_checks=())

    unselected_report = app.process(
        source_episode,
        stages={hflow.Stage.SYNC, hflow.Stage.MEDIA},
        step_names=set(),
    )
    assert not unselected_report.enrichments

    selected_report = app.process(
        source_episode,
        stages={hflow.Stage.MEDIA},
        step_names={"media/contact_sheet"},
    )
    assert [run.enrichment.name for run in selected_report.enrichments] == ["media/contact_sheet"]
    assert selected_report.enrichments[0].status == hflow.CheckStatus.MEASURED


def test_media_stage_is_silently_absent_without_cameras(
    source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App("profiles-no-media", data_root=tmp_path / "data")
    report = app.process(source_episode, stages={hflow.Stage.SYNC, hflow.Stage.MEDIA}, record=False)
    assert not report.enrichments  # no cameras: no media row claims otherwise


def test_stages_accepts_profile_string_and_explicit_set(
    source_episode: Path, tmp_path: Path
) -> None:
    app = _app_with_check_and_enrichment(tmp_path / "data")
    app.process(source_episode, stages={hflow.Stage.SYNC}, record=False)

    from_profile = app.process(source_episode, stages="metadata_backfill", record=False)
    from_set = app.process(source_episode, stages={hflow.Stage.META}, record=False)
    from_list = app.process(source_episode, stages=[hflow.Stage.META], record=False)
    assert (
        from_profile.stages_run
        == from_set.stages_run
        == from_list.stages_run
        == frozenset({hflow.Stage.META})
    )
    assert "stages: metadata_backfill (meta)" in from_profile.summary()


def test_unknown_profile_errors_with_valid_names(source_episode: Path, tmp_path: Path) -> None:
    app = _app_with_check_and_enrichment(tmp_path / "data")
    with pytest.raises(ValueError, match="metadata_backfill"):
        app.process(source_episode, stages="everything")


def test_run_profiles_vocabulary() -> None:
    assert hflow.stages_for_profile("full") == frozenset(hflow.Stage)
    assert hflow.RUN_PROFILES["relabel"] == frozenset({hflow.Stage.LABELS})
    assert hflow.RUN_PROFILES["metadata_backfill"] == frozenset({hflow.Stage.META})
