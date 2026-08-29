"""Enrichment step registration, execution, and catalog outcomes."""

import json
from pathlib import Path
from typing import cast

import pytest

import hflow
from hflow.app import MEDIA_CONTACT_SHEET_STEP_NAME
from hflow.curation import open_catalog_connection
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode

FAST_SPEC = SyntheticEpisodeSpec(duration_s=2.0, cameras=())


@pytest.fixture()
def source_episode(tmp_path: Path) -> Path:
    return synthesize_episode(tmp_path / "episode.mcap", FAST_SPEC)


def test_enrichments_run_after_all_checks(source_episode: Path, tmp_path: Path) -> None:
    app = hflow.App("enrich-order", data_root=tmp_path / "data")
    execution_order: list[str] = []

    @app.enrich(version="1")
    def caption(ep: hflow.Episode) -> hflow.EnrichmentResult:
        execution_order.append("caption")
        return hflow.EnrichmentResult(labels={"caption": "a robot arm moves"})

    @app.check(version="1")
    def joints(ep: hflow.Episode) -> hflow.CheckResult:
        execution_order.append("joints")
        return hflow.CheckResult()

    report = app.test(source_episode, verbose=False)
    assert execution_order == ["joints", "caption"]
    assert [run.status for run in report.enrichments] == [hflow.CheckStatus.MEASURED]
    assert "caption = a robot arm moves" in report.summary()


def test_quarantine_skips_enrichments(source_episode: Path, tmp_path: Path) -> None:
    app = hflow.App("enrich-gate", data_root=tmp_path / "data")
    enrichment_ran = False

    @app.check(version="1", critical=True)
    def always_fails(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(verdict=False)

    @app.enrich(version="1")
    def expensive_labeling(ep: hflow.Episode) -> hflow.EnrichmentResult:
        nonlocal enrichment_ran
        enrichment_ran = True
        return hflow.EnrichmentResult()

    report = app.test(source_episode, verbose=False)
    assert report.quarantined
    assert not enrichment_ran
    assert report.enrichments[0].status == hflow.CheckStatus.SKIPPED


def test_enrichment_wrong_return_type_is_an_error(source_episode: Path, tmp_path: Path) -> None:
    app = hflow.App("enrich-boundary", data_root=tmp_path / "data")

    @app.enrich(version="1")
    def returns_a_string(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return cast(hflow.EnrichmentResult, "a caption")  # deliberate misuse

    report = app.test(source_episode, verbose=False)
    assert report.enrichments[0].status == hflow.CheckStatus.ERROR
    assert report.enrichments[0].error is not None
    assert "expected hflow.EnrichmentResult" in report.enrichments[0].error


def test_enrichment_labels_and_artifacts_land_in_the_catalog(
    source_episode: Path, tmp_path: Path
) -> None:
    import numpy as np

    data_root = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    app = hflow.App("enrich-catalog", data_root=data_root)

    @app.enrich(version="1")
    def labeling(ep: hflow.Episode) -> hflow.EnrichmentResult:
        artifact_path = artifact_dir / "segments.json"
        artifact_path.write_text(json.dumps({"segments": []}))
        return hflow.EnrichmentResult(
            labels=cast(
                dict,
                {
                    "caption": "synthetic joints wiggle",
                    # real labelers return NumPy scalars; they must store, not NULL out
                    "confidence": np.float32(0.9),
                },
            ),
            artifacts={"segments": artifact_path},
            tags=["labeled"],
        )

    app.test(source_episode, verbose=False, record=True)
    connection = open_catalog_connection(data_root / "catalog")
    try:
        caption_row = connection.execute(
            "SELECT value_text FROM measurements WHERE key = 'caption'"
        ).fetchone()
        assert caption_row == ("synthetic joints wiggle",)
        confidence_row = connection.execute(
            "SELECT value_double FROM measurements WHERE key = 'confidence'"
        ).fetchone()
        assert confidence_row == pytest.approx((np.float32(0.9).item(),))
        artifact_row = connection.execute(
            "SELECT value_text FROM measurements WHERE key = 'artifact/segments'"
        ).fetchone()
        assert artifact_row is not None
        assert str(artifact_row[0]).endswith("segments.json")
        run_row = connection.execute(
            "SELECT status FROM check_runs WHERE check_name = 'labeling'"
        ).fetchone()
        assert run_row == ("measured",)
        tag_row = connection.execute(
            "SELECT tag FROM tags WHERE check_name = 'labeling'"
        ).fetchone()
        assert tag_row == ("labeled",)
    finally:
        connection.close()


def test_step_names_are_unique_across_checks_and_enrichments(tmp_path: Path) -> None:
    app = hflow.App("enrich-names", data_root=tmp_path)

    @app.check(version="1")
    def labeling(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    with pytest.raises(ValueError, match="already registered"):

        @app.enrich(version="1", name="labeling")
        def another(ep: hflow.Episode) -> hflow.EnrichmentResult:
            return hflow.EnrichmentResult()


def test_built_in_media_step_name_is_reserved_for_user_steps(tmp_path: Path) -> None:
    app = hflow.App("reserved-media-name", data_root=tmp_path)

    def check(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    def enrichment(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult()

    with pytest.raises(ValueError, match=r"media/contact_sheet.*already registered"):
        app.check(version="1", name=MEDIA_CONTACT_SHEET_STEP_NAME)(check)

    with pytest.raises(ValueError, match=r"media/contact_sheet.*already registered"):
        app.enrich(version="1", name=MEDIA_CONTACT_SHEET_STEP_NAME)(enrichment)


def test_enrichment_uses_alias_is_preflighted(source_episode: Path, tmp_path: Path) -> None:
    app = hflow.App("enrich-preflight", data_root=tmp_path)

    @app.enrich(version="1", uses="captioner")
    def needs_endpoint(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult()

    with pytest.raises(ValueError, match="captioner"):
        app.test(source_episode, verbose=False)


def test_enrichment_label_claiming_the_artifact_namespace_is_refused(
    source_episode: Path, tmp_path: Path
) -> None:
    """A user label under `artifact/` is indistinguishable from a published
    artifact in the catalog, and snapshot.py ships every such key as media."""
    data_root = tmp_path / "data"
    app = hflow.App("artifact-claim", data_root=data_root)

    @app.enrich(version="1")
    def labeling(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult(
            labels=cast(
                dict,
                {"artifact/notes": "s3://bucket/notes", "caption": "legit label"},
            )
        )

    with pytest.raises(ValueError, match=r"'artifact/notes'.*'labeling'"):
        app.test(source_episode, verbose=False, record=True)
    assert list((data_root / "catalog" / "episodes").glob("*.parquet")) == []


def test_check_measurement_claiming_the_artifact_namespace_is_refused(
    source_episode: Path, tmp_path: Path
) -> None:
    """A check measurement key under `artifact/` is refused the same way."""
    data_root = tmp_path / "data"
    app = hflow.App("artifact-claim-check", data_root=data_root)

    @app.check(version="1")
    def labeled(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"artifact/frames": 1.0})

    with pytest.raises(ValueError, match=r"'artifact/frames'.*'labeled'"):
        app.test(source_episode, verbose=False, record=True)
    assert list((data_root / "catalog" / "episodes").glob("*.parquet")) == []


def test_labels_near_the_artifact_namespace_still_land_with_real_artifacts(
    source_episode: Path, tmp_path: Path
) -> None:
    """Only the exact `artifact/` prefix is reserved; real artifacts land unchanged."""
    data_root = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    app = hflow.App("artifact-boundary", data_root=data_root)

    @app.enrich(version="1")
    def labeling(ep: hflow.Episode) -> hflow.EnrichmentResult:
        artifact_path = artifact_dir / "segments.json"
        artifact_path.write_text(json.dumps({"segments": []}))
        return hflow.EnrichmentResult(
            labels=cast(dict, {"artifact_notes": "s3://bucket/notes"}),
            artifacts={"segments": artifact_path},
        )

    app.test(source_episode, verbose=False, record=True)
    connection = open_catalog_connection(data_root / "catalog")
    try:
        label_row = connection.execute(
            "SELECT value_text FROM measurements WHERE key = 'artifact_notes'"
        ).fetchone()
        assert label_row == ("s3://bucket/notes",)
        artifact_row = connection.execute(
            "SELECT value_text FROM measurements WHERE key = 'artifact/segments'"
        ).fetchone()
        assert artifact_row is not None
        assert str(artifact_row[0]).endswith("segments.json")
    finally:
        connection.close()


_EGOCENTRIC_PIPELINE = Path(__file__).resolve().parents[1] / "examples/egocentric/pipeline.py"


def test_the_egocentric_example_renders_a_sheet_on_a_multi_camera_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The example itself, not a copy of its shape.

    A test that defines its own enrichment calling
    ``frames(camera=..., fps=1.0)`` passes whether or not the example does the
    same, so it cannot catch a regression in the file it ships alongside. This
    loads the real pipeline: reverting the fix in
    ``examples/egocentric/pipeline.py`` turns ``contact_sheet`` into an ERROR
    here.

    The example resolves DATA_ROOT relative to the working directory, so the
    chdir keeps its artifacts inside tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    raw = synthesize_episode(
        tmp_path / "raw.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam", "top_cam")),
    )
    canonical = tmp_path / "canonical.mcap"
    write_canonical_episode(raw, canonical)

    application = hflow.import_pipeline_application(str(_EGOCENTRIC_PIPELINE))
    report = application.test(canonical, verbose=False)

    sheet_run = next(run for run in report.enrichments if run.enrichment.name == "contact_sheet")
    assert sheet_run.status is hflow.CheckStatus.MEASURED, sheet_run.error
    assert sheet_run.result is not None
    # The episode has two cameras and the sheet names the one it rendered.
    assert sheet_run.result.labels["contact_sheet_camera"] == "/top_cam/compressed"
