"""Enrichment step registration, execution, and catalog outcomes."""

import json
from pathlib import Path
from typing import cast

import pytest

import hflow
from hflow.curation import open_catalog_connection
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

FAST_SPEC = SyntheticEpisodeSpec(duration_s=2.0, cameras=())


@pytest.fixture()
def source_episode(tmp_path: Path) -> Path:
    return synthesize_episode(tmp_path / "episode.mcap", FAST_SPEC)


def test_enrichments_run_after_all_checks(source_episode: Path, tmp_path: Path) -> None:
    app = hflow.App("enrich-order", data_root=tmp_path / "data")
    execution_order: list[str] = []

    @app.enrich()
    def caption(ep: hflow.Episode) -> hflow.EnrichmentResult:
        execution_order.append("caption")
        return hflow.EnrichmentResult(labels={"caption": "a robot arm moves"})

    @app.check()
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

    @app.check(critical=True)
    def always_fails(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(verdict=False)

    @app.enrich()
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

    @app.enrich()
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

    @app.enrich()
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

    @app.check()
    def labeling(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    with pytest.raises(ValueError, match="already registered"):

        @app.enrich(name="labeling")
        def another(ep: hflow.Episode) -> hflow.EnrichmentResult:
            return hflow.EnrichmentResult()


def test_enrichment_uses_alias_is_preflighted(source_episode: Path, tmp_path: Path) -> None:
    app = hflow.App("enrich-preflight", data_root=tmp_path)

    @app.enrich(uses="captioner")
    def needs_endpoint(ep: hflow.Episode) -> hflow.EnrichmentResult:
        return hflow.EnrichmentResult()

    with pytest.raises(ValueError, match="captioner"):
        app.test(source_episode, verbose=False)
