"""External consumers receive selected evidence and explicit execution outcomes."""

import json
from pathlib import Path

import pytest

import hflow
from hflow.evidence import CameraQualityEvidence
from hflow.results import CheckSelection, MeasuredCheck, ResultProjection, project_results
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode


def test_result_projection_preserves_verdicts_without_disclosing_internal_details(
    tmp_path: Path,
) -> None:
    source = synthesize_episode(
        tmp_path / "private-source.mcap", SyntheticEpisodeSpec(duration_s=0.2, cameras=())
    )
    application = hflow.App(
        "private-application", data_root=tmp_path / "workspace", default_checks=()
    )

    @application.check(version="private-version")
    def private_measurement(_episode: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(
            measurements={"internal_score": 25.0, "private_prompt": "secret"},
            tags=["private_tag"],
            verdict=False,
        )

    @application.check(version="private-version")
    def private_failure(_episode: hflow.Episode) -> hflow.CheckResult:
        raise RuntimeError("private diagnostic")

    report = application.process(source, record=False)
    projection = project_results(
        report,
        (
            CheckSelection("private_measurement", "quality", {"score": "internal_score"}),
            CheckSelection("private_failure", "activity", {}),
        ),
    )
    encoded = projection.to_json(maximum_bytes=1024)
    assert json.loads(encoded) == {
        "schema_version": 1,
        "checks": [
            {
                "name": "quality",
                "status": "measured",
                "measurements": {"score": 25.0},
                "verdict": False,
            },
            {"name": "activity", "status": "error"},
        ],
    }
    assert b"private" not in encoded
    result = report.check("private_measurement").result
    assert result is not None
    result.measurements["internal_score"] = 99.0
    assert projection.to_json(maximum_bytes=len(encoded)) == encoded
    with pytest.raises(ValueError, match="byte limit"):
        projection.to_json(maximum_bytes=len(encoded) - 1)
    with pytest.raises(KeyError):
        project_results(report, (CheckSelection("unknown", "quality", {}),))
    with pytest.raises(KeyError):
        project_results(
            report, (CheckSelection("private_measurement", "quality", {"score": "missing"}),)
        )


def test_result_domain_refuses_nonfinite_measurements_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="finite"):
        MeasuredCheck("quality", {"score": float("nan")}, None)
    measured = MeasuredCheck("quality", {"score": 0.0}, None)
    with pytest.raises(ValueError, match="unique"):
        ResultProjection((measured, measured))


def test_camera_evidence_preserves_units_and_rejects_invalid_percentages() -> None:
    measurements = {
        "/head/black_frame_pct": 25.0,
        "/head/clipped_highlight_pct": 10.0,
        "/head/crushed_shadow_pct": 0.0,
        "/head/freeze_total_s": 2.0,
    }
    evidence = CameraQualityEvidence.from_measurements(measurements, "/head")
    assert evidence.black_frame_percent == 25.0
    assert evidence.frozen_seconds == 2.0
    with pytest.raises(ValueError, match="range"):
        CameraQualityEvidence.from_measurements(
            {**measurements, "/head/black_frame_pct": 101.0}, "/head"
        )
