"""External consumers receive selected evidence and explicit execution outcomes."""

import asyncio
import json
from pathlib import Path

import pytest

import hflow
from hflow.evidence import CameraQualityEvidence
from hflow.results import (
    CheckSelection,
    MeasuredCheck,
    ResultProjection,
    UnavailableCheck,
    project_results,
)
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
    async def private_measurement(_episode: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(
            measurements={"internal_score": 25.0, "private_prompt": "secret"},
            tags=["private_tag"],
            verdict=False,
        )

    @application.check(version="private-version")
    async def private_failure(_episode: hflow.Episode) -> hflow.CheckResult:
        raise RuntimeError("private diagnostic")

    report = asyncio.run(application.process(source, record=False))
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
    projection = project_results(
        report, (CheckSelection("private_measurement", "quality", {"score": "missing"}),)
    )
    match projection.checks[0]:
        case MeasuredCheck():
            raise AssertionError("expected superseded, got measured")
        case UnavailableCheck(status="superseded"):
            pass


def test_project_results_handles_superseded_default_without_keyerror(
    tmp_path: Path,
) -> None:
    """A default check runs in full, then a later user step with ``requires``
    overlaps one of its keys. ``_yield_defaults_superseded_by_the_pipeline``
    filters the overlapping key post-execution; project_results reports
    ``superseded`` for the filtered key instead of raising KeyError, and
    ``measured`` for the keys that survive.
    """
    source = synthesize_episode(
        tmp_path / "source.mcap", SyntheticEpisodeSpec(duration_s=1.0, cameras=())
    )
    application = hflow.App("superseded-default", data_root=tmp_path / "workspace")

    @application.check(version="1", requires={"pod"})
    async def my_duration(_episode: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"duration_s": 1.0})

    report = asyncio.run(application.process(source, record=False))
    by_name = {run.check.name: run for run in report.checks}

    # The default ran in full, then had duration_s filtered out post-execution.
    default = by_name["episode_duration"]
    assert default.result is not None
    assert default.status is hflow.CheckStatus.MEASURED
    assert "duration_s" not in default.result.measurements
    assert "message_count_total" in default.result.measurements
    assert any(tag == "superseded:duration_s" for tag in default.result.tags)

    # project_results reports superseded for the filtered key, not KeyError.
    projection = project_results(
        report,
        (CheckSelection("episode_duration", "duration", {"value": "duration_s"}),),
    )
    match projection.checks[0]:
        case MeasuredCheck():
            raise AssertionError("expected superseded, got measured")
        case UnavailableCheck(status="superseded"):
            pass

    # project_results reports measured for a surviving key.
    projection = project_results(
        report,
        (
            CheckSelection(
                "episode_duration",
                "count",
                {"count": "message_count_total"},
            ),
        ),
    )
    match projection.checks[0]:
        case MeasuredCheck(measurements=measurements):
            assert measurements == {"count": default.result.measurements["message_count_total"]}
        case UnavailableCheck():
            raise AssertionError("expected measured, got superseded")


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
