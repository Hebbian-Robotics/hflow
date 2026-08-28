"""Opt-in gates: shipped recommended thresholds that never fire uninvited."""

import math
import re
from pathlib import Path
from typing import cast

import pytest

import hflow
from hflow.checks import (
    RECOMMENDED_CAMERA_INTEGRITY,
    idle_fraction,
    joint_discontinuity,
    trajectory_metrics,
    trajectory_segments,
)
from hflow.steps import evaluate_gate
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode


def _state_only_episode(tmp_path: Path) -> Path:
    return synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=(), joint_jump_at_s=1.0),
    )


def _gate(*thresholds: hflow.Threshold) -> hflow.Gate:
    return hflow.Gate(accept_when=thresholds)


def test_a_gate_fires_only_when_the_pipeline_opts_in(tmp_path: Path) -> None:
    """The requirement: HFlow ships the number, the pipeline decides it gates.

    Same check, same evidence, both directions -- without a gate the episode is
    evidence only; with one it quarantines.
    """
    episode_path = _state_only_episode(tmp_path)

    ungated = hflow.App("ungated", data_root=tmp_path / "ungated", default_checks=())

    @ungated.check(version="1", critical=True)
    def blackout(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"black_frame_pct": 99.0})

    ungated_report = ungated.test(episode_path, verbose=False)
    assert ungated_report.checks[0].result is not None
    assert ungated_report.checks[0].result.verdict is None
    assert ungated_report.checks[0].status is hflow.CheckStatus.MEASURED
    assert not ungated_report.quarantined

    gated = hflow.App("gated", data_root=tmp_path / "gated", default_checks=())

    @gated.check(version="1", critical=True, gate=RECOMMENDED_CAMERA_INTEGRITY)
    def blackout_gated(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"black_frame_pct": 99.0})

    gated_report = gated.test(episode_path, verbose=False)
    assert gated_report.checks[0].status is hflow.CheckStatus.FAILED
    assert gated_report.quarantine_tags == ["quarantined:blackout_gated"]


def test_a_shipped_gate_accepts_healthy_evidence(tmp_path: Path) -> None:
    app = hflow.App("healthy", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1", critical=True, gate=RECOMMENDED_CAMERA_INTEGRITY)
    def camera(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"black_frame_pct": 0.0, "freeze_total_s": 0.0})

    report = app.test(_state_only_episode(tmp_path), verbose=False)
    assert report.checks[0].status is hflow.CheckStatus.PASSED
    assert not report.quarantined


def test_no_shipped_gate_thresholds_a_motion_smoothness_key(tmp_path: Path) -> None:
    """Smoothness metrics ship as flags only, never a default reject rule.

    Voxel51's audit found them scoring an early-gripper-release defect better
    than clean demos, so a shipped threshold on one would reject the wrong
    episodes with our name on it. This walks the keys the smoothness checks
    actually emit rather than trusting the constant to look innocent.
    """
    source = _state_only_episode(tmp_path)
    with hflow.Episode(source) as episode:
        smoothness_keys: set[str] = set()
        for smoothness_check in (
            joint_discontinuity,
            idle_fraction,
            trajectory_metrics,
            trajectory_segments,
        ):
            smoothness_keys |= set(smoothness_check(episode).measurements)
    assert smoothness_keys, "fixture produced no smoothness measurements to check against"

    shipped_gates = [
        value for value in vars(hflow.checks).values() if isinstance(value, hflow.Gate)
    ]
    assert shipped_gates, "no shipped Gate constants found to audit"
    for gate in shipped_gates:
        for key in smoothness_keys:
            decision = evaluate_gate(gate, {key: 0.0})
            assert isinstance(decision, hflow.GateAbstained), (
                f"a shipped gate thresholds the motion-smoothness key {key!r}"
            )


def test_a_clause_matching_no_key_abstains_instead_of_passing(tmp_path: Path) -> None:
    """A typo'd threshold must not report a pass over evidence nobody read."""
    decision = evaluate_gate(
        _gate(hflow.Threshold("*/absent_key", hflow.Comparison.AT_MOST, 1.0)),
        {"/cam/black_frame_pct": 0.0},
    )
    assert decision == hflow.GateAbstained(unevaluated_patterns=("*/absent_key",))

    app = hflow.App("abstain", data_root=tmp_path / "data", default_checks=())

    @app.check(
        version="1",
        critical=True,
        gate=_gate(hflow.Threshold("*/nope", hflow.Comparison.AT_MOST, 1.0)),
    )
    def measures(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"present": 5.0})

    report = app.test(_state_only_episode(tmp_path), verbose=False)
    run_result = report.checks[0].result
    assert run_result is not None
    assert run_result.verdict is None
    assert report.checks[0].status is hflow.CheckStatus.MEASURED
    assert not report.quarantined
    assert "gate-unevaluated:*/nope" in run_result.tags


def test_a_failing_threshold_rejects_even_while_another_is_unreadable() -> None:
    """A conjunction is settled by one false conjunct, so a blatant defect must
    not slip through because an unrelated threshold had no key to read.
    """
    decision = evaluate_gate(
        RECOMMENDED_CAMERA_INTEGRITY,
        {"black_frame_pct": 99.0},  # no freeze_total_s for the second clause
    )
    assert decision == hflow.GateDecided(verdict=False)


def test_non_numeric_and_nan_measurements_abstain_but_infinity_compares() -> None:
    threshold = hflow.Threshold("value", hflow.Comparison.AT_MOST, 50.0)
    for unusable in ("text", True, math.nan):
        decision = evaluate_gate(_gate(threshold), {"value": unusable})
        assert isinstance(decision, hflow.GateAbstained), f"{unusable!r} should not be compared"
    assert evaluate_gate(_gate(threshold), {"value": math.inf}) == hflow.GateDecided(verdict=False)


def test_every_key_and_any_key_aggregate_over_topic_prefixed_keys() -> None:
    measurements: dict[str, hflow.MeasurementValue] = {
        "/left/black_frame_pct": 0.0,
        "/right/black_frame_pct": 99.0,
    }
    every = hflow.Threshold("*/black_frame_pct", hflow.Comparison.AT_MOST, 50.0)
    any_key = hflow.Threshold(
        "*/black_frame_pct", hflow.Comparison.AT_MOST, 50.0, hflow.Aggregation.ANY_KEY
    )
    assert evaluate_gate(_gate(every), measurements) == hflow.GateDecided(verdict=False)
    assert evaluate_gate(_gate(any_key), measurements) == hflow.GateDecided(verdict=True)


def test_comparisons_are_inclusive_at_the_threshold() -> None:
    at_most = hflow.Threshold("v", hflow.Comparison.AT_MOST, 50.0)
    at_least = hflow.Threshold("v", hflow.Comparison.AT_LEAST, 50.0)
    assert evaluate_gate(_gate(at_most), {"v": 50.0}) == hflow.GateDecided(verdict=True)
    assert evaluate_gate(_gate(at_least), {"v": 50.0}) == hflow.GateDecided(verdict=True)


def test_a_gate_can_only_tighten_a_checks_own_verdict(tmp_path: Path) -> None:
    """A gate is an additional accept condition, so it never resurrects an
    episode the check itself rejected.
    """
    app = hflow.App("compose", data_root=tmp_path / "data", default_checks=())
    permissive = _gate(hflow.Threshold("v", hflow.Comparison.AT_MOST, 100.0))

    @app.check(version="1", critical=True, gate=permissive)
    def rejects_itself(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"v": 1.0}, verdict=False)

    report = app.test(_state_only_episode(tmp_path), verbose=False)
    assert report.checks[0].status is hflow.CheckStatus.FAILED
    assert report.quarantine_tags == ["quarantined:rejects_itself"]


def test_a_gate_on_a_noncritical_check_tags_and_the_run_proceeds(tmp_path: Path) -> None:
    app = hflow.App("flags-only", data_root=tmp_path / "data", default_checks=())

    @app.check(version="1", gate=_gate(hflow.Threshold("v", hflow.Comparison.AT_MOST, 1.0)))
    def flags(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"v": 99.0})

    @app.check(version="1")
    def runs_after(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"ran": 1})

    report = app.test(_state_only_episode(tmp_path), verbose=False)
    flags_result = report.checks[0].result
    assert flags_result is not None
    assert "failed:flags" in flags_result.tags
    assert not report.quarantined
    assert report.checks[1].status is hflow.CheckStatus.MEASURED


def test_a_gate_uses_the_version_the_pipeline_author_declares() -> None:
    def probe(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    strict = hflow.App("strict", default_checks=())
    strict.check(
        version="quality-v2",
        gate=_gate(hflow.Threshold("v", hflow.Comparison.AT_MOST, 30.0)),
    )(probe)
    retuned_without_a_bump = hflow.App("retuned", default_checks=())
    retuned_without_a_bump.check(
        version="quality-v2",
        gate=_gate(hflow.Threshold("v", hflow.Comparison.AT_MOST, 50.0)),
    )(probe)
    bumped = hflow.App("bumped", default_checks=())
    bumped.check(
        version="quality-v3",
        gate=_gate(hflow.Threshold("v", hflow.Comparison.AT_MOST, 50.0)),
    )(probe)

    assert strict.checks[0].version == retuned_without_a_bump.checks[0].version
    assert bumped.checks[0].version != strict.checks[0].version


def test_a_non_gate_argument_is_refused_at_registration() -> None:
    app = hflow.App("bad-gate", data_root=Path("/tmp"), default_checks=())

    with pytest.raises(ValueError, match=re.escape("expected an hflow.Gate")):

        @app.check(version="1", gate=cast(hflow.Gate, "black_frame_pct < 50"))
        def wrong(ep: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult()

    assert app.checks == []


def test_an_empty_gate_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="holds no thresholds"):
        hflow.Gate(accept_when=())


def test_a_nan_threshold_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="not NaN"):
        hflow.Threshold("v", hflow.Comparison.AT_MOST, math.nan)


@pytest.mark.parametrize(
    "value,desc",
    [
        pytest.param(float("inf"), "+inf", id="positive_inf"),
        pytest.param(float("-inf"), "-inf", id="negative_inf"),
    ],
)
def test_an_infinite_threshold_is_refused_at_construction(
    value: float, desc: str
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        hflow.Threshold("v", hflow.Comparison.AT_MOST, value)


@pytest.mark.parametrize(
    "value,desc",
    [
        pytest.param(True, "True", id="true"),
        pytest.param(False, "False", id="false"),
    ],
)
def test_a_bool_threshold_is_refused_at_construction(
    value: bool, desc: str
) -> None:
    with pytest.raises(ValueError, match=r"not True|not False"):
        hflow.Threshold("v", hflow.Comparison.AT_MOST, value)
