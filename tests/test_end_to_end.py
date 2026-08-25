"""The vertical slice, end to end: one episode in, measurements out, zero
infrastructure. Mirrors the README design-target example."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hflow
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode


def check_joint_smoothness(joints: np.ndarray, rate_hz: float) -> dict[str, float]:
    velocities = np.abs(np.diff(joints, axis=0)) * rate_hz
    return {
        "max_velocity_rad_s": float(velocities.max()),
        "mean_velocity_rad_s": float(velocities.mean()),
    }


@pytest.fixture(scope="module")
def source_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(
        tmp_path_factory.mktemp("e2e-source") / "episode_0001.mcap",
        SyntheticEpisodeSpec(
            duration_s=2.0,
            black_segment=(0.5, 0.75),
            joint_jump_at_s=1.0,
            timestamp_offset_segment=(1.4, 1.7),
        ),
    )


@pytest.fixture(scope="module")
def state_only_source_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(
        tmp_path_factory.mktemp("e2e-state-only-source") / "episode_0001.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=(), joint_jump_at_s=1.0),
    )


@pytest.fixture(scope="module")
def report_and_app(
    source_episode: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[hflow.TestReport, hflow.App]:
    app = hflow.App("kitchen-pipeline", data_root=tmp_path_factory.mktemp("e2e-data"))

    @app.check()
    def joint_smoothness(ep: hflow.Episode) -> hflow.CheckResult:
        joints = ep.channel("/joint_states").to_numpy()
        result = check_joint_smoothness(joints, rate_hz=100)
        return hflow.CheckResult(measurements=dict(result))

    @app.check()
    def timestamps(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.timestamp_regularity(ep, tolerance_s=0.001)

    @app.check()
    def joint_jumps(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.joint_discontinuity(ep, velocity_limit=3.0)

    @app.check(critical=True)
    def camera_blackout(ep: hflow.Episode) -> hflow.CheckResult:
        stats = hflow.ffmpeg.frame_stats(ep.video("wrist_cam"))
        return hflow.CheckResult(
            measurements={"black_pct": stats.black_frame_pct},
            verdict=stats.black_frame_pct < 50.0,
        )

    report = app.test(source_episode, verbose=False)
    return report, app


def test_whole_pipeline_runs_without_quarantine(
    report_and_app: tuple[hflow.TestReport, hflow.App],
) -> None:
    report, _app = report_and_app
    assert not report.quarantined
    assert report.canonical_path.is_file()
    status_by_check = {run.check.name: run.status for run in report.checks}
    assert status_by_check["joint_smoothness"] == "measured"
    assert status_by_check["timestamps"] == "measured"
    assert status_by_check["joint_jumps"] == "measured"
    assert status_by_check["camera_blackout"] == "passed"
    # The baseline nobody registered, running beside the pipeline's own steps.
    assert status_by_check["episode_duration"] == "measured"
    assert status_by_check["media_digest"] == "measured"
    # ...except where the pipeline already measures the same thing: `timestamps`
    # wraps timestamp_regularity, so the automatic copy yields rather than
    # colliding with it.
    assert status_by_check["timestamp_regularity"] == "superseded"
    summary_text = report.summary()
    assert "pipeline_version=" in summary_text
    assert "camera_blackout" in summary_text


def test_ported_user_check_saw_real_joints(
    report_and_app: tuple[hflow.TestReport, hflow.App],
) -> None:
    report, _app = report_and_app
    by_name = {run.check.name: run for run in report.checks}
    smoothness = by_name["joint_smoothness"].result
    assert smoothness is not None
    max_velocity = smoothness.measurements["max_velocity_rad_s"]
    assert isinstance(max_velocity, float)
    # The fixture injects a 0.8 rad step at 100 Hz -> ~80 rad/s spike.
    assert max_velocity > 10.0


def test_builtin_checks_found_the_injected_defects(
    report_and_app: tuple[hflow.TestReport, hflow.App],
) -> None:
    report, _app = report_and_app
    by_name = {run.check.name: run for run in report.checks}

    jumps = by_name["joint_jumps"].result
    assert jumps is not None
    violation_count = jumps.measurements["/joint_states/violation_count"]
    assert isinstance(violation_count, int)
    assert violation_count >= 1
    assert jumps.intervals, "the injected joint jump must yield an interval"

    stamps_result = by_name["timestamps"].result
    assert stamps_result is not None
    median_dt = stamps_result.measurements["/joint_states/median_dt_s"]
    assert isinstance(median_dt, float)
    assert median_dt == pytest.approx(0.01, rel=1e-3)

    blackout = by_name["camera_blackout"].result
    assert blackout is not None
    black_pct = blackout.measurements["black_pct"]
    assert isinstance(black_pct, float)
    # The fixture blacks out 0.25s of the 2s wrist stream: ~12.5% of frames.
    assert 5.0 < black_pct < 25.0


def test_canonical_episode_accessors(
    report_and_app: tuple[hflow.TestReport, hflow.App],
) -> None:
    report, _app = report_and_app
    with hflow.Episode(report.canonical_path) as episode:
        assert len(episode.cameras) == 2
        assert episode.metadata["task"] == "fold_napkin"
        assert len(episode.metadata["pipeline_version"]) == 12

        joints = episode.channel("/joint_states").to_numpy()
        assert joints.shape[1] == 7

        mp4 = episode.video("overhead_cam")
        assert mp4.is_file() and mp4.stat().st_size > 0

        frames = episode.frames("overhead_cam", fps=2.0)
        assert 3 <= len(frames) <= 5  # ~2s at 2 fps
        assert frames[0].path.read_bytes()[:2] == b"\xff\xd8"
        # Frame log times map to SOURCE messages: at t=0.5s the fps filter
        # emits the frame visible then -- source frame floor(0.5 * 15) = 7,
        # i.e. round(7e9/15) ns after the first frame.
        assert abs(frames[1].log_time_ns - frames[0].log_time_ns - 466_666_667) <= 1


def test_arrow_export(report_and_app: tuple[hflow.TestReport, hflow.App]) -> None:
    report, _app = report_and_app
    with hflow.Episode(report.canonical_path) as episode:
        table: Any = episode.channel("/joint_states").to_arrow()
    assert table.num_rows == 200
    assert "log_time_ns" in table.column_names
    assert "position" in table.column_names


def test_failed_critical_verdict_quarantines_and_skips_downstream(
    source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App("strict-pipeline", data_root=tmp_path)

    @app.check(critical=True)
    def camera_blackout(ep: hflow.Episode) -> hflow.CheckResult:
        stats = hflow.ffmpeg.frame_stats(ep.video("wrist_cam"))
        return hflow.CheckResult(
            measurements={"black_pct": stats.black_frame_pct},
            verdict=stats.black_frame_pct < 1.0,  # fixture blackout exceeds this
        )

    @app.check()
    def never_reached(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"ran": True})

    report = app.test(source_episode, verbose=False)
    assert report.quarantined
    assert report.quarantine_tags == ["quarantined:camera_blackout"]
    by_name = {run.check.name: run for run in report.checks}
    assert by_name["never_reached"].status == "skipped"
    assert by_name["camera_blackout"].status == "failed"


def test_crashing_check_is_infrastructure_not_data(
    state_only_source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App("crashy-pipeline", data_root=tmp_path)

    @app.check()
    def exploding(ep: hflow.Episode) -> hflow.CheckResult:
        raise RuntimeError("boom")

    @app.check()
    def still_runs(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"ran": True})

    report = app.test(state_only_source_episode, verbose=False)
    assert not report.quarantined
    by_name = {run.check.name: run for run in report.checks}
    assert by_name["exploding"].status == "error"
    assert by_name["exploding"].error is not None and "boom" in by_name["exploding"].error
    assert by_name["still_runs"].status == "measured"


def test_resource_declaring_checks_run_after_plain_ones(
    state_only_source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App(
        "ordered-pipeline", data_root=tmp_path, endpoints={"judge": "http://localhost:9"}
    )
    execution_order: list[str] = []

    @app.check(uses="judge")
    def expensive(ep: hflow.Episode) -> hflow.CheckResult:
        execution_order.append("expensive")
        return hflow.CheckResult()

    @app.check()
    def cheap(ep: hflow.Episode) -> hflow.CheckResult:
        execution_order.append("cheap")
        return hflow.CheckResult()

    app.test(state_only_source_episode, verbose=False)
    assert execution_order == ["cheap", "expensive"]


def test_missing_provider_alias_fails_preflight(
    state_only_source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App("misconfigured", data_root=tmp_path)

    @app.check(uses="judge")
    def needs_endpoint(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    with pytest.raises(ValueError, match="judge") as error_info:
        app.test(state_only_source_episode, verbose=False)
    # The failure names the environment escape hatch a deployment would use.
    assert "HFLOW_ENDPOINT_JUDGE" in str(error_info.value)


def test_endpoint_alias_satisfied_by_environment_only(
    state_only_source_episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment (or control plane) supplies an endpoint alias through
    HFLOW_ENDPOINT_<ALIAS>: preflight passes and the running step sees the
    injected value, with nothing endpoint-shaped in the pipeline file."""
    monkeypatch.setenv("HFLOW_ENDPOINT_JUDGE", "http://injected:8000/v1")
    app = hflow.App("env-endpoints", data_root=tmp_path)
    endpoint_values_seen_by_step: list[str] = []

    # version=: steps that read app.endpoints capture the App, which is
    # opaque to version hashing -- the documented pattern declares a version.
    @app.check(uses="judge", version="v1")
    def needs_endpoint(ep: hflow.Episode) -> hflow.CheckResult:
        endpoint_values_seen_by_step.append(app.endpoints["judge"])
        return hflow.CheckResult()

    report = app.test(state_only_source_episode, verbose=False)
    assert not report.has_errors
    assert endpoint_values_seen_by_step == ["http://injected:8000/v1"]


def test_endpoint_environment_override_wins_over_the_literal(
    state_only_source_episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HFLOW_ENDPOINT_JUDGE", "http://deployment-injected:9000")
    app = hflow.App(
        "env-endpoints", data_root=tmp_path, endpoints={"judge": "http://from-code:8000"}
    )
    endpoint_values_seen_by_step: list[str] = []

    @app.check(uses="judge", version="v1")
    def needs_endpoint(ep: hflow.Episode) -> hflow.CheckResult:
        endpoint_values_seen_by_step.append(app.endpoints["judge"])
        return hflow.CheckResult()

    app.test(state_only_source_episode, verbose=False)
    assert endpoint_values_seen_by_step == ["http://deployment-injected:9000"]

    # Unsetting the override restores the pipeline literal on the next run:
    # resolution rebuilds from the pristine literals, never mutates them.
    monkeypatch.delenv("HFLOW_ENDPOINT_JUDGE")
    app.test(state_only_source_episode, verbose=False)
    assert endpoint_values_seen_by_step[-1] == "http://from-code:8000"


def test_endpoints_mapping_refuses_direct_mutation(tmp_path: Path) -> None:
    """The resolved mapping is rebuilt at every run start, so a direct
    mutation would be silently discarded -- it must refuse loudly instead."""
    from typing import cast

    app = hflow.App("read-only", data_root=tmp_path, endpoints={"judge": "http://a:1"})
    with pytest.raises(TypeError):
        cast("dict[str, str]", app.endpoints)["judge"] = "http://mutated:2"


def test_colliding_endpoint_alias_names_are_refused(
    state_only_source_episode: Path, tmp_path: Path
) -> None:
    """HFLOW_ENDPOINT_* naming is lossy (non-alphanumerics collapse to '_'):
    two aliases mapping to one variable would be silently co-overridden, so
    preflight refuses the ambiguity loudly."""
    app = hflow.App(
        "colliding",
        data_root=tmp_path,
        endpoints={"judge-v1": "http://a:1", "judge.v1": "http://b:2"},
    )

    @app.check(uses="judge-v1", version="v1")
    def needs_endpoint(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    with pytest.raises(ValueError, match="HFLOW_ENDPOINT_JUDGE_V1"):
        app.test(state_only_source_episode, verbose=False)


def test_check_claiming_an_episode_column_refuses_the_append(
    state_only_source_episode: Path, tmp_path: Path
) -> None:
    """The #130 repro: a check measuring ``task`` used to land as ``task_1``
    beside the real column, so SELECT task returned the metadata (here
    'fold_napkin') and never the measurement. The append now refuses."""
    app = hflow.App("reserved-key", data_root=tmp_path / "data")

    @app.check()
    def claims_task(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"task": 99.0})

    with pytest.raises(ValueError, match=r"'claims_task'.*shadows 'task'"):
        app.test(state_only_source_episode, verbose=False, record=True)
