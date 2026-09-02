"""The vertical slice, end to end: one episode in, measurements out, zero
infrastructure. Mirrors the README design-target example."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hflow
from hflow.checks import camera_frame_stats
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

    @app.check(version="1")
    def joint_smoothness(ep: hflow.Episode) -> hflow.CheckResult:
        joints = ep.channel("/joint_states").to_numpy()
        result = check_joint_smoothness(joints, rate_hz=100)
        return hflow.CheckResult(measurements=dict(result))

    @app.check(version="1")
    def timestamps(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.timestamp_regularity(ep, tolerance_s=0.001)

    @app.check(version="1")
    def joint_jumps(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.joint_discontinuity(ep, velocity_limit=3.0)

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


def test_report_returns_a_named_check_and_explains_missing_names(
    report_and_app: tuple[hflow.TestReport, hflow.App],
) -> None:
    report, _application = report_and_app

    named_check_run = report.check("joint_smoothness")

    assert named_check_run.check.name == "joint_smoothness"
    assert named_check_run.result is not None
    with pytest.raises(KeyError) as missing_check_error:
        report.check("unregistered_check")
    assert "test report has no check named 'unregistered_check'" in str(missing_check_error.value)
    assert "'joint_smoothness'" in str(missing_check_error.value)


def test_many_runs_distinct_episodes_and_preserves_input_order(tmp_path: Path) -> None:
    source_paths = tuple(
        synthesize_episode(
            tmp_path / "sources" / f"episode_{episode_number:04d}.mcap",
            SyntheticEpisodeSpec(
                duration_s=0.1,
                cameras=(),
                task=f"task-{episode_number}",
            ),
        )
        for episode_number in range(3)
    )
    application = hflow.App(
        "batch-test",
        data_root=tmp_path / "data",
        default_checks=(),
    )

    @application.check(version="1")
    def episode_task(episode: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"episode/task": str(episode.metadata["task"])})

    requested_source_paths = (source_paths[2], source_paths[0], source_paths[1])
    progress_events: list[hflow.TestManyProgress] = []
    batch_report = application.test_many(
        requested_source_paths,
        max_workers=2,
        stages=(hflow.Stage.SYNC, hflow.Stage.META),
        on_progress=progress_events.append,
    )
    reports = batch_report.reports

    assert [report.source_path for report in reports] == list(requested_source_paths)
    assert batch_report.has_errors is False
    assert [progress.completed_count for progress in progress_events] == [1, 2, 3]
    assert {progress.total_count for progress in progress_events} == {3}
    assert {progress.input_index for progress in progress_events} == {0, 1, 2}
    assert all(
        progress.report.source_path == requested_source_paths[progress.input_index]
        for progress in progress_events
    )
    measured_tasks: list[str] = []
    for report in reports:
        task_check_run = report.check("episode_task")
        assert task_check_run.result is not None
        measured_tasks.append(str(task_check_run.result.measurements["episode/task"]))
    assert measured_tasks == ["task-2", "task-0", "task-1"]


def test_many_refuses_duplicate_source_identities(tmp_path: Path) -> None:
    source_path = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=0.1, cameras=()),
    )
    application = hflow.App("batch-test", data_root=tmp_path / "data", default_checks=())

    with pytest.raises(ValueError, match="duplicate episode source identity"):
        application.test_many((source_path, source_path), max_workers=2)


@pytest.mark.parametrize("invalid_max_workers", (0, -1, True))
def test_many_refuses_invalid_concurrency_limits(
    tmp_path: Path,
    invalid_max_workers: int,
) -> None:
    application = hflow.App("batch-test", data_root=tmp_path / "data", default_checks=())

    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        application.test_many((), max_workers=invalid_max_workers)


def test_many_stops_scheduling_new_episodes_after_preparation_failure(
    tmp_path: Path,
) -> None:
    first_valid_source = synthesize_episode(
        tmp_path / "sources" / "first-valid.mcap",
        SyntheticEpisodeSpec(duration_s=0.1, cameras=()),
    )
    source_that_must_not_start = synthesize_episode(
        tmp_path / "sources" / "must-not-start.mcap",
        SyntheticEpisodeSpec(duration_s=0.1, cameras=()),
    )
    missing_source = tmp_path / "sources" / "missing.mcap"
    application = hflow.App("batch-test", data_root=tmp_path / "data", default_checks=())

    with pytest.raises(FileNotFoundError):
        application.test_many(
            (missing_source, first_valid_source, source_that_must_not_start),
            max_workers=2,
            stages=(hflow.Stage.SYNC,),
        )

    test_run_directories = tuple(application.workspace.test_runs_root.workspace.iterdir())
    assert not any(
        run_directory.name.startswith(f"{source_that_must_not_start.stem}-")
        for run_directory in test_run_directories
    )


def test_many_stops_scheduling_new_episodes_after_progress_failure(tmp_path: Path) -> None:
    source_paths = tuple(
        synthesize_episode(
            tmp_path / "sources" / f"episode_{episode_number}.mcap",
            SyntheticEpisodeSpec(duration_s=0.1, cameras=()),
        )
        for episode_number in range(3)
    )
    application = hflow.App("batch-test", data_root=tmp_path / "data", default_checks=())

    def reject_first_progress(_progress: hflow.TestManyProgress) -> None:
        raise RuntimeError("progress consumer failed")

    with pytest.raises(RuntimeError, match="progress consumer failed"):
        application.test_many(
            source_paths,
            max_workers=2,
            stages=(hflow.Stage.SYNC,),
            on_progress=reject_first_progress,
        )

    test_run_directories = tuple(application.workspace.test_runs_root.workspace.iterdir())
    assert not any(
        run_directory.name.startswith(f"{source_paths[2].stem}-")
        for run_directory in test_run_directories
    )


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


def test_canonical_episode_extracts_exact_source_frame_indices(
    report_and_app: tuple[hflow.TestReport, hflow.App],
) -> None:
    report, _app = report_and_app
    selected_frame_indices = [0, 1, 2, 7, 8, 29]
    with hflow.Episode(report.canonical_path) as episode:
        camera_topic = next(topic for topic in episode.cameras if "overhead_cam" in topic)
        expected_log_times = episode.channel(camera_topic).timestamps[selected_frame_indices]

        selected_frames = episode.frames_at_indices(
            camera_topic,
            frame_indices=selected_frame_indices,
        )

        numpy_frame_indices = [
            np.int32(0),
            np.int64(1),
            np.uint32(2),
            np.uint64(7),
            np.int64(8),
            np.uint32(29),
        ]
        numpy_frames = episode.frames_at_indices(
            camera_topic,
            frame_indices=numpy_frame_indices,
        )

        assert [frame.log_time_ns for frame in selected_frames] == expected_log_times.tolist()
        assert [frame.log_time_ns for frame in numpy_frames] == expected_log_times.tolist()
        assert [frame.path.read_bytes() for frame in numpy_frames] == [
            frame.path.read_bytes() for frame in selected_frames
        ]
        assert numpy_frames == selected_frames
        assert all(frame.path.read_bytes()[:2] == b"\xff\xd8" for frame in selected_frames)
        assert (
            episode.frames_at_indices(
                camera_topic,
                frame_indices=selected_frame_indices,
            )
            == selected_frames
        )


@pytest.mark.parametrize(
    "invalid_frame_indices",
    [[True], [np.bool_(True)], [3.0], [np.float64(3.0)]],
)
def test_canonical_episode_rejects_non_integer_frame_indices(
    report_and_app: tuple[hflow.TestReport, hflow.App],
    invalid_frame_indices: list[Any],
) -> None:
    report, _app = report_and_app
    with hflow.Episode(report.canonical_path) as episode:
        camera_topic = next(topic for topic in episode.cameras if "overhead_cam" in topic)
        with pytest.raises(ValueError):
            episode.frames_at_indices(
                camera_topic,
                frame_indices=invalid_frame_indices,
            )


def test_arrow_export(report_and_app: tuple[hflow.TestReport, hflow.App]) -> None:
    report, _app = report_and_app
    with hflow.Episode(report.canonical_path) as episode:
        table: Any = episode.channel("/joint_states").to_arrow()
    assert table.num_rows == 200
    assert "log_time_ns" in table.column_names
    assert "position" in table.column_names


def test_channel_exposes_publish_times(
    report_and_app: tuple[hflow.TestReport, hflow.App],
) -> None:
    report, _app = report_and_app
    with hflow.Episode(report.canonical_path) as episode:
        channel = episode.channel("/joint_states")
        publish_times = channel.publish_times

    assert publish_times.shape == (len(channel),)
    # Same length as the log times they pair with, so the two can be
    # subtracted to get per-message recording latency.
    assert publish_times.shape == channel.timestamps.shape


def test_failed_critical_verdict_quarantines_and_skips_downstream(
    source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App("strict-pipeline", data_root=tmp_path)

    @app.check(version="1", critical=True)
    def camera_blackout(ep: hflow.Episode) -> hflow.CheckResult:
        camera_topic = next(topic for topic in ep.cameras if "wrist_cam" in topic)
        camera_evidence = camera_frame_stats(ep, cameras=[camera_topic])
        black_frame_percent = camera_evidence.measurements[f"{camera_topic}/black_frame_pct"]
        assert isinstance(black_frame_percent, float)
        return hflow.CheckResult(
            measurements={"black_pct": black_frame_percent},
            verdict=black_frame_percent < 1.0,  # fixture blackout exceeds this
        )

    @app.check(version="1")
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

    @app.check(version="1")
    def exploding(ep: hflow.Episode) -> hflow.CheckResult:
        raise RuntimeError("boom")

    @app.check(version="1")
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
    app = hflow.App("ordered-pipeline", data_root=tmp_path)
    execution_order: list[str] = []

    @app.check(version="1", requires=("vision-model",))
    def expensive(ep: hflow.Episode) -> hflow.CheckResult:
        execution_order.append("expensive")
        return hflow.CheckResult()

    @app.check(version="1")
    def cheap(ep: hflow.Episode) -> hflow.CheckResult:
        execution_order.append("cheap")
        return hflow.CheckResult()

    app.test(state_only_source_episode, verbose=False)
    assert execution_order == ["cheap", "expensive"]


def test_check_claiming_an_episode_column_refuses_the_append(
    state_only_source_episode: Path, tmp_path: Path
) -> None:
    """The #130 repro: a check measuring ``task`` used to land as ``task_1``
    beside the real column, so SELECT task returned the metadata (here
    'fold_napkin') and never the measurement. The append now refuses."""
    app = hflow.App("reserved-key", data_root=tmp_path / "data")

    @app.check(version="1")
    def claims_task(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"task": 99.0})

    with pytest.raises(ValueError, match=r"'claims_task'.*shadows 'task'"):
        app.test(state_only_source_episode, verbose=False, record=True)
