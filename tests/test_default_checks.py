"""The baseline every episode gets without anyone registering it."""

from pathlib import Path

import pytest

import hflow
from hflow.checks import DEFAULT_CHECKS, camera_frame_stats, episode_duration
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode


@pytest.fixture(scope="module")
def source_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(
        tmp_path_factory.mktemp("defaults-source") / "episode_0001.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
    )


def test_a_pipeline_that_registers_nothing_still_records_evidence(
    source_episode: Path, tmp_path: Path
) -> None:
    app = hflow.App("baseline", data_root=tmp_path / "data")

    report = app.test(source_episode, verbose=False)

    # Pinned by name rather than derived from DEFAULT_CHECKS: which checks
    # every corpus pays for is a product decision, and a test that recomputes
    # it from the same tuple would agree with any change to it.
    measured = {run.check.name for run in report.checks if run.result is not None}
    assert measured == {
        "episode_duration",
        "timestamp_regularity",
        "camera_frame_stats",
        "keyframe_interval",
        "content_digest",
        "media_digest",
    }
    assert len(DEFAULT_CHECKS) == len(measured)
    duration = next(run for run in report.checks if run.check.name == "episode_duration")
    assert duration.result is not None
    assert duration.result.measurements["duration_s"] == pytest.approx(1.0, abs=0.2)


def test_the_baseline_can_be_turned_off_entirely(source_episode: Path, tmp_path: Path) -> None:
    app = hflow.App("bare", data_root=tmp_path / "data", default_checks=())

    assert app.test(source_episode, verbose=False).checks == []


def test_a_subset_is_expressible_because_that_is_the_real_need(
    source_episode: Path, tmp_path: Path
) -> None:
    """The reason this is a collection and not a boolean: keeping the baseline
    while dropping the one default you configure yourself."""
    app = hflow.App(
        "subset",
        data_root=tmp_path / "data",
        default_checks=[check for check in DEFAULT_CHECKS if check is not camera_frame_stats],
    )

    registered = {registered.name for registered in app.checks}
    assert "camera_frame_stats" not in registered
    assert "episode_duration" in registered


def test_registering_a_default_yourself_configures_it_rather_than_colliding(
    tmp_path: Path,
) -> None:
    app = hflow.App("configured", data_root=tmp_path / "data")

    app.check(critical=True)(episode_duration)

    registrations = [
        registered for registered in app.checks if registered.name == "episode_duration"
    ]
    assert len(registrations) == 1
    assert registrations[0].critical is True


def test_two_pipeline_steps_sharing_a_name_are_still_refused(tmp_path: Path) -> None:
    """Replacement applies to defaults only: between two of the pipeline's own
    steps the engine has no basis to pick a winner."""
    app = hflow.App("clash", data_root=tmp_path / "data")

    @app.check(name="mine")
    def first(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    with pytest.raises(ValueError, match="already registered"):

        @app.check(name="mine")
        def second(ep: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult()


def test_a_default_yields_to_a_pipeline_step_measuring_the_same_thing(
    source_episode: Path, tmp_path: Path
) -> None:
    """The documented way to configure a built-in is to wrap it under your own
    name, which emits the built-in's keys. That must not be a collision."""
    app = hflow.App("wrapping", data_root=tmp_path / "data")

    @app.check()
    def timestamps(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.timestamp_regularity(ep, tolerance_s=0.001)

    report = app.test(source_episode, verbose=False)

    by_name = {run.check.name: run for run in report.checks}
    assert by_name["timestamps"].result is not None
    superseded = by_name["timestamp_regularity"]
    # A status of its own, not `skipped`: this default will stand down on every
    # episode forever, where a quarantine skip lifts the moment the critical
    # check is retuned, and the planner and the dataset policy have to be able
    # to tell those apart (hflow.steps.SETTLED_STATUSES).
    assert superseded.status is hflow.CheckStatus.SUPERSEDED
    assert superseded.result is None
    assert isinstance(superseded.not_run, hflow.SupersededByPipeline)
    assert "default_checks" in superseded.not_run.reason
    # The pipeline's own measurements survive intact.
    assert any(key.endswith("/median_dt_s") for key in by_name["timestamps"].result.measurements)


def test_a_default_that_does_not_overlap_keeps_running(
    source_episode: Path, tmp_path: Path
) -> None:
    """Only the overlapping default yields; the rest of the baseline stands."""
    app = hflow.App("partial-overlap", data_root=tmp_path / "data")

    @app.check()
    def timestamps(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.timestamp_regularity(ep)

    report = app.test(source_episode, verbose=False)

    by_name = {run.check.name: run for run in report.checks}
    assert by_name["timestamp_regularity"].status is hflow.CheckStatus.SUPERSEDED
    assert by_name["episode_duration"].result is not None
    assert by_name["content_digest"].result is not None
