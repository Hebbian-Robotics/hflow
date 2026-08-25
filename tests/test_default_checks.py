"""The baseline every episode gets without anyone registering it."""

from pathlib import Path
from typing import Any

import pytest

import hflow
from hflow.checks import (
    _DEFAULT_KEY_PATTERNS,
    DEFAULT_CHECKS,
    camera_frame_stats,
    episode_duration,
)
from hflow.ffmpeg import _instrument
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


@pytest.fixture
def camera_source(tmp_path: Path) -> Path:
    """A short camera clip -- the smallest episode ``camera_frame_stats`` will
    measure, so the test only pays for the work the bug is about."""
    return synthesize_episode(
        tmp_path / "with_camera.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",)),
    )


def test_a_wrapper_with_non_default_parameters_supersedes_the_default_without_running_it(
    camera_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #177: a wrapper that calls ``camera_frame_stats`` with non-default
    parameters still triggers the automatic default to run in full, paying for
    a decode whose result is then thrown away at ``App._yield_defaults_super…``
    (src/hflow/app.py:987). The pipeline's step is what the user wants, the
    default's measurement is a duplicate-key collision the engine discards
    after the fact. The superseded default should never have run.

    The #175 cache collapses the ffmpeg decode count to one when the wrapper
    keeps the default's filter graph, but the default still executes in full:
    it calls ``frame_stats``, decodes the video, writes a cache file the
    wrapper will then read, and produces a ``CheckResult`` that
    ``_yield_defaults_superseded_by_the_pipeline`` then nulls out. That is
    wasted work whose only effect is to confirm the wrapper is doing the job
    the user asked for.

    This test counts ffmpeg decode passes (the ``-vf`` instrument pass) and
    asserts the default does no work after the wrapper registers under its
    own name with a non-default parameter. Concretely:

    - the default's ``CheckRunReport`` carries ``SupersededByPipeline`` as
      today, with the same reason;
    - the default's ``duration_s`` is zero (it never ran);
    - the wrapper's measurements are kept;
    - exactly one ffmpeg decode pass happened for the one camera.

    The fake forwards encode, probe, and version calls to the real
    ``subprocess.run`` so the canonical transcode still happens; only the
    instrument's ``-vf`` pass is counted. ``Any`` keeps ``ty`` happy on the
    forwarded call (the variadic ``object`` types ``ty`` rejects because
    they fail to match any of ``subprocess.run``'s overloads).
    """
    app = hflow.App("non-default-wrap", data_root=tmp_path / "data")

    @app.check()
    def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
        # Non-default parameter: a different freeze_min_duration_s builds a
        # different filter graph inside frame_stats.
        return hflow.checks.camera_frame_stats(ep, freeze_min_duration_s=5.0)

    decode_calls: list[int] = []
    real_run = _instrument.subprocess.run

    def fake_run(*arguments: Any, **keywords: Any) -> Any:
        command = arguments[0] if arguments else keywords.get("args", [])
        # The instrument pass is the one whose filter graph contains
        # ``blackframe`` (the camera-measuring graph). ``Episode.frames``
        # uses ``fps=0.5`` and contact-sheet uses ``scale=...tile=...``, so
        # checking for ``blackframe`` isolates the instrument pass.
        if (
            isinstance(command, (list, tuple))
            and "-vf" in command
            and any("blackframe" in str(arg) for arg in command)
        ):
            decode_calls.append(1)
        return real_run(*arguments, **keywords)

    monkeypatch.setattr(_instrument.subprocess, "run", fake_run)
    report = app.process(camera_source, stages="full", record=False)
    by_name = {run.check.name: run for run in report.checks}

    # The wrapper ran, the default did not, and exactly one ffmpeg decode
    # pass paid for the wrapper's measurement. The decode belongs to the
    # wrapper: with the fix, the default is short-circuited before any
    # ffmpeg work happens, so the only instrument pass is the one the
    # wrapper asked for.
    assert by_name["camera_health"].result is not None
    wrapper_keys = by_name["camera_health"].result.measurements
    assert any(key.endswith("/decoded_frame_count") for key in wrapper_keys), wrapper_keys
    assert len(decode_calls) == 1, f"got {decode_calls}"
    # The default is superseded, as today, with the same reason, and never
    # ran: duration_s is zero, not "ran in 0.4s and was then nulled out".
    default = by_name["camera_frame_stats"]
    assert default.status is hflow.CheckStatus.SUPERSEDED
    assert default.result is None
    assert isinstance(default.not_run, hflow.SupersededByPipeline)
    assert "default_checks" in default.not_run.reason
    assert default.duration_s == pytest.approx(0.0)


def test_a_wrapper_with_default_parameters_uses_the_post_execution_backstop(
    camera_source: Path, tmp_path: Path
) -> None:
    """The pre-execution short-circuit only fires when the wrapper's
    measurements actually overlap the default's predicted key set. A wrapper
    that calls ``camera_frame_stats`` with the default parameters emits the
    same key set, so the short-circuit does fire; the post-execution
    ``_yield_defaults_superseded_by_the_…`` path is the backstop for any
    case the pre-execution check can't see, and its outcome must agree.

    The wrapper's parameters here match the default exactly, so the
    predicted-set and the emitted-set are the same. Either path can win:
    what matters is that the default is superseded (not running) and the
    wrapper's measurements survive.
    """
    app = hflow.App("default-params", data_root=tmp_path / "data")

    @app.check()
    def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.camera_frame_stats(ep)

    report = app.test(camera_source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    assert by_name["camera_health"].result is not None
    default = by_name["camera_frame_stats"]
    assert default.status is hflow.CheckStatus.SUPERSEDED
    assert default.result is None
    assert isinstance(default.not_run, hflow.SupersededByPipeline)
    # The reason text is the same regardless of which path fired: the
    # message names the source ("default_checks") and the user can grep
    # for it.
    assert "default_checks" in default.not_run.reason


def test_a_non_overlapping_user_step_does_not_supersede(
    camera_source: Path, tmp_path: Path
) -> None:
    """The short-circuit must not be a catch-all. A user step that emits
    keys the default would not emit keeps the default running -- the
    default's measurements are not duplicates, the user's are
    complementary."""
    app = hflow.App("non-overlap", data_root=tmp_path / "data")

    @app.check()
    def unrelated(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"my/custom/key": 1.0})

    report = app.test(camera_source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    assert by_name["camera_frame_stats"].result is not None
    assert by_name["unrelated"].result is not None


def test_superseded_default_reports_the_overlapping_keys(
    camera_source: Path, tmp_path: Path
) -> None:
    """The post-execution backstop also records the overlapping key list;
    the pre-execution path must produce the same information so the
    planner, the catalog, and downstream consumers see one shape for
    "superseded by the pipeline" regardless of which path fired."""
    app = hflow.App("keys-listed", data_root=tmp_path / "data")

    @app.check()
    def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.camera_frame_stats(ep, freeze_min_duration_s=2.0)

    report = app.test(camera_source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    superseded = by_name["camera_frame_stats"].not_run
    assert isinstance(superseded, hflow.SupersededByPipeline)
    # At least one ``/decoded_frame_count``-shaped key survived, and the
    # default's per-camera expected/deficit and luma keys are all in the
    # overlapping list.
    assert len(superseded.superseded_keys) > 0
    assert any(key.endswith("/decoded_frame_count") for key in superseded.superseded_keys)


def test_partial_camera_coverage_drops_the_uncovered_camera_too(tmp_path: Path) -> None:
    """The default runs over every camera in the episode. A user step
    that wraps ``camera_frame_stats`` and only covers one of two
    cameras emits a strict subset of the default's keys. The
    pre-execution short-circuit sees any overlap and supersedes the
    whole default, so the still-uncovered camera's keys are also
    dropped. The post-execution backstop
    (``_yield_defaults_superseded_by_the_…``) has done the same since
    before #177: any non-empty intersection with the pipeline's
    emitted keys nulls the default's whole ``result``.

    Pinning it here so a future change to the pattern or the
    short-circuit cannot quietly flip this case. A user who wants the
    un-covered camera's keys to survive must either cover it in the
    wrapper or opt out of the default and supply their own full
    measurement. The plan and the PR body both call this out as the
    same edge case the pre-#177 code already had, not a new
    regression.
    """
    two_camera_source = synthesize_episode(
        tmp_path / "two_cams.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam", "overhead_cam")),
    )
    app = hflow.App("partial", data_root=tmp_path / "data")

    @app.check()
    def only_wrist(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.camera_frame_stats(ep)

    report = app.test(two_camera_source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    default = by_name["camera_frame_stats"]
    assert default.status is hflow.CheckStatus.SUPERSEDED
    assert default.result is None


def test_every_default_in_the_pattern_registry_is_drift_free(
    source_episode: Path, tmp_path: Path
) -> None:
    """The pre-execution short-circuit trusts ``_DEFAULT_KEY_PATTERNS`` to
    predict the keys each default will emit. If a future change to a
    default adds or removes a measurement without updating its pattern,
    the short-circuit will silently misfire: it will either skip a
    default whose measurements the pipeline did not actually replace
    (a false positive), or it will run a default whose measurements the
    pipeline did replace (a false negative, the bug we are fixing).

    This test runs each default on a synthetic episode, collects what
    it actually emitted, and asserts the pattern is exact. A drift here
    fails the build rather than waiting for a missed supersession in
    production.
    """
    app = hflow.App("drift-guard", data_root=tmp_path / "data")
    report = app.test(source_episode, verbose=False)
    actual = {
        run.check.name: (set(run.result.measurements) if run.result is not None else set())
        for run in report.checks
    }
    # Replay each pattern against the same canonical episode the
    # defaults saw, so the predicted and emitted sets are over the
    # same input. The workspace is what ``app.test`` used to build the
    # canonical form, and the patterns only read from the episode
    # handle's structural state (channel counts, declared cameras) that
    # does not vary between the test path and the short-circuit's.
    from hflow.episode import Episode

    canonical = Episode(source_episode)
    for default, pattern in _DEFAULT_KEY_PATTERNS.items():
        predicted = pattern(canonical)
        # ``CheckFunction`` is ``Callable`` in the type system, so the
        # function object has no public ``__name__``; the check name
        # arrives via the registered step's ``name`` and matches
        # ``default.__name__`` at registration time. Look it up the
        # same way the runner would.
        emitted_name = next(
            (run.check.name for run in report.checks if run.check.function is default),
            None,
        )
        assert emitted_name is not None
        emitted = actual.get(emitted_name, set())
        assert predicted == emitted, (
            f"drift in {emitted_name}: pattern predicts {sorted(predicted)} "
            f"but the default emitted {sorted(emitted)}"
        )
