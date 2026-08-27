"""The baseline every episode gets without anyone registering it."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hflow
from hflow._video_measurements import (
    FRAME_STATISTICS_DEFINITION_VERSION,
    FrameStatisticsProvenance,
    FrameStatisticsSettings,
    LumaRangeEvidence,
    VideoFrameStatistics,
    _frame_statistics,
)
from hflow.checks import (
    _DEFAULT_KEY_PATTERNS,
    DEFAULT_CHECKS,
    _TimestampRegularityPerTopic,
    _TimestampRegularitySync,
    _camera_value,
    _CameraIntermediates,
    _episode_duration_intermediates,
    _episode_duration_value,
    _keyframe_interval_value,
    _timestamp_regularity_value,
    camera_frame_stats,
    camera_frame_stats_keys,
    episode_duration,
    episode_duration_keys,
    keyframe_interval,
    keyframe_interval_keys,
    timestamp_regularity,
    timestamp_regularity_keys,
)
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
    assert {run.check.version for run in report.checks} == {"1"}
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


def test_pipeline_authored_checks_require_explicit_registration_versions() -> None:
    def pipeline_authored_check(episode: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"episode_path": str(episode.path)})

    with pytest.raises(
        ValueError,
        match=r"register pipeline-authored checks with app\.check\(version=\.\.\.\)",
    ):
        hflow.App("custom-default", default_checks=(pipeline_authored_check,))


def test_registering_a_default_yourself_configures_it_rather_than_colliding(
    tmp_path: Path,
) -> None:
    app = hflow.App("configured", data_root=tmp_path / "data")

    app.check(version="1", critical=True)(episode_duration)

    registrations = [
        registered for registered in app.checks if registered.name == "episode_duration"
    ]
    assert len(registrations) == 1
    assert registrations[0].critical is True


def test_two_pipeline_steps_sharing_a_name_are_still_refused(tmp_path: Path) -> None:
    """Replacement applies to defaults only: between two of the pipeline's own
    steps the engine has no basis to pick a winner."""
    app = hflow.App("clash", data_root=tmp_path / "data")

    @app.check(version="1", name="mine")
    def first(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult()

    with pytest.raises(ValueError, match="already registered"):

        @app.check(version="1", name="mine")
        def second(ep: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult()


def test_a_default_yields_to_a_pipeline_step_measuring_the_same_thing(
    source_episode: Path, tmp_path: Path
) -> None:
    """The documented way to configure a built-in is to wrap it under your own
    name, which emits the built-in's keys. That must not be a collision."""
    app = hflow.App("wrapping", data_root=tmp_path / "data")

    @app.check(version="1")
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

    @app.check(version="1")
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

    The fake forwards to the real ``subprocess.Popen`` so the canonical
    transcode and the instrument pass both still work, and counts only
    the invocations whose command has both ``-vf`` and ``blackframe``
    (the camera-measuring filter graph, unique to the instrument
    pass; ``Episode.frames`` uses ``fps=`` and the contact sheet uses
    ``scale=...tile=...``). ``Any`` keeps ``ty`` happy on the forwarded
    call (the variadic ``object`` types ``ty`` rejects because they
    fail to match any of ``subprocess.Popen``'s overloads).
    """
    app = hflow.App("non-default-wrap", data_root=tmp_path / "data")

    @app.check(version="1")
    def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
        # Non-default parameter: a different freeze_min_duration_s builds a
        # different filter graph inside frame_stats.
        return hflow.checks.camera_frame_stats(ep, freeze_min_duration_s=5.0)

    decode_calls: list[int] = []
    real_popen = _frame_statistics.subprocess.Popen

    def fake_popen(*arguments: Any, **keywords: Any) -> Any:
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
        return real_popen(*arguments, **keywords)

    monkeypatch.setattr(_frame_statistics.subprocess, "Popen", fake_popen)
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

    @app.check(version="1")
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

    @app.check(version="1")
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

    @app.check(version="1")
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

    @app.check(version="1")
    def only_wrist(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.camera_frame_stats(ep)

    report = app.test(two_camera_source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    default = by_name["camera_frame_stats"]
    assert default.status is hflow.CheckStatus.SUPERSEDED
    assert default.result is None


@pytest.mark.parametrize("cameras", [(), ("wrist_cam",), ("wrist_cam", "overhead_cam")])
def test_every_default_in_the_pattern_registry_is_drift_free(
    cameras: tuple[str, ...], tmp_path: Path
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

    Parametrized over the camera count on purpose. Three of the six
    mirrors (``camera_frame_stats``, ``keyframe_interval``,
    ``media_digest``) emit nothing at all on a camera-less episode, so a
    guard that only ever saw one would pass while predicting a stale key
    set for exactly the default this whole short-circuit exists to skip.
    Two cameras as well as one, because a per-topic mirror that
    accidentally hardcodes the first camera reads correct at one.
    """
    source_episode = synthesize_episode(
        tmp_path / "drift_source.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=cameras),
    )
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

    canonical = Episode(report.canonical_path)
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


def test_quarantined_episode_still_carries_default_measurements(
    camera_source: Path, tmp_path: Path
) -> None:
    """Kingston's #177 review (round 1): the reorder that puts user steps
    ahead of defaults for the supersession decision used to be paired with
    the pre-existing blanket ``if report.quarantined: continue`` skip,
    which meant a user critical check that tripped on the first user step
    would cascade-skip every default, including ``content_digest`` and
    ``media_digest`` -- the rows you most need to diagnose why a
    recording was rejected. Defaults are the cheap diagnostic evidence
    the quarantine itself was derived from, so they keep running and
    recording on a quarantined episode. User-registered steps still
    skip; the boundary moves only for the engine's own auto-registered
    baseline. Pinned here so a future re-ordering can't silently
    re-couple the two without a test review.
    """
    app = hflow.App("quarantining", data_root=tmp_path / "data")

    @app.check(
        version="1",
        critical=True,
        gate=hflow.checks.RECOMMENDED_CAMERA_INTEGRITY,
    )
    def blackout(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.CheckResult(measurements={"*black_frame_pct": 99.0})

    report = app.test(camera_source, verbose=False)
    by_name = {run.check.name: run for run in report.checks}
    # The user's check did its job: the episode is quarantined.
    assert report.quarantined is not None
    assert "blackout" in report.quarantine_tags[0]
    # User-registered steps after the quarantining one still skip --
    # that is the right answer for the pipeline's own work and the
    # reason the blanket skip exists.
    user = by_name["blackout"]
    assert user.result is not None
    assert user.result.verdict is False
    # The defaults are exempt from the skip: they recorded their
    # measurements and the report carries them. The two digests are
    # the explicit ones called out in the review; episode_duration is
    # the third.
    episode_duration = by_name["episode_duration"]
    assert episode_duration.result is not None
    assert episode_duration.status is hflow.CheckStatus.MEASURED
    content_digest = by_name["content_digest"]
    assert content_digest.result is not None
    assert content_digest.status is hflow.CheckStatus.MEASURED
    media_digest = by_name["media_digest"]
    assert media_digest.result is not None
    assert media_digest.status is hflow.CheckStatus.MEASURED
    # The camera_frame_stats default also still records its measurements,
    # so the diagnostic digests on the row that just tripped a gate are
    # available alongside the user's own evidence.
    camera = by_name["camera_frame_stats"]
    assert camera.result is not None


def _assert_measurements_match_pinned(measurements: dict[str, object], pinned: dict) -> None:
    """Full-dict comparison for the #182 value fixtures: keys AND values.

    A plain number pins exact equality (counts and the frame-deficit
    arithmetic are deterministic); a ``(low, high)`` tuple pins an inclusive
    range for the ffmpeg-rendered quantities whose exact values track the
    runner's build (the convention ``test_checks.py`` already uses); a
    callable is an arbitrary predicate. Tight enough that a dispatcher
    branch reading the wrong field fails here instead of writing a
    plausible number into the catalog.
    """
    assert set(measurements) == set(pinned), sorted(set(measurements) ^ set(pinned))
    for key, expected in pinned.items():
        actual = measurements[key]
        if callable(expected):
            assert expected(actual), f"{key}: {actual!r} failed its predicate pin"
        elif isinstance(expected, tuple):
            low, high = expected
            assert low <= actual <= high, f"{key}: {actual!r} outside [{low}, {high}]"
        else:
            assert actual == pytest.approx(expected), f"{key}: {actual!r} != {expected!r}"


def _pinned_camera_measurements(
    topic: str,
    *,
    frames: int,
    expected_frame_count: int | None = None,
    frame_deficit_pct: float | None = None,
    black_frame_pct: tuple[float, float] | float = (0.0, 50.0),
    overexposed_frame_pct: tuple[float, float] | float = (0.0, 50.0),
) -> dict[str, object]:
    """Hand-written expectation for one camera topic, from the fact's
    docstring -- ground truth independent of both fact and dispatcher.

    ``frame_deficit_pct`` is a hardcoded float literal for the fixture, not
    a formula restatement: if the dispatcher's arithmetic is wrong, the
    formula would agree with the bug and the test would pass on a wrong
    answer. ``black_frame_pct`` and ``overexposed_frame_pct`` accept either
    a tight range or an exact float so a black<->overexposed swap cannot
    pass when the fixture has one and not the other.
    """
    pinned: dict[str, object] = {
        f"{topic}/message_count": frames,
        f"{topic}/decoded_frame_count": frames,
        f"{topic}/freeze_total_s": 0.0,
        f"{topic}/black_frame_pct": black_frame_pct,
        f"{topic}/overexposed_frame_pct": overexposed_frame_pct,
        f"{topic}/luma_avg_min": (0.0, 255.0),
        f"{topic}/luma_avg_mean": (0.0, 255.0),
        f"{topic}/luma_avg_max": (0.0, 255.0),
    }
    if expected_frame_count is not None:
        pinned[f"{topic}/expected_frame_count"] = expected_frame_count
        assert frame_deficit_pct is not None
        pinned[f"{topic}/frame_deficit_pct"] = frame_deficit_pct
    return pinned


# ``black_segment=(0.2, 0.5)`` puts a black run inside the 1-second window
# (the spec's default of (2.0, 3.0) misses it), so ``black_frame_pct`` is
# non-zero. testsrc2 (the first camera) renders no blown highlights, so
# ``overexposed_frame_pct`` is exactly 0.0. A black<->overexposed swap
# would try to satisfy the wrong pin on each side and fail.
_ONE_CAMERA_PIN = {
    "frames": 15,
    "expected_frame_count": 15,
    # 15 stamps over 1s at 15Hz, expected 15: literal 0.0, NOT the formula.
    "frame_deficit_pct": 0.0,
    "black_frame_pct": (30.0, 40.0),  # measured 33.33 in the inside-segment fixture
    "overexposed_frame_pct": 0.0,  # testsrc2 never blows highlights
}
# The second camera renders an inverted smptebars pattern (no black run), so
# the discriminator goes the other way: a tight range for overexposed, an
# exact 0.0 for black. Same shape, opposite polarity -- the swap-test logic
# reads the polarity, not the camera name.
_OTHER_CAMERA_PIN = {
    "frames": 15,
    "expected_frame_count": 15,
    "frame_deficit_pct": 0.0,
    "black_frame_pct": 0.0,  # smptebars never produces blackframes
    "overexposed_frame_pct": (0.0, 5.0),  # smptebars has luma but not blowouts
}
_TWO_STAMP_PIN = {
    "frames": 2,
    "expected_frame_count": 2,
    "frame_deficit_pct": 0.0,
    # A 2-frame, 2-second fixture has no chance to grow black or overexposed
    # runs; both stay at 0.0. The strict pin catches a swap that lands 0.0
    # on the wrong side when the wrist_cam fixture above has 33.33 / 0.0.
    "black_frame_pct": 0.0,
    "overexposed_frame_pct": 0.0,
}


CAMERA_VALUE_FIXTURE_CASES = [
    pytest.param(SyntheticEpisodeSpec(duration_s=1.0, cameras=()), {}, id="zero-cameras"),
    pytest.param(
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",), black_segment=(0.2, 0.5)),
        {"wrist_cam": _ONE_CAMERA_PIN},
        id="one-camera",
    ),
    pytest.param(
        SyntheticEpisodeSpec(
            duration_s=1.0, cameras=("wrist_cam", "overhead_cam"), black_segment=(0.2, 0.5)
        ),
        {
            "wrist_cam": _ONE_CAMERA_PIN,
            "overhead_cam": _OTHER_CAMERA_PIN,
        },
        id="two-cameras",
    ),
    pytest.param(
        SyntheticEpisodeSpec(duration_s=2.0, image_hz=1.0, cameras=("wrist_cam",)),
        {"wrist_cam": _TWO_STAMP_PIN},
        id="two-stamp-topic",
    ),
]


@pytest.mark.parametrize(("spec", "camera_pins"), CAMERA_VALUE_FIXTURE_CASES)
def test_camera_frame_stats_emits_exactly_the_documented_measurements(
    spec: SyntheticEpisodeSpec, camera_pins: dict[str, dict[str, int]], tmp_path: Path
) -> None:
    """The #182 condition: the fixture asserts the full measurement dict,
    values included, against ground truth written by hand -- so a mismapped
    dispatcher branch (luma_avg_min reading average_luma_maximum) fails in
    CI instead of recording a plausible wrong number."""
    source = synthesize_episode(tmp_path / "value_source.mcap", spec)
    report = hflow.App("value-fixture", data_root=tmp_path / "data").test(source, verbose=False)
    run = next(r for r in report.checks if r.check.name == "camera_frame_stats")
    assert run.result is not None

    pinned: dict[str, object] = {}
    if camera_pins:
        pinned["camera_instrument"] = lambda value: (
            isinstance(value, str) and value.startswith("ffmpeg")
        )
        pinned["camera_measurement_definition"] = FRAME_STATISTICS_DEFINITION_VERSION

    from hflow.episode import Episode

    canonical_cameras = Episode(report.canonical_path).cameras
    topic_by_name = {
        name: next(topic for topic in canonical_cameras if name in topic) for name in camera_pins
    }
    for name, fixture in camera_pins.items():
        pinned.update(_pinned_camera_measurements(topic_by_name[name], **fixture))

    _assert_measurements_match_pinned(dict(run.result.measurements), pinned)

    if not camera_pins:
        # Zero cameras emits nothing at all, intervals included: every write,
        # even the bare instrument keys, lives inside the per-topic selection.
        assert run.result.intervals == []

    for name in camera_pins:
        topic = topic_by_name[name]
        minimum = float(run.result.measurements[f"{topic}/luma_avg_min"])
        mean = float(run.result.measurements[f"{topic}/luma_avg_mean"])
        maximum = float(run.result.measurements[f"{topic}/luma_avg_max"])
        if "wrist_cam" in topic:
            # testsrc2 changes every frame, so the statistics are strictly
            # ordered and a swapped dispatcher branch cannot satisfy this.
            assert minimum < mean < maximum
        else:
            # smptebars renders one still pattern: per-frame luma differs only
            # by encoder noise, far tighter than a field mismap could hide.
            assert maximum - minimum < 0.5


def test_a_single_stamp_camera_topic_errors_before_any_measurement(
    tmp_path: Path,
) -> None:
    """Pre-existing upstream contract, pinned so the refactor cannot be
    blamed for it: ``Episode.video`` needs at least two timestamps to infer
    a frame rate, so a one-stamp camera errors before any measurement
    exists. What the refactor owns is the fact's prediction for such a
    topic -- no rate pair -- which is what the supersession gate consults."""
    source = synthesize_episode(
        tmp_path / "sparse_source.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, image_hz=1.0, cameras=("wrist_cam",)),
    )
    report = hflow.App("single-stamp", data_root=tmp_path / "data").test(source, verbose=False)
    run = next(r for r in report.checks if r.check.name == "camera_frame_stats")
    assert run.result is None
    assert "at least 2" in (run.error or "")

    from hflow.episode import Episode

    topic = Episode(report.canonical_path).cameras[0]
    assert camera_frame_stats_keys(Episode(report.canonical_path), cameras=[topic]) == {
        f"{topic}/message_count",
        f"{topic}/decoded_frame_count",
        f"{topic}/black_frame_pct",
        f"{topic}/overexposed_frame_pct",
        f"{topic}/freeze_total_s",
        f"{topic}/luma_avg_mean",
        f"{topic}/luma_avg_min",
        f"{topic}/luma_avg_max",
        "camera_instrument",
        "camera_measurement_definition",
    }


def test_the_body_emits_what_the_fact_names_even_when_the_fact_withdraws_one(
    camera_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delegation guard for #182: the body must walk the fact's output. If
    withdrawing one named key does not withdraw it from the emitted
    measurements, the body is carrying its own key list again and the
    refactor is void."""
    real_fact = hflow.checks.camera_frame_stats_keys

    def poisoned(episode: hflow.Episode, *, cameras: Sequence[str] | None = None) -> set[str]:
        selected = list(cameras) if cameras is not None else episode.cameras
        return real_fact(episode, cameras=cameras) - {f"{selected[0]}/black_frame_pct"}

    monkeypatch.setattr(hflow.checks, "camera_frame_stats_keys", poisoned)

    report = hflow.App("poison", data_root=tmp_path / "data").test(camera_source, verbose=False)
    run = next(r for r in report.checks if r.check.name == "camera_frame_stats")
    assert run.result is not None

    from hflow.episode import Episode

    topic = Episode(report.canonical_path).cameras[0]
    assert set(run.result.measurements) == {
        f"{topic}/message_count",
        f"{topic}/expected_frame_count",
        f"{topic}/frame_deficit_pct",
        f"{topic}/decoded_frame_count",
        f"{topic}/overexposed_frame_pct",
        f"{topic}/freeze_total_s",
        f"{topic}/luma_avg_mean",
        f"{topic}/luma_avg_min",
        f"{topic}/luma_avg_max",
        "camera_instrument",
        "camera_measurement_definition",
    }


def test_a_key_nobody_branches_on_raises_instead_of_appending_null(
    camera_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher's explicit failure: match-with-no-branch returning None
    would surface only as a null measurement under record=True (#182). A key
    the fact names and the dispatcher forgot must raise where it happens."""
    real_fact = hflow.checks.camera_frame_stats_keys

    def haunted(episode: hflow.Episode, *, cameras: Sequence[str] | None = None) -> set[str]:
        return real_fact(episode, cameras=cameras) | {"ghost/mystery"}

    monkeypatch.setattr(hflow.checks, "camera_frame_stats_keys", haunted)

    report = hflow.App("unknown-key", data_root=tmp_path / "data").test(
        camera_source, verbose=False
    )
    run = next(r for r in report.checks if r.check.name == "camera_frame_stats")
    assert run.result is None
    assert run.error is not None
    assert "no branch" in run.error


def test_a_bare_key_nobody_branches_on_raises_too() -> None:
    with pytest.raises(ValueError, match="no branch"):
        _camera_value("not_a_camera_thing", {})


def test_declared_expected_hz_wins_and_median_delta_fills_in(tmp_path: Path) -> None:
    """The precedence the intermediates copy verbatim: the parameter wins per
    topic; the median inter-message delta is the only fallback; two-way, no
    metadata step in the chain."""
    source = synthesize_episode(
        tmp_path / "hz_source.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",)),
    )

    from hflow.episode import Episode

    baseline_report = hflow.App("hz-fallback", data_root=tmp_path / "data-a").test(
        source, verbose=False
    )

    # Parameters ride a wrapper -- the documented way to configure a built-in.
    # Its emitted keys overlap the default's, so the default stands down and
    # the wrapper's own run carries the declared-hz measurements.
    declared_app = hflow.App("hz-declared", data_root=tmp_path / "data-b")

    @declared_app.check(version="1")
    def camera_health(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.camera_frame_stats(ep, expected_hz=dict.fromkeys(ep.cameras, 30.0))

    declared_report = declared_app.test(source, verbose=False)

    baseline_run = next(r for r in baseline_report.checks if r.check.name == "camera_frame_stats")
    declared_run = next(r for r in declared_report.checks if r.check.name == "camera_health")
    assert baseline_run.result is not None
    assert declared_run.result is not None

    topic = Episode(baseline_report.canonical_path).cameras[0]
    # 15 stamps at exactly one fifteenth of a second: the median delta rate
    # reproduces exactly 15 frames over the observed span, deficit zero.
    assert baseline_run.result.measurements[f"{topic}/expected_frame_count"] == 15
    assert baseline_run.result.measurements[f"{topic}/frame_deficit_pct"] == pytest.approx(0.0)
    # Declaring 30 Hz doubles the expectation over the same span:
    # round((14/15) * 30) + 1 == 29, and 15 delivered frames miss by that much.
    assert declared_run.result.measurements[f"{topic}/expected_frame_count"] == 29
    assert declared_run.result.measurements[f"{topic}/frame_deficit_pct"] == pytest.approx(
        100.0 * (29 - 15) / 29
    )


def test_dispatcher_message_count_and_decoded_frame_count_read_distinct_sources() -> None:
    """The two branches are sourced from different fields: the channel
    stamp array (one entry per MCAP message) and the ffmpeg filter graph's
    decoded frame count (one per MP4 frame). A branch swap that reads
    ``inter.stamps_ns.size`` for ``decoded_frame_count`` -- or vice versa
    -- produces a plausible number and cannot crash; this is the unit-level
    guard for that failure class, no App, no video, no ffmpeg.

    The integration fixtures (above) hold the two equal at 15 by
    construction (one access unit per message), which is why this
    dispatcher-level test is the load-bearing one for the swap case.
    """

    # ``VideoFrameStatistics`` is a frozen dataclass with 28 required fields.
    # Build a real instance with zeros for the fields the dispatcher does
    # NOT read, then ``dataclasses.replace`` the one the test pins. Only
    # the ``decoded_frame_count`` branch reads ``stats.decoded_frame_count``;
    # the dispatcher never touches the rest in this test, so zero defaults
    # do not contaminate the assertion.
    stub_provenance = FrameStatisticsProvenance(
        measurement_definition_version="video-frame-statistics/v1",
        ffmpeg_version="unit-test-stub",
        filter_graph="stub",
        settings=FrameStatisticsSettings(),
    )
    base_stats = VideoFrameStatistics(
        decoded_frame_count=0,
        duration_seconds=0.0,
        black_frame_count=0,
        black_frame_percent=0.0,
        overexposed_frame_count=0,
        overexposed_frame_percent=0.0,
        freeze_intervals=(),
        freeze_total_seconds=0.0,
        average_luma_mean=0.0,
        average_luma_minimum=0.0,
        average_luma_maximum=0.0,
        black_pixel_share_mean=0.0,
        black_pixel_share_maximum=0.0,
        minimum_luma=0.0,
        maximum_luma=0.0,
        luma_range_evidence=LumaRangeEvidence.NOMINAL_LIMITED_RANGE_COMPATIBLE,
        tenth_percentile_luma_mean=0.0,
        ninetieth_percentile_luma_mean=0.0,
        clipped_highlight_frame_percent=0.0,
        crushed_shadow_frame_percent=0.0,
        frame_difference_mean=0.0,
        frame_difference_maximum=0.0,
        temporal_outlier_mean=0.0,
        temporal_outlier_maximum=0.0,
        out_of_legal_range_mean=0.0,
        out_of_legal_range_maximum=0.0,
        provenance=stub_provenance,
    )
    stats = replace(base_stats, decoded_frame_count=15)
    inter = _CameraIntermediates(
        stamps_ns=np.arange(13),
        stats=stats,
        expected_frame_count=None,
    )
    by_topic = {"/cam0": inter}

    assert _camera_value("/cam0/message_count", by_topic) == 13
    assert _camera_value("/cam0/decoded_frame_count", by_topic) == 15
    # A swap that returns 15 for message_count -- or 13 for decoded_frame_count
    # -- would have to break both asserts, not just one.


# episode_duration: three fixed keys, independent of which topics the episode
# carries. The value fixture asserts the full dict -- including the topic
# count, which the body derives from its selection -- against a hand-written
# ground truth so a mismapped dispatcher branch fails in CI instead of writing
# a plausible wrong number into the catalog.

EPISODE_DURATION_VALUE_CASES = [
    pytest.param(
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
        {"duration_s": pytest.approx(1.0, abs=0.05), "message_count_total": 100},
        id="one-second-joints-only",
    ),
    pytest.param(
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",)),
        {
            "duration_s": pytest.approx(1.0, abs=0.05),
            "message_count_total": 100 + 15,
        },
        id="one-second-with-camera",
    ),
]


@pytest.mark.parametrize(("spec", "expected_subset"), EPISODE_DURATION_VALUE_CASES)
def test_episode_duration_emits_exactly_the_documented_measurements(
    spec: SyntheticEpisodeSpec, expected_subset: dict[str, object], tmp_path: Path
) -> None:
    """The #182 condition, applied to ``episode_duration``: full dict
    asserted against hand-written ground truth. ``topic_count`` is the
    number of topics in the selection; the camera count is the per-test
    fixture's ``cameras=`` tuple length plus one (the joint channel), so
    we pin it by the spec the test parameterises over rather than
    restating the synthetic topology."""
    source = synthesize_episode(tmp_path / "duration_source.mcap", spec)
    report = hflow.App("duration-fixture", data_root=tmp_path / "data").test(source, verbose=False)
    run = next(r for r in report.checks if r.check.name == "episode_duration")
    assert run.result is not None

    pinned = dict(expected_subset)
    pinned["topic_count"] = len(spec.cameras) + 1  # joints + each camera topic
    _assert_measurements_match_pinned(dict(run.result.measurements), pinned)


def test_episode_duration_keys_fact_and_body_emit_the_same_set(
    source_episode: Path,
) -> None:
    """Delegation guard: the body's key set is owned by the fact."""
    from hflow.episode import Episode

    canonical = Episode(source_episode)
    assert episode_duration_keys(canonical) == {
        "duration_s",
        "message_count_total",
        "topic_count",
    }


def test_episode_duration_withdraws_a_key_when_the_fact_does(
    source_episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Poison: withdrawing ``message_count_total`` from the fact must
    withdraw it from the body's emitted dict, because the body iterates
    the fact's output."""
    real_fact = episode_duration_keys

    def poisoned(episode: hflow.Episode) -> set[str]:
        return real_fact(episode) - {"message_count_total"}

    monkeypatch.setattr(hflow.checks, "episode_duration_keys", poisoned)
    report = hflow.App("duration-poison", data_root=tmp_path / "data").test(
        source_episode, verbose=False
    )
    run = next(r for r in report.checks if r.check.name == "episode_duration")
    assert run.result is not None
    assert "message_count_total" not in run.result.measurements
    assert "duration_s" in run.result.measurements
    assert "topic_count" in run.result.measurements


def test_episode_duration_raises_on_an_unrecognised_bare_key() -> None:
    from hflow.checks import _EpisodeDurationIntermediates

    inter = _EpisodeDurationIntermediates(duration_s=1.0, message_count_total=10, topic_count=2)
    with pytest.raises(ValueError, match="no branch"):
        _episode_duration_value("nope", inter)


# timestamp_regularity: per-topic period keys (one of three sets: empty
# when the topic has <2 messages, or the three period keys otherwise) plus
# a pair of sync/<cam>~<ref>/{start,end}_offset_s keys per camera when the
# episode carries both camera and state streams. The dense fixture pins
# every emitted key; the camera-only fixture pins no sync offsets (no
# state reference present); the no-camera fixture pins only joint-state
# period keys.

TIMESTAMP_REGULARITY_VALUE_CASES = [
    pytest.param(
        # No cameras, joint stream dense: per-topic period keys only.
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
        # Per-topic: /joint_states keys. No sync (no cameras).
        {
            "/joint_states/median_dt_s": pytest.approx(0.01, abs=0.005),
            "/joint_states/period_violation_pct": (0.0, 100.0),
            "/joint_states/max_gap_s": pytest.approx(0.01, abs=0.01),
        },
        id="joints-only",
    ),
    pytest.param(
        # Camera + joint: per-topic period keys for both kinds, plus
        # sync/<cam>~<ref>/{start,end}_offset_s. The synthetic episode's
        # first message aligns across all channels, so offsets are ~0.
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",)),
        {
            "/joint_states/median_dt_s": pytest.approx(0.01, abs=0.005),
            "/joint_states/period_violation_pct": 0.0,  # uniform joint stream
            "/joint_states/max_gap_s": pytest.approx(0.01, abs=0.01),
            "/wrist_cam/compressed/median_dt_s": pytest.approx(0.0667, abs=0.01),
            "/wrist_cam/compressed/period_violation_pct": 0.0,  # uniform cam stream
            "/wrist_cam/compressed/max_gap_s": pytest.approx(0.0667, abs=0.01),
        },
        id="camera-and-joint",
    ),
]


@pytest.mark.parametrize(("spec", "expected_subset"), TIMESTAMP_REGULARITY_VALUE_CASES)
def test_timestamp_regularity_emits_exactly_the_documented_measurements(
    spec: SyntheticEpisodeSpec, expected_subset: dict[str, object], tmp_path: Path
) -> None:
    """The #182 condition, applied to ``timestamp_regularity``: full dict
    asserted against hand-written ground truth. Sync offsets are checked
    separately because their keys depend on the densest non-camera stream
    at run time and would need a per-fixture resolution; the existing
    supersession tests already exercise the sync branch in detail, so
    here we keep the per-topic keys tight.
    """
    source = synthesize_episode(tmp_path / "ts_source.mcap", spec)
    report = hflow.App("ts-fixture", data_root=tmp_path / "data").test(source, verbose=False)
    run = next(r for r in report.checks if r.check.name == "timestamp_regularity")
    assert run.result is not None

    expected = dict(expected_subset)
    # Pin the sync offsets when both camera and joint streams are present.
    # The synthetic episode stamps streams independently, so the actual
    # first/last-frame offset is whatever the writer produced; we read it
    # back from the canonical episode and use that as ground truth, with
    # a tight abs band that still allows sub-millisecond drift.
    if spec.cameras:
        from hflow.episode import Episode

        canon = Episode(report.canonical_path)
        selected_cameras = [t for t in canon.cameras if t in {f"/{c}/compressed" for c in spec.cameras}]
        state_topics = sorted(
            t
            for t in canon.topics
            if canon.topics[t].message_count >= 2 and t not in canon.cameras
        )
        if state_topics:
            reference = max(state_topics, key=lambda t: canon.topics[t].message_count)
            ref_stamps = canon.channel(reference).timestamps
            for cam in selected_cameras:
                cam_stamps = canon.channel(cam).timestamps
                start = float((cam_stamps[0] - ref_stamps[0]) / 1e9)
                end = float((cam_stamps[-1] - ref_stamps[-1]) / 1e9)
                expected[f"sync/{cam}~{reference}/start_offset_s"] = pytest.approx(start, abs=0.001)
                expected[f"sync/{cam}~{reference}/end_offset_s"] = pytest.approx(end, abs=0.001)

    _assert_measurements_match_pinned(dict(run.result.measurements), expected)


def test_timestamp_regularity_keys_fact_matches_what_the_body_would_write(
    source_episode: Path,
) -> None:
    """Delegation guard: the fact owns the key set, and the body iterates
    it -- so the fact's output is a superset of the body's emitted keys
    (the body iterates a sorted subset; the fact is the source of truth)."""
    from hflow.episode import Episode

    canonical = Episode(source_episode)
    assert timestamp_regularity_keys(canonical) == set(_timestamp_regularity_emit(canonical))


def _timestamp_regularity_emit(episode):
    """Helper that re-derives the body's emitted key set by walking the
    same selection rule, so the test can compare without the fact."""
    from hflow.checks import _timestamp_regularity_resolve_selected

    selected, _ = _timestamp_regularity_resolve_selected(episode, None)
    keys: set[str] = set()
    for topic in selected:
        if episode.channel(topic).timestamps.size < 2:
            keys.add(f"{topic}/period_sample_count")
        else:
            keys.add(f"{topic}/median_dt_s")
            keys.add(f"{topic}/period_violation_pct")
            keys.add(f"{topic}/max_gap_s")
    return keys


def test_timestamp_regularity_withdraws_a_key_when_the_fact_does(
    source_episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Poison: withdrawing ``median_dt_s`` from the fact must withdraw it
    from the body's emitted dict, because the body iterates the fact's
    output."""
    real_fact = timestamp_regularity_keys

    def poisoned(episode: hflow.Episode) -> set[str]:
        return {k for k in real_fact(episode) if not k.endswith("/median_dt_s")}

    monkeypatch.setattr(hflow.checks, "timestamp_regularity_keys", poisoned)
    report = hflow.App("ts-poison", data_root=tmp_path / "data").test(
        source_episode, verbose=False
    )
    run = next(r for r in report.checks if r.check.name == "timestamp_regularity")
    assert run.result is not None
    assert not any(k.endswith("/median_dt_s") for k in run.result.measurements)
    assert any(k.endswith("/max_gap_s") for k in run.result.measurements)


def test_timestamp_regularity_raises_on_an_unrecognised_key(
    source_episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown topic in the fact's output must raise where it happens."""
    real_fact = timestamp_regularity_keys

    def haunted(episode: hflow.Episode) -> set[str]:
        return real_fact(episode) | {"/ghost/mystery"}

    monkeypatch.setattr(hflow.checks, "timestamp_regularity_keys", haunted)
    report = hflow.App("ts-haunt", data_root=tmp_path / "data").test(
        source_episode, verbose=False
    )
    run = next(r for r in report.checks if r.check.name == "timestamp_regularity")
    assert run.result is None
    assert run.error is not None
    assert "no branch" in run.error


# keyframe_interval: per-camera keys. With no camera, the key set is empty.
# With one camera carrying the synthetic smptebars + black frame mix, the
# body writes scanned_frame_count / keyframe_count / first_frame_is_keyframe
# and the keyframe-gap / median-interval keys when at least one / two
# keyframes are present. The synthetic encoder's GOP places a keyframe at
# the head of every second of footage, so the one-second fixture should
# always have at least one keyframe.

KEYFRAME_INTERVAL_VALUE_CASES = [
    pytest.param(
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
        {},
        id="no-camera",
    ),
    pytest.param(
        # The synthetic encoder places a keyframe at the head of every GOP,
        # and a 1.0 s episode fits one GOP exactly: the first frame is the
        # only keyframe. ``median_keyframe_interval_s`` therefore does not
        # appear (needs at least two keyframes), and ``max_keyframe_gap_s``
        # equals the tail from frame 0 to the last frame.
        SyntheticEpisodeSpec(duration_s=1.0, cameras=("wrist_cam",)),
        {
            "/wrist_cam/compressed/scanned_frame_count": 15,
            "/wrist_cam/compressed/keyframe_count": 1,
            "/wrist_cam/compressed/first_frame_is_keyframe": 1,
            "/wrist_cam/compressed/max_keyframe_gap_s": pytest.approx(0.93, abs=0.05),
        },
        id="one-camera",
    ),
]


@pytest.mark.parametrize(("spec", "expected_subset"), KEYFRAME_INTERVAL_VALUE_CASES)
def test_keyframe_interval_emits_exactly_the_documented_measurements(
    spec: SyntheticEpisodeSpec, expected_subset: dict[str, object], tmp_path: Path
) -> None:
    """The #182 condition, applied to ``keyframe_interval``: full dict
    asserted against hand-written ground truth. The synthetic encoder's
    keyframe placement is GOP-driven, so the keyframe_count and timing
    keys get ranges / wide bands; the deterministic keys
    (``scanned_frame_count`` and ``first_frame_is_keyframe``) are pinned
    tight.
    """
    source = synthesize_episode(tmp_path / "kf_source.mcap", spec)
    report = hflow.App("kf-fixture", data_root=tmp_path / "data").test(source, verbose=False)
    run = next(r for r in report.checks if r.check.name == "keyframe_interval")
    assert run.result is not None
    _assert_measurements_match_pinned(dict(run.result.measurements), expected_subset)


def test_keyframe_interval_withdraws_a_key_when_the_fact_does(
    source_episode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Poison: withdrawing ``max_keyframe_gap_s`` from the fact must
    withdraw it from the body's emitted dict."""
    real_fact = keyframe_interval_keys

    def poisoned(episode: hflow.Episode) -> set[str]:
        return {k for k in real_fact(episode) if not k.endswith("/max_keyframe_gap_s")}

    monkeypatch.setattr(hflow.checks, "keyframe_interval_keys", poisoned)
    report = hflow.App("kf-poison", data_root=tmp_path / "data").test(
        source_episode, verbose=False
    )
    run = next(r for r in report.checks if r.check.name == "keyframe_interval")
    assert run.result is not None
    assert not any(k.endswith("/max_keyframe_gap_s") for k in run.result.measurements)
    # source_episode is joints-only, so the camera set is empty and the
    # fact returned no camera keys; the assertion is that the body matched
    # the poisoned fact (no gap_s key in the result).
    assert run.result.measurements == {}


def test_keyframe_interval_raises_on_an_unrecognised_camera() -> None:
    from hflow.checks import _KeyframeIntervalPerCamera

    inter = _KeyframeIntervalPerCamera(frame_count=5, keyframe_indices=(0,))
    with pytest.raises(ValueError, match="no branch"):
        _keyframe_interval_value("/wrist_cam/compressed", "mystery", inter, np.array([0, 1, 2, 3, 4]))


def test_keyframe_interval_raises_when_median_is_named_for_one_keyframe() -> None:
    from hflow.checks import _KeyframeIntervalPerCamera

    inter = _KeyframeIntervalPerCamera(frame_count=5, keyframe_indices=(0,))
    with pytest.raises(ValueError, match="fewer than two keyframes"):
        _keyframe_interval_value(
            "/wrist_cam/compressed",
            "median_keyframe_interval_s",
            inter,
            np.array([0, 1, 2, 3, 4]),
        )
