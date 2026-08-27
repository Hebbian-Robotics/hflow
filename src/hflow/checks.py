"""Built-in checks, shipped in the same shape users write (evidence, not
verdicts; thresholds user-owned). Wrap them to register::

    @app.check(version="1")
    def timestamps(ep: hflow.Episode) -> hflow.CheckResult:
        return hflow.checks.timestamp_regularity(ep, tolerance_s=0.010)

No two checks here may claim the same measurement key. The catalog ranks
measurement rows per ``(episode_id, key)`` and every step of one run shares
that run's fingerprint and timestamp, so two checks emitting one key on one
episode is a tie one of them silently loses -- vanishing from
``measurements_latest`` and from the wide ``episodes`` view built on it. Hence
the per-check prefixes on otherwise-identical quantities
(``period_sample_count``, ``velocity_sample_count``, ``idle_sample_count``)
rather than a shared ``message_count``.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from hflow._video_measurement_toolchain import (
    measure_video_frame_statistics_for_hflow,
    resolved_video_measurement_toolchain,
)
from hflow._video_measurements import (
    CAMERA_MOTION_DEFINITION_VERSION,
    DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES,
    FRAME_STATISTICS_DEFINITION_VERSION,
    CameraMotionMeasurements,
    CameraMotionSettings,
    FrameStatisticsSettings,
    InsufficientVideoFrames,
    LumaRangeEvidence,
    VideoFrameStatistics,
    measure_camera_motion,
)
from hflow.episode import Episode
from hflow.steps import (
    CheckFunction,
    CheckResult,
    Comparison,
    Gate,
    Interval,
    MeasurementValue,
    Threshold,
)
from hflow.video import split_annex_b_stream

# Recommended camera-integrity thresholds over the keys `camera_frame_stats`
# emits. Shipped as a VALUE, not a default: nothing gates until a pipeline
# passes it to `@app.check(version="1", gate=...)`. Copy it with your own numbers to tune,
# or build a Gate of your own.
#
# Deliberately no motion-smoothness clause. Smoothness metrics ship as flags
# only, never a default reject rule -- Voxel51's audit found them scoring an
# early-gripper-release defect BETTER than clean demos, so a shipped threshold
# on one would reject the wrong episodes with our name on it. Nothing here
# matches a key from `joint_discontinuity` or `idle_fraction`, and
# `tests/test_gates.py` pins that.
#
# `*black_frame_pct` rather than `*/black_frame_pct`: the glob's `*` crosses
# `/`, but a leading `*/` still requires one, and examples/egocentric emits the
# key unprefixed. This form covers both shapes.
RECOMMENDED_CAMERA_INTEGRITY = Gate(
    accept_when=(
        # Half the episode blind is a dead camera, not a matter of taste. The
        # number the README, quickstart, and end-to-end camera gates all use.
        Threshold("*black_frame_pct", Comparison.AT_MOST, 50.0),
        # Matches camera_frame_stats' own freeze_min_duration_s default, so the
        # metric and the threshold agree on what counts as a freeze.
        Threshold("*freeze_total_s", Comparison.AT_MOST, 2.0),
    )
)


@dataclass(frozen=True)
class _JointMotionProfile:
    """Finite-difference motion facts of one state channel, computed once for
    every check that reasons about joint speed."""

    stamps_ns: np.ndarray
    deltas_s: np.ndarray
    per_step_max_speed: np.ndarray  # max over joints of |dq/dt|, one per step
    nonpositive_dt_count: int


def _joint_motion_profile(
    episode: Episode, topic: str, field: str | None
) -> _JointMotionProfile | None:
    """``None`` when the channel has fewer than two messages (no motion to
    profile); callers record the message count and stop."""
    channel = episode.channel(topic)
    positions = channel.to_numpy(field)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    stamps_ns = channel.timestamps
    if len(stamps_ns) < 2:
        return None
    deltas_s = np.diff(stamps_ns) / 1e9
    safe_deltas_s = np.where(deltas_s > 0, deltas_s, np.nan)
    velocities = np.abs(np.diff(positions, axis=0)) / safe_deltas_s[:, np.newaxis]
    return _JointMotionProfile(
        stamps_ns=stamps_ns,
        deltas_s=deltas_s,
        per_step_max_speed=np.nanmax(velocities, axis=1),
        nonpositive_dt_count=int(np.sum(deltas_s <= 0)),
    )


def _mask_run_intervals(
    stamps_ns: np.ndarray,
    step_mask: np.ndarray,
    label: str,
    *,
    min_duration_s: float = 0.0,
) -> list[Interval]:
    """Contiguous True runs of a per-step mask as labeled intervals.

    Step ``i`` spans ``stamps_ns[i]``..``stamps_ns[i + 1]``, so a run of steps
    ``r``..``i - 1`` spans ``stamps_ns[r]``..``stamps_ns[i]``.
    """
    intervals: list[Interval] = []

    def append_run(run_start_index: int, run_end_index: int) -> None:
        start_ns = int(stamps_ns[run_start_index])
        end_ns = int(stamps_ns[run_end_index])
        if (end_ns - start_ns) / 1e9 >= min_duration_s:
            intervals.append(Interval(start_ns=start_ns, end_ns=end_ns, label=label))

    run_start: int | None = None
    for index, in_run in enumerate(step_mask):
        if in_run and run_start is None:
            run_start = index
        elif not in_run and run_start is not None:
            append_run(run_start, index)
            run_start = None
    if run_start is not None:
        append_run(run_start, len(stamps_ns) - 1)
    return intervals


@dataclass(frozen=True)
class _TimestampRegularityPerTopic:
    """One selected topic's stamps and the rate the dispatcher needs.

    ``sparse`` is True when the topic has fewer than two messages: the
    body writes only ``period_sample_count`` in that case and the
    dispatcher must not produce any of the three period keys.
    """

    stamps_ns: np.ndarray
    deltas_s: np.ndarray | None
    expected_period_s: float
    sparse: bool


@dataclass(frozen=True)
class _TimestampRegularitySync:
    """The global sync-edge decision: which cameras pair with which
    state reference, and the reference stamps the offsets need."""

    camera_topics: tuple[str, ...]
    reference_topic: str | None
    reference_stamps_ns: np.ndarray | None


def _timestamp_regularity_resolve_selected(
    episode: Episode, topics: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    """Return ``(selected_topics, candidate_state_topics)``.

    The two pieces both the fact and the body walk; the fact only needs
    the selected list, the body also needs the state topics to pick the
    densest non-camera reference. Same selection rule, same partition --
    one statement of "which topics the check ran over".
    """
    infos = episode.topics
    selected = (
        list(topics)
        if topics is not None
        else sorted(topic for topic, info in infos.items() if info.message_count >= 2)
    )
    state_topics = [
        topic
        for topic in selected
        if topic not in episode.cameras and infos[topic].message_count >= 2
    ]
    return selected, state_topics


def timestamp_regularity_keys(episode: Episode) -> set[str]:
    """The one statement of ``timestamp_regularity``'s measurement key set.

    Per selected topic: ``period_sample_count`` when the topic has fewer
    than two messages, else the three period keys
    (``median_dt_s``/``period_violation_pct``/``max_gap_s``). Across all
    selected topics: a pair of ``sync/<cam>~<ref>/{start,end}_offset_s``
    keys per camera when the episode carries both camera and state
    streams -- the densest non-camera stream is the reference. ``App``'s
    pre-decode supersession consults this function through the routing
    map, which only ever sees the automatic bare registration. The
    selection rule mirrors the body's exactly: ``topics=`` is taken as
    given, otherwise every topic with at least two messages (#182).
    """
    selected, state_topics = _timestamp_regularity_resolve_selected(episode, None)
    keys: set[str] = set()
    for topic in selected:
        if episode.channel(topic).timestamps.size < 2:
            keys.add(f"{topic}/period_sample_count")
            continue
        keys.add(f"{topic}/median_dt_s")
        keys.add(f"{topic}/period_violation_pct")
        keys.add(f"{topic}/max_gap_s")
    camera_topics = [topic for topic in episode.cameras if topic in selected]
    if camera_topics and state_topics:
        reference = max(state_topics, key=lambda topic: episode.topics[topic].message_count)
        for camera in camera_topics:
            keys.add(f"sync/{camera}~{reference}/start_offset_s")
            keys.add(f"sync/{camera}~{reference}/end_offset_s")
    return keys


def _timestamp_regularity_value(
    episode: Episode,
    key: str,
    per_topic: dict[str, _TimestampRegularityPerTopic],
    sync: _TimestampRegularitySync,
    tolerance_s: float,
) -> MeasurementValue:
    """The value of one measurement key from the resolved intermediates.

    Raises on any key it does not recognise: an unbranched name means the
    fact and the dispatcher disagree, and letting ``match`` fall through
    to ``None`` would surface only as a null measurement under
    ``record=True`` (#182). ``episode`` is needed for the sync dispatch
    because the camera stamps are not on the per-topic struct.

    Two key shapes:
    - ``<topic>/<name>`` -- per-topic period keys
    - ``sync/<cam>~<ref>/<name>`` -- cross-stream offsets; the rightmost
      ``/`` divides the name, the rest is the camera~reference pair.
    """
    if key.startswith("sync/"):
        # ``sync/<cam>~<ref>/<start|end>_offset_s``
        cam_ref, _, name = key[len("sync/") :].rpartition("/")
        if name not in {"start_offset_s", "end_offset_s"}:
            raise ValueError(f"timestamp_regularity has no branch for the key {key!r}")
        if sync.reference_topic is None or sync.reference_stamps_ns is None:
            raise ValueError(
                f"timestamp_regularity has no branch for the key {key!r}: "
                "no sync reference (no state stream paired with a camera)"
            )
        cam_topic, sep, ref_in_key = cam_ref.rpartition("~")
        if not sep:
            raise ValueError(f"timestamp_regularity has no branch for the key {key!r}")
        if ref_in_key != sync.reference_topic or cam_topic not in sync.camera_topics:
            raise ValueError(f"timestamp_regularity has no branch for the key {key!r}")
        cam_stamps = episode.channel(cam_topic).timestamps
        ref_stamps = sync.reference_stamps_ns
        match name:
            case "start_offset_s":
                return float((cam_stamps[0] - ref_stamps[0]) / 1e9)
            case "end_offset_s":
                return float((cam_stamps[-1] - ref_stamps[-1]) / 1e9)
        raise ValueError(f"timestamp_regularity has no branch for the key {key!r}")
    topic, sep, name = key.rpartition("/")
    if not sep:
        raise ValueError(f"timestamp_regularity has no branch for the key {key!r}")
    if topic not in per_topic:
        raise ValueError(f"timestamp_regularity has no branch for the key {key!r}")
    inter = per_topic[topic]
    match name:
        case "period_sample_count":
            if not inter.sparse:
                raise ValueError(
                    f"{key!r}: fact named for a non-sparse topic; per-topic is dense"
                )
            return inter.stamps_ns.size
        case "median_dt_s":
            if inter.sparse or inter.deltas_s is None:
                raise ValueError(f"{key!r}: fact named for a sparse topic; per-topic is empty")
            return float(np.median(inter.deltas_s))
        case "period_violation_pct":
            if inter.sparse or inter.deltas_s is None:
                raise ValueError(f"{key!r}: fact named for a sparse topic; per-topic is empty")
            return float(
                np.mean(np.abs(inter.deltas_s - inter.expected_period_s) > tolerance_s) * 100.0
            )
        case "max_gap_s":
            if inter.sparse or inter.deltas_s is None:
                raise ValueError(f"{key!r}: fact named for a sparse topic; per-topic is empty")
            return float(np.max(inter.deltas_s))
    raise ValueError(f"timestamp_regularity has no branch for the key {key!r}")


def timestamp_regularity(
    episode: Episode,
    *,
    topics: Sequence[str] | None = None,
    expected_hz: dict[str, float] | None = None,
    tolerance_s: float = 0.010,
    gap_factor: float = 3.0,
) -> CheckResult:
    """Intra-stream timestamp regularity plus cross-stream sync offsets.

    Per topic: message deltas against the expected period (``1/expected_hz``
    when declared, else the stream's median delta -- LeRobot validates the
    same way post-hoc with a far tighter default tolerance; raw multi-sensor
    capture needs the looser default here). Deltas beyond ``gap_factor``
    periods become labeled gap intervals. Cross-stream: start/end offsets of
    every camera stream against the densest non-camera stream.

    The emitted key set is owned by :func:`timestamp_regularity_keys`: this
    body iterates that function's output and routes each key through
    ``_timestamp_regularity_value``, so a key the fact does not name cannot
    be emitted (#182).
    """
    selected, state_topics = _timestamp_regularity_resolve_selected(episode, topics)
    infos = episode.topics

    per_topic: dict[str, _TimestampRegularityPerTopic] = {}
    gap_indices_by_topic: dict[str, list[int]] = {}
    for topic in selected:
        stamps_ns = episode.channel(topic).timestamps
        if len(stamps_ns) < 2:
            per_topic[topic] = _TimestampRegularityPerTopic(
                stamps_ns=stamps_ns, deltas_s=None, expected_period_s=0.0, sparse=True
            )
            continue
        deltas_s = np.diff(stamps_ns) / 1e9
        declared_hz = (expected_hz or {}).get(topic)
        expected_period_s = 1.0 / declared_hz if declared_hz else float(np.median(deltas_s))
        per_topic[topic] = _TimestampRegularityPerTopic(
            stamps_ns=stamps_ns,
            deltas_s=deltas_s,
            expected_period_s=expected_period_s,
            sparse=False,
        )
        gap_indices_by_topic[topic] = np.flatnonzero(
            deltas_s > gap_factor * expected_period_s
        ).tolist()

    camera_topics = [topic for topic in episode.cameras if topic in selected]
    if camera_topics and state_topics:
        reference = max(state_topics, key=lambda topic: infos[topic].message_count)
        sync = _TimestampRegularitySync(
            camera_topics=tuple(camera_topics),
            reference_topic=reference,
            reference_stamps_ns=episode.channel(reference).timestamps,
        )
    else:
        sync = _TimestampRegularitySync(
            camera_topics=tuple(camera_topics),
            reference_topic=None,
            reference_stamps_ns=None,
        )

    measurements: dict[str, MeasurementValue] = {
        key: _timestamp_regularity_value(episode, key, per_topic, sync, tolerance_s)
        for key in sorted(timestamp_regularity_keys(episode))
    }

    intervals: list[Interval] = []
    for topic, gap_indices in gap_indices_by_topic.items():
        stamps_ns = per_topic[topic].stamps_ns
        intervals.extend(
            Interval(
                start_ns=int(stamps_ns[index]),
                end_ns=int(stamps_ns[index + 1]),
                label=f"gap:{topic}",
            )
            for index in gap_indices
        )
    return CheckResult(measurements=measurements, intervals=intervals)


def joint_discontinuity(
    episode: Episode,
    *,
    topic: str = "/joint_states",
    field: str = "position",
    velocity_limit: float = 3.0,
) -> CheckResult:
    """Finite-difference joint velocities against a configurable limit.

    Flags the "choppy joint states" issue named in Dyna's article. Ships as
    measurements and intervals only -- never a default reject rule:
    motion-smoothness heuristics are known to invert on real defects (the
    Voxel51 result), so the threshold and any verdict stay user-owned.
    """
    profile = _joint_motion_profile(episode, topic, field)
    if profile is None:
        return CheckResult(
            measurements={f"{topic}/velocity_sample_count": len(episode.channel(topic).timestamps)}
        )
    violation_mask = profile.per_step_max_speed > velocity_limit
    return CheckResult(
        measurements={
            f"{topic}/max_abs_velocity": float(np.nanmax(profile.per_step_max_speed)),
            f"{topic}/velocity_limit": velocity_limit,
            f"{topic}/violation_count": int(np.sum(violation_mask)),
            f"{topic}/violation_pct": float(np.mean(violation_mask) * 100.0),
            f"{topic}/nonpositive_dt_count": profile.nonpositive_dt_count,
        },
        intervals=_mask_run_intervals(
            profile.stamps_ns, violation_mask, f"joint_discontinuity:{topic}"
        ),
    )


def camera_frame_stats_keys(episode: Episode, *, cameras: Sequence[str] | None = None) -> set[str]:
    """The one statement of ``camera_frame_stats``' measurement key set.

    The check's body iterates this function's output, so a key is emitted
    exactly when it is named here -- there is no second list in production
    code to fall out of sync with (#182). ``cameras=None`` resolves to
    ``episode.cameras`` here exactly as it does in the body, so the fact and
    the body can never disagree about what was selected. ``App``'s
    pre-decode supersession consults this function through
    ``_DEFAULT_KEY_PATTERNS``, which only ever sees the automatic bare
    registration (a configured variant is a different function object and
    takes the post-execution path), so the fact is always called with
    default parameters.

    Per topic: ``message_count`` always; ``expected_frame_count`` and
    ``frame_deficit_pct`` only when the topic carries at least two messages;
    the seven decoded-evidence keys always. ``camera_instrument`` and
    ``camera_measurement_definition`` are the two non-topic keys; they ride
    inside the per-topic selection, so an episode with no cameras emits
    nothing at all.
    """
    selected = list(cameras) if cameras is not None else episode.cameras
    keys: set[str] = set()
    for topic in selected:
        keys.add(f"{topic}/message_count")
        if episode.channel(topic).timestamps.size >= 2:
            keys.add(f"{topic}/expected_frame_count")
            keys.add(f"{topic}/frame_deficit_pct")
        keys.update(
            f"{topic}/{name}"
            for name in (
                "decoded_frame_count",
                "black_frame_pct",
                "overexposed_frame_pct",
                "freeze_total_s",
                "luma_avg_mean",
                "luma_avg_min",
                "luma_avg_max",
            )
        )
    if selected:
        keys.add("camera_instrument")
        keys.add("camera_measurement_definition")
    return keys


@dataclass(frozen=True)
class _CameraIntermediates:
    """Everything one camera key's value needs from its topic, computed once.

    The instrument call is the expensive step (one ffmpeg decode per topic,
    cached on disk across runs); one instance per topic keeps the key
    dispatch at dict-lookup cost instead of re-decoding per key.
    """

    stamps_ns: np.ndarray
    stats: VideoFrameStatistics
    # None below two messages, mirroring the branches that need a rate.
    expected_frame_count: int | None


def _camera_intermediates(
    episode: Episode,
    topic: str,
    *,
    expected_hz: dict[str, float] | None = None,
    black_frame_amount_pct: int = 98,
    black_pixel_threshold: int = 17,
    freeze_noise_db: float = -60.0,
    freeze_min_duration_s: float = 2.0,
    bright_luma_threshold: float = 235.0,
) -> _CameraIntermediates:
    stamps_ns = episode.channel(topic).timestamps
    stats = measure_video_frame_statistics_for_hflow(
        episode.video(topic),
        settings=FrameStatisticsSettings(
            black_frame_minimum_pixel_share_percent=black_frame_amount_pct,
            black_pixel_luma_threshold=black_pixel_threshold,
            freeze_noise_tolerance_decibels=freeze_noise_db,
            freeze_minimum_duration_seconds=freeze_min_duration_s,
            overexposed_average_luma_threshold=bright_luma_threshold,
        ),
    )
    expected_frame_count = None
    if stamps_ns.size >= 2:
        deltas_s = np.diff(stamps_ns) / 1e9
        declared_hz = (expected_hz or {}).get(topic)
        expected_period_s = 1.0 / declared_hz if declared_hz else float(np.median(deltas_s))
        span_s = float((stamps_ns[-1] - stamps_ns[0]) / 1e9)
        expected_frame_count = round(span_s / expected_period_s) + 1
    return _CameraIntermediates(stamps_ns, stats, expected_frame_count)


def _camera_value(
    key: str, intermediates_by_topic: dict[str, _CameraIntermediates]
) -> MeasurementValue:
    """The value of one measurement key from its topic's intermediates.

    Raises on any key it does not recognise. The fact names the keys, so an
    unbranched name means the fact and the dispatcher disagree, and letting
    ``match`` fall through to ``None`` would surface only as a null value in
    a run that used ``record=True`` (#182). A mismapped *field* returns a
    plausible number and cannot crash -- the full-measurement fixtures in
    ``tests/test_default_checks.py`` are what catch those.
    """
    # Parse from the right: topics are paths (``/wrist_cam/compressed``), so
    # the topic prefix itself contains slashes and only the LAST separator
    # divides topic from measurement name. A bare key has no slash at all.
    topic, sep, name = key.rpartition("/")
    if sep:
        if topic not in intermediates_by_topic:
            raise ValueError(f"camera_frame_stats has no branch for the key {key!r}")
        inter = intermediates_by_topic[topic]
        match name:
            case "message_count":
                return inter.stamps_ns.size
            case "expected_frame_count":
                if inter.expected_frame_count is None:
                    raise ValueError(
                        f"{key!r} named for a topic that does not carry a rate: the fact "
                        "only emits it for topics with at least two stamps"
                    )
                return inter.expected_frame_count
            case "frame_deficit_pct":
                expected_frame_count = inter.expected_frame_count
                if expected_frame_count is None:
                    raise ValueError(
                        f"{key!r} named for a topic that does not carry a rate: the fact "
                        "only emits it for topics with at least two stamps"
                    )
                return float(
                    100.0 * (expected_frame_count - inter.stamps_ns.size) / expected_frame_count
                )
            case "decoded_frame_count":
                return inter.stats.decoded_frame_count
            case "black_frame_pct":
                return inter.stats.black_frame_percent
            case "overexposed_frame_pct":
                return inter.stats.overexposed_frame_percent
            case "freeze_total_s":
                return inter.stats.freeze_total_seconds
            case "luma_avg_mean":
                return inter.stats.average_luma_mean
            case "luma_avg_min":
                return inter.stats.average_luma_minimum
            case "luma_avg_max":
                return inter.stats.average_luma_maximum
        raise ValueError(f"camera_frame_stats has no branch for the key {key!r}")
    if name == "camera_instrument":
        # An explicit check version does not identify the measuring instrument,
        # and FFmpeg builds can produce different readings. Record the pinned
        # build so measurements remain interpretable across upgrades. One build
        # measures every topic in a run, so any computed topic's provenance
        # answers it.
        return next(iter(intermediates_by_topic.values())).stats.provenance.ffmpeg_version
    if name == "camera_measurement_definition":
        return FRAME_STATISTICS_DEFINITION_VERSION
    raise ValueError(f"camera_frame_stats has no branch for the key {key!r}")


def camera_frame_stats(
    episode: Episode,
    *,
    cameras: Sequence[str] | None = None,
    expected_hz: dict[str, float] | None = None,
    black_frame_amount_pct: int = 98,
    black_pixel_threshold: int = 17,
    freeze_noise_db: float = -60.0,
    freeze_min_duration_s: float = 2.0,
    bright_luma_threshold: float = 235.0,
) -> CheckResult:
    """Camera blackout, freeze, exposure, and frame-count evidence per camera.

    Wraps the incubating single-decode video measurement instrument
    (blackframe + freezedetect + signalstats in one filter graph, one shared
    frame denominator) over each camera's lossless MP4 remux, and compares
    the stored frame count against the rate the stream claims (declared
    ``expected_hz`` when given, else the stream's median delta) -- the
    LeRobot-style frame-count-vs-rate question. Freeze spans become labeled
    ``freeze:<topic>`` intervals in log time. Requires a canonical episode
    (``Episode.video`` remuxes in-band H.264 only).

    Evidence only, as always: blackout/exposure thresholds (including
    ``bright_luma_threshold``) and any verdict stay user-owned.

    ``black_pixel_threshold`` defaults to 17 rather than ffmpeg's own 32:
    17 still counts video-range black (16) as black, where 32 also counts
    ordinary dark detail and reads a few percent of unremarkable footage as
    dark. Signal-quality measurements from the same decode pass -- coding
    range, exposure, impulse noise -- live in
    :func:`camera_signal_quality`.

    The emitted key set is owned by :func:`camera_frame_stats_keys`: this
    body iterates that function's output and routes each key through
    ``_camera_value``, so a key the fact does not name cannot be emitted.
    The trade is one right-to-left key parse (``rpartition``) plus one dict
    lookup per key on top of the ffmpeg decode each topic already pays (#182).
    """
    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    intermediates_by_topic = {
        topic: _camera_intermediates(
            episode,
            topic,
            expected_hz=expected_hz,
            black_frame_amount_pct=black_frame_amount_pct,
            black_pixel_threshold=black_pixel_threshold,
            freeze_noise_db=freeze_noise_db,
            freeze_min_duration_s=freeze_min_duration_s,
            bright_luma_threshold=bright_luma_threshold,
        )
        for topic in selected_cameras
    }
    measurements: dict[str, MeasurementValue] = {
        key: _camera_value(key, intermediates_by_topic)
        for key in sorted(camera_frame_stats_keys(episode, cameras=selected_cameras))
    }
    intervals: list[Interval] = []
    for topic in selected_cameras:
        inter = intermediates_by_topic[topic]
        if inter.stamps_ns.size:
            # Instrument times are seconds from the MP4 start, which is the
            # camera's first message; map freezes back onto the log clock.
            stream_start_ns = int(inter.stamps_ns[0])
            intervals.extend(
                Interval(
                    start_ns=stream_start_ns + int(freeze_interval.start_seconds * 1e9),
                    end_ns=stream_start_ns + int(freeze_interval.end_seconds * 1e9),
                    label=f"freeze:{topic}",
                )
                for freeze_interval in inter.stats.freeze_intervals
            )
    return CheckResult(measurements=measurements, intervals=intervals)


def idle_fraction(
    episode: Episode,
    *,
    topic: str = "/joint_states",
    field: str | None = None,
    velocity_epsilon: float = 0.05,
    min_interval_s: float = 1.0,
) -> CheckResult:
    """Time-weighted fraction of the episode spent with no joint moving.

    A step is idle when every joint's finite-difference speed is below
    ``velocity_epsilon``; the fraction weights each step by its own duration,
    so irregular sampling does not skew it. Idle runs at least
    ``min_interval_s`` long become labeled ``idle:<topic>`` intervals.
    Evidence for curation cuts over mostly-stationary demonstrations -- the
    keep/drop policy (and any verdict) stays user-owned.
    """
    profile = _joint_motion_profile(episode, topic, field)
    if profile is None:
        return CheckResult(
            measurements={f"{topic}/idle_sample_count": len(episode.channel(topic).timestamps)}
        )
    idle_mask = profile.per_step_max_speed < velocity_epsilon
    positive_deltas_s = np.where(profile.deltas_s > 0, profile.deltas_s, 0.0)
    total_span_s = float(np.sum(positive_deltas_s))
    idle_total_s = float(np.sum(positive_deltas_s[idle_mask]))
    return CheckResult(
        measurements={
            f"{topic}/idle_fraction": idle_total_s / total_span_s if total_span_s else 0.0,
            f"{topic}/idle_total_s": idle_total_s,
            f"{topic}/velocity_epsilon": velocity_epsilon,
        },
        intervals=_mask_run_intervals(
            profile.stamps_ns, idle_mask, f"idle:{topic}", min_duration_s=min_interval_s
        ),
    )


@dataclass(frozen=True)
class _TimestampRegularityPerTopic:
    """One selected topic's deltas and the rate the dispatcher needs.

    ``sparse`` is True when the topic has fewer than two messages: the
    body writes only ``period_sample_count`` in that case and the
    dispatcher must not produce any of the three period keys.
    """

    stamps_ns: np.ndarray
    deltas_s: np.ndarray | None
    expected_period_s: float
    sparse: bool


@dataclass(frozen=True)
class _TimestampRegularitySync:
    """The global sync-edge decision: which cameras pair with which
    state reference, and the reference stamps the offsets need."""

    camera_topics: tuple[str, ...]
    reference_topic: str | None
    reference_stamps_ns: np.ndarray | None


def _timestamp_regularity_resolve_selected(
    episode: Episode, topics: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    """Return ``(selected_topics, candidate_state_topics)``.

    The two pieces both the fact and the body walk; the fact only needs
    the selected list, the body also needs the state topics to pick the
    densest non-camera reference. Same selection rule, same partition --
    one statement of "which topics the check ran over".
    """
    infos = episode.topics
    selected = (
        list(topics)
        if topics is not None
        else sorted(topic for topic, info in infos.items() if info.message_count >= 2)
    )
    state_topics = [
        topic
        for topic in selected
        if topic not in episode.cameras and infos[topic].message_count >= 2
    ]
    return selected, state_topics


@dataclass(frozen=True)
class _EpisodeDurationIntermediates:
    """Everything one ``episode_duration`` key's value reads, computed once."""

    duration_s: float
    message_count_total: int
    topic_count: int


def _episode_duration_intermediates(
    episode: Episode,
    topics: Sequence[str] | None,
) -> _EpisodeDurationIntermediates:
    """Verbatim aggregation from the pre-fact body: explicit ``topics``
    select as given, otherwise every topic carrying at least one message."""
    infos = episode.topics
    selected = (
        list(topics)
        if topics is not None
        else sorted(topic for topic, info in infos.items() if info.message_count >= 1)
    )
    start_candidates_ns: list[int] = []
    end_candidates_ns: list[int] = []
    message_count_total = 0
    for topic in selected:
        stamps_ns = episode.channel(topic).timestamps
        if len(stamps_ns) == 0:
            continue
        message_count_total += len(stamps_ns)
        start_candidates_ns.append(int(stamps_ns[0]))
        end_candidates_ns.append(int(stamps_ns[-1]))
    duration_s = (
        (max(end_candidates_ns) - min(start_candidates_ns)) / 1e9 if start_candidates_ns else 0.0
    )
    return _EpisodeDurationIntermediates(
        duration_s=duration_s,
        message_count_total=message_count_total,
        topic_count=len(selected),
    )


def _episode_duration_value(
    key: str, intermediates: _EpisodeDurationIntermediates
) -> MeasurementValue:
    """The value of one measurement key from the aggregated intermediates.

    Raises on any key it does not recognise: an unbranched name means the
    fact and the dispatcher disagree, and letting ``match`` fall through to
    ``None`` would surface only as a null measurement under ``record=True``
    (#182).
    """
    match key:
        case "duration_s":
            return intermediates.duration_s
        case "message_count_total":
            return intermediates.message_count_total
        case "topic_count":
            return intermediates.topic_count
    raise ValueError(f"episode_duration has no branch for the key {key!r}")


def episode_duration_keys(_episode: Episode) -> set[str]:
    """The one statement of ``episode_duration``'s measurement key set.

    Three fixed keys, independent of which topics the episode carries --
    including of the check's own ``topics=`` selection, which changes only
    which streams' stamps feed the numbers, never their names (#182).
    ``App``'s pre-decode supersession consults this function through the
    routing map, which only ever sees the automatic bare registration.
    """
    return {"duration_s", "message_count_total", "topic_count"}


def episode_duration(episode: Episode, *, topics: Sequence[str] | None = None) -> CheckResult:
    """Episode span and message volume, recorded for curation-side outlier cuts.

    An outlier is a corpus-relative judgment, so it cannot be decided inside
    a per-episode check without baking a threshold into the corpus; this
    check records the evidence and the cut is a curation query, e.g.::

        SELECT episode_id FROM episodes
        WHERE duration_s < 2 OR duration_s > 300

    The emitted key set is owned by :func:`episode_duration_keys`: this body
    iterates that function's output and routes each key through
    ``_episode_duration_value``, so a key the fact does not name cannot be
    emitted (#182).
    """
    intermediates = _episode_duration_intermediates(episode, topics)
    measurements: dict[str, MeasurementValue] = {
        key: _episode_duration_value(key, intermediates)
        for key in sorted(episode_duration_keys(episode))
    }
    return CheckResult(measurements=measurements)


def required_topics(episode: Episode, *, topics: Sequence[str]) -> CheckResult:
    """Presence and message volume for every topic a recording is expected to carry.

    A declared channel makes its topic present even when it contains no
    messages; the separate message count lets curation policy distinguish an
    absent topic from an empty publisher. Several channels may share one topic,
    so their message counts are combined from :attr:`Episode.channels` rather
    than resolving the ambiguous topic through :meth:`Episode.channel`.

    Evidence only: whether a missing or empty topic rejects an episode remains
    a user-owned curation decision.
    """
    message_counts_by_topic: dict[str, int] = {}
    for info in episode.channels.values():
        message_counts_by_topic[info.topic] = (
            message_counts_by_topic.get(info.topic, 0) + info.message_count
        )

    measurements: dict[str, MeasurementValue] = {}
    missing_topic_count = 0
    for topic in dict.fromkeys(topics):
        present = topic in message_counts_by_topic
        measurements[f"{topic}/present"] = present
        measurements[f"{topic}/message_count"] = message_counts_by_topic.get(topic, 0)
        if not present:
            missing_topic_count += 1
    measurements["missing_topic_count"] = missing_topic_count
    return CheckResult(measurements=measurements)


def action_rate(episode: Episode, *, topics: Sequence[str]) -> CheckResult:
    """Message rate of each given action topic, in hertz, plus a pooled total.

    Per topic, over that topic's own span -- so three 100 Hz streams read
    100 Hz each, not one number. ``pooled_message_rate_hz`` is the combined
    throughput over the union span, useful for "how much command traffic did
    this episode carry" but NOT a rate any single stream ran at; it rises with
    the number of topics you pass.

    Speed-vs-skill is a corpus-relative judgment, so it cannot be decided
    inside a per-episode check; this check records the evidence and the cut
    is a curation query, e.g.::

        SELECT episode_id FROM episodes
        WHERE "/joint_states/message_rate_hz_z" > 1.645

    (the window function producing the ``_z`` column is documented in the
    Cohort statistics section of docs/CATALOG.md).
    """
    measurements: dict[str, MeasurementValue] = {}
    start_candidates_ns: list[int] = []
    end_candidates_ns: list[int] = []
    interval_count_total = 0
    for topic in topics:
        stamps_ns = episode.channel(topic).timestamps
        if len(stamps_ns) == 0:
            continue
        start_candidates_ns.append(int(stamps_ns[0]))
        end_candidates_ns.append(int(stamps_ns[-1]))
        # n timestamps on a stream define n - 1 intervals
        interval_count = len(stamps_ns) - 1
        interval_count_total += interval_count
        topic_span_s = float((stamps_ns[-1] - stamps_ns[0]) / 1e9)
        measurements[f"{topic}/message_rate_hz"] = (
            interval_count / topic_span_s if topic_span_s > 0 else 0.0
        )
    union_span_s = (
        (max(end_candidates_ns) - min(start_candidates_ns)) / 1e9 if start_candidates_ns else 0.0
    )
    measurements["pooled_message_rate_hz"] = (
        interval_count_total / union_span_s if union_span_s > 0 else 0.0
    )
    return CheckResult(measurements=measurements)


def content_digest(episode: Episode) -> CheckResult:
    """A stable digest of the episode's message content, for duplicate hunts.

    SHA-256 over every channel's log times and raw payloads (channels in
    (topic, channel_id) order), so it identifies the recorded *content*
    independent of container layout: two files that differ only in chunking,
    compression, or metadata records digest the same. Exact-duplicate
    detection is then a curation query::

        SELECT value_text AS digest, count(*) AS copies
        FROM measurements WHERE key = 'content_digest'
        GROUP BY digest HAVING count(*) > 1

    Note ``episode_id`` already content-addresses the canonical *file*, which
    dedupes byte-identical re-ingests on its own; this digest additionally
    catches the same recording landed under different names or provenance.
    """
    digest = hashlib.sha256()
    ordered_channels = sorted(
        episode.channels.values(), key=lambda info: (info.topic, info.channel_id)
    )
    for info in ordered_channels:
        channel = episode.channel(info.channel_id)
        digest.update(info.topic.encode())
        digest.update(len(channel.raw).to_bytes(8, "big"))
        for log_time_ns, payload in zip(channel.timestamps, channel.raw, strict=True):
            digest.update(int(log_time_ns).to_bytes(8, "big", signed=True))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return CheckResult(measurements={"content_digest": digest.hexdigest()})


def camera_stability(
    episode: Episode,
    *,
    cameras: Sequence[str] | None = None,
    horizontal_field_of_view_degrees: float = DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES,
) -> CheckResult:
    """How much of each camera's footage is shaky rather than deliberately moving.

    Requires the ``motion`` extra (``pip install 'hflow[motion]'``). This is the
    one built-in with a dependency outside the core install, because optical flow
    and a RANSAC similarity fit have no numpy-only equivalent that measures the
    same thing.

    Tracks features between adjacent frames, fits a similarity transform to the
    tracks so independently moving subjects are discarded as outliers, converts
    the fit to angular rates, and splits those rates in time: below roughly 1 Hz
    is deliberate camera movement, above it is shake. A pair is unstable when its
    shake exceeds both the deliberate motion in that same pair and the
    instrument's own resolution floor -- so nothing here is a threshold anyone
    chose. Verified on synthetic footage: a static camera and a smooth pan both
    report no unstable footage, while injected shake scales monotonically with
    its amplitude.

    Read ``unstable_share`` next to ``coverage_share``. Footage no transform
    could be fitted to is reported as unclassified rather than as steady, so a
    low coverage means the share describes only the part that was measurable.

    Rates are degrees per second, named ``_dps``. Do not read the frame-to-frame
    difference in :func:`camera_signal_quality` as a stability signal: it cannot
    separate a shaking camera from a moving subject, which is the entire reason
    this check exists.
    """

    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    measurements: dict[str, MeasurementValue] = {}
    intervals: list[Interval] = []
    for topic in selected_cameras:
        stamps_ns = episode.channel(topic).timestamps
        if len(stamps_ns) < 2:
            measurements[f"{topic}/stability_sample_count"] = len(stamps_ns)
            continue
        # The stream's real rate, not a nominal one: the rates below are per
        # second, so a wrong fps scales every one of them.
        median_interval_s = float(np.median(np.diff(stamps_ns)) / 1e9)
        if median_interval_s <= 0:
            measurements[f"{topic}/stability_sample_count"] = len(stamps_ns)
            continue
        camera_motion_result = measure_camera_motion(
            episode.video(topic),
            toolchain=resolved_video_measurement_toolchain(),
            settings=CameraMotionSettings(
                frames_per_second=1.0 / median_interval_s,
                horizontal_field_of_view_degrees=horizontal_field_of_view_degrees,
            ),
        )
        if isinstance(camera_motion_result, InsufficientVideoFrames):
            measurements[f"{topic}/stability_sample_count"] = len(stamps_ns)
            continue
        assert isinstance(camera_motion_result, CameraMotionMeasurements)
        motion = camera_motion_result
        observed_s = motion.measured_seconds + motion.unclassified_seconds
        measurements.update(
            {
                f"{topic}/stability_sample_count": len(stamps_ns),
                f"{topic}/unstable_share": motion.unstable_share,
                f"{topic}/unstable_s": motion.unstable_seconds,
                f"{topic}/measured_s": motion.measured_seconds,
                f"{topic}/unclassified_s": motion.unclassified_seconds,
                f"{topic}/coverage_share": (
                    motion.measured_seconds / observed_s if observed_s > 0 else 0.0
                ),
                f"{topic}/shake_rate_p50_dps": motion.shake_rate_p50_degrees_per_second,
                f"{topic}/shake_rate_p95_dps": motion.shake_rate_p95_degrees_per_second,
                f"{topic}/intentional_rate_p50_dps": (
                    motion.intentional_rate_p50_degrees_per_second
                ),
                f"{topic}/resolution_floor_dps": motion.resolution_floor_degrees_per_second,
                f"{topic}/median_inlier_ratio": motion.median_inlier_ratio,
                f"{topic}/horizontal_fov_degrees": horizontal_field_of_view_degrees,
                "camera_motion_measurement_definition": CAMERA_MOTION_DEFINITION_VERSION,
                "camera_motion_ffmpeg": motion.provenance.ffmpeg_version,
                "camera_motion_opencv": motion.provenance.opencv_version,
            }
        )
        # Pair i spans frame i to frame i + 1, and canonical episodes carry one
        # frame per message, so a pair maps onto two consecutive log times.
        unstable_mask = np.zeros(max(len(stamps_ns) - 1, 0), dtype=bool)
        for pair_index in motion.unstable_pair_indices:
            if pair_index < len(unstable_mask):
                unstable_mask[pair_index] = True
        intervals.extend(_mask_run_intervals(stamps_ns, unstable_mask, f"unstable:{topic}"))
    return CheckResult(measurements=measurements, intervals=intervals)


def trajectory_metrics(
    episode: Episode,
    *,
    topic: str = "/joint_states",
    field: str | None = None,
    dimension_scales: Sequence[float] | None = None,
    motionless_speed_epsilon: float = 1e-3,
    final_pose_window_s: float = 0.5,
) -> CheckResult:
    """Episode-scope motion facts, for corpus-relative cuts.

    Records how far and how fast the stream moved, how much of it stood still,
    and whether it was still settling when the recording stopped -- the inputs
    to "which episodes in this task group look unlike the others", which is a
    curation query rather than a per-episode judgment.

    Reported in the stream's own units by default. Pass ``dimension_scales``
    (one positive divisor per dimension) to normalize dimensions that share no
    unit, and ``{topic}/scale_source`` records which you got. Deliberately no
    per-episode auto-scaling: a near-still episode has a tiny observed range, so
    dividing by it would inflate its own sensor jitter into apparent motion --
    inverting the metric on exactly the episodes worth catching.

    Evidence only, and this one ships no recommended gate at all: these are
    motion-smoothness metrics, which are known to invert on real defects
    (Voxel51's audit scored an early-gripper-release defect better than clean
    demos), so a threshold HFlow chose would reject the wrong episodes.
    """
    profile = _trajectory_profile(episode, topic, field, dimension_scales)
    if profile is None:
        return CheckResult(
            measurements={
                f"{topic}/trajectory_sample_count": len(episode.channel(topic).timestamps)
            }
        )

    measurements: dict[str, MeasurementValue] = {
        f"{topic}/trajectory_sample_count": len(profile.stamps_ns),
        f"{topic}/non_finite_sample_count": profile.non_finite_sample_count,
        f"{topic}/scale_source": profile.scale_source,
        f"{topic}/motionless_speed_epsilon": motionless_speed_epsilon,
        # Span of the accepted timeline: one corrupt trailing stamp shortens the
        # episode rather than stretching it, because it was never accepted.
        f"{topic}/trajectory_span_s": float((profile.stamps_ns[-1] - profile.stamps_ns[0]) / 1e9),
    }

    measured = np.isfinite(profile.speeds)
    if not np.any(measured):
        return CheckResult(measurements=measurements)

    measured_speeds = profile.speeds[measured]
    measured_durations_s = profile.step_durations_s[measured]
    # The denominator is the time actually measured, not the episode span:
    # dividing by the span would silently count rejected samples as motionless.
    valid_velocity_s = float(np.sum(measured_durations_s))
    measurements[f"{topic}/valid_velocity_s"] = valid_velocity_s
    measurements[f"{topic}/peak_velocity"] = float(np.max(measured_speeds))
    measurements[f"{topic}/mean_velocity"] = float(
        np.sum(measured_speeds * measured_durations_s) / valid_velocity_s
    )
    if valid_velocity_s > 0:
        motionless_s = float(
            np.sum(measured_durations_s[measured_speeds < motionless_speed_epsilon])
        )
        measurements[f"{topic}/motionless_fraction"] = motionless_s / valid_velocity_s
        measurements[f"{topic}/motionless_total_s"] = motionless_s

    measured_curvatures = profile.curvatures[np.isfinite(profile.curvatures)]
    if len(measured_curvatures):
        measurements[f"{topic}/trajectory_change_p95"] = float(
            np.percentile(measured_curvatures, 95)
        )
        measurements[f"{topic}/max_trajectory_change"] = float(np.max(measured_curvatures))

    # Was the arm still moving when recording stopped? A high ratio means the
    # episode was cut mid-motion, which matters for anything learning an
    # end-of-task pose.
    final_window_mask = measured & (
        (profile.stamps_ns[-1] - profile.stamps_ns[:-1]) / 1e9 <= final_pose_window_s
    )
    if np.any(final_window_mask):
        mean_speed = measurements[f"{topic}/mean_velocity"]
        final_speed = float(np.mean(profile.speeds[final_window_mask]))
        measurements[f"{topic}/final_pose_speed"] = final_speed
        if isinstance(mean_speed, float) and mean_speed > 0:
            measurements[f"{topic}/final_pose_unsettled_ratio"] = final_speed / mean_speed
    return CheckResult(measurements=measurements)


def trajectory_segments(
    episode: Episode,
    *,
    topic: str = "/joint_states",
    field: str | None = None,
    dimension_scales: Sequence[float] | None = None,
    motionless_speed_epsilon: float = 1e-3,
    min_motionless_span_s: float = 0.4,
) -> CheckResult:
    """Where in the episode the motion did something worth looking at.

    The companion to :func:`trajectory_metrics`: that one answers "how does this
    episode compare", this one answers "when, inside it, did things happen".
    Emits ``motionless:<topic>``, ``trajectory_change:<topic>``, and
    ``peak_velocity:<topic>`` intervals in log time, which is what a timeline
    view renders and what an operator scrubs to.

    The change threshold is derived from the episode's own curvature
    distribution -- a duration-weighted median plus two robust sigmas, sigma
    from the median absolute deviation -- so it adapts to how energetic the task
    is rather than assuming a scale. That makes the count comparable across
    episodes of one task and NOT comparable across tasks; ranking episodes
    against their peers is a curation query over
    ``{topic}/max_trajectory_change``.

    Evidence only, and no recommended gate: see :func:`trajectory_metrics`.
    """
    profile = _trajectory_profile(episode, topic, field, dimension_scales)
    if profile is None:
        return CheckResult(
            measurements={f"{topic}/segment_sample_count": len(episode.channel(topic).timestamps)}
        )

    measurements: dict[str, MeasurementValue] = {
        f"{topic}/segment_sample_count": len(profile.stamps_ns),
        f"{topic}/min_motionless_span_s": min_motionless_span_s,
    }
    intervals: list[Interval] = []

    # Strict `<`: a speed exactly at the epsilon is moving, not motionless.
    motionless_mask = np.isfinite(profile.speeds) & (profile.speeds < motionless_speed_epsilon)
    motionless_intervals = _mask_run_intervals(
        profile.stamps_ns,
        motionless_mask,
        f"motionless:{topic}",
        min_duration_s=min_motionless_span_s,
    )
    intervals.extend(motionless_intervals)
    measurements[f"{topic}/motionless_span_count"] = len(motionless_intervals)

    measured_curvatures = profile.curvatures[np.isfinite(profile.curvatures)]
    if len(measured_curvatures) >= 2:
        # Weight each curvature sample by the time it covers, so an irregularly
        # sampled stream does not let its dense stretches set the threshold.
        curvature_weights = (profile.step_durations_s[:-1] + profile.step_durations_s[1:])[
            np.isfinite(profile.curvatures)
        ]
        median_curvature = _duration_weighted_median(measured_curvatures, curvature_weights)
        robust_sigma = _MEDIAN_ABSOLUTE_DEVIATION_TO_SIGMA * _duration_weighted_median(
            np.abs(measured_curvatures - median_curvature), curvature_weights
        )
        threshold = median_curvature + _CHANGE_SIGMA_MULTIPLIER * robust_sigma
        # The episode's own maximum is published by ``trajectory_metrics``,
        # which owns the episode-scope comparison facts; duplicating it here
        # would make the catalog pick one of the two rows arbitrarily.
        measurements[f"{topic}/trajectory_change_threshold"] = threshold
        measurements[f"{topic}/curvature_robust_sigma"] = robust_sigma

        # Strict `>`: a sample exactly at the threshold is not a change. A
        # curvature sample spans three stamps, so a run ending at sample e ends
        # at stamps[e + 2] -- using e + 1 would shrink every reported segment.
        change_mask = np.isfinite(profile.curvatures) & (profile.curvatures > threshold)
        change_intervals = _curvature_run_intervals(
            profile.stamps_ns, change_mask, f"trajectory_change:{topic}"
        )
        intervals.extend(change_intervals)
        measurements[f"{topic}/trajectory_change_count"] = len(change_intervals)

    measured = np.isfinite(profile.speeds)
    if np.any(measured):
        measured_speeds = profile.speeds[measured]
        measured_durations_s = profile.step_durations_s[measured]
        peak_speed = float(np.max(measured_speeds))
        mean_speed = float(
            np.sum(measured_speeds * measured_durations_s) / np.sum(measured_durations_s)
        )
        measurements[f"{topic}/peak_speed"] = peak_speed
        measurements[f"{topic}/peak_to_mean_speed_ratio"] = (
            peak_speed / mean_speed if mean_speed > 0 else 0.0
        )
        if mean_speed > 0 and peak_speed > _PEAK_VELOCITY_FLAG_FACTOR * mean_speed:
            peak_step = int(np.argmax(np.where(measured, profile.speeds, -np.inf)))
            # The speed belongs to the step, so the moment it was reached is the
            # step's END, not its start.
            intervals.append(
                Interval(
                    start_ns=int(profile.stamps_ns[peak_step]),
                    end_ns=int(profile.stamps_ns[peak_step + 1]),
                    label=f"peak_velocity:{topic}",
                )
            )
    return CheckResult(measurements=measurements, intervals=intervals)


def _curvature_run_intervals(
    stamps_ns: np.ndarray, curvature_mask: np.ndarray, label: str
) -> list[Interval]:
    """Contiguous curvature runs as intervals, honouring the three-stamp span.

    Curvature sample ``i`` is computed from stamps ``i``, ``i + 1`` and
    ``i + 2``, so a run of samples ``r``..``e`` covers ``stamps[r]`` through
    ``stamps[e + 2]``, clamped to the last stamp.
    """
    intervals: list[Interval] = []
    last_index = len(stamps_ns) - 1
    run_start: int | None = None
    for index, in_run in enumerate(curvature_mask):
        if in_run and run_start is None:
            run_start = index
        elif not in_run and run_start is not None:
            intervals.append(
                Interval(
                    start_ns=int(stamps_ns[run_start]),
                    end_ns=int(stamps_ns[min(index + 1, last_index)]),
                    label=label,
                )
            )
            run_start = None
    if run_start is not None:
        intervals.append(
            Interval(
                start_ns=int(stamps_ns[run_start]),
                end_ns=int(stamps_ns[last_index]),
                label=label,
            )
        )
    return intervals


def camera_signal_quality(
    episode: Episode,
    *,
    cameras: Sequence[str] | None = None,
    black_pixel_threshold: int = 17,
    freeze_noise_db: float = -60.0,
    freeze_min_duration_s: float = 2.0,
) -> CheckResult:
    """Signal-level camera evidence: coding range, exposure, noise, stillness.

    Separate from :func:`camera_frame_stats` rather than folded into it, because
    a check is the unit of three things at once -- one gate decision, one
    coverage denominator, and one explicit version. Twenty-odd measurements
    under one name would mean a threshold on any of them gating all of them, and
    a deliberate version bump for one definition changing every measurement's
    version. HFlow's adapter caches the instrument's raw output beside the
    workdir MP4, keyed on the measurement definition, FFmpeg version, and filter
    graph. Registering both checks against the same episode therefore pays one
    FFmpeg decode per camera, not two. Both checks use identical graph parameters
    by default, so they share the decode. Override a graph parameter on only one
    check and HFlow decodes again because you asked FFmpeg for a different
    measurement.

    The coding range is measured from the pixels, never read from the
    container's declared range. That tag lies in practice -- a corpus can
    declare limited range while filling the full scale -- and believing it
    published an exposure-defect share off by more than two orders of
    magnitude. ``{topic}/full_range_detected`` records what was measured, and it
    selects which exposure gates apply, so the shares below mean the same thing
    across corpora encoded differently.

    Read ``out_of_legal_range_mean`` together with ``full_range_detected``: on
    full-range footage it is dominated by that range mismatch rather than by any
    defect, so it is an encoding-hygiene signal there, not a quality one.

    These readings are comparable only within one ffmpeg build. Builds disagree
    about absolute luma by more than rounding -- on the same file, ffmpeg 6.1
    range-scales full-range footage to a floor of 8 where the pinned 8.1
    preserves 0, which moves every percentile here and the range-gated shares
    with them. That is why HFlow pins its own binary and why
    ``camera_frame_stats`` records which one measured; compare across a pin bump
    only after re-measuring, not by reading old rows next to new ones.
    """
    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    measurements: dict[str, MeasurementValue] = {}
    for topic in selected_cameras:
        frame_statistics = measure_video_frame_statistics_for_hflow(
            episode.video(topic),
            settings=FrameStatisticsSettings(
                black_pixel_luma_threshold=black_pixel_threshold,
                freeze_noise_tolerance_decibels=freeze_noise_db,
                freeze_minimum_duration_seconds=freeze_min_duration_s,
            ),
        )
        measurements.update(
            {
                f"{topic}/signal_frame_count": frame_statistics.decoded_frame_count,
                # Coding range, and the whole-scale extremes behind the verdict,
                # so a borderline call is auditable rather than asserted.
                f"{topic}/full_range_detected": int(
                    frame_statistics.luma_range_evidence
                    is LumaRangeEvidence.EXTENDS_BEYOND_NOMINAL_LIMITED_RANGE
                ),
                f"{topic}/luma_min": frame_statistics.minimum_luma,
                f"{topic}/luma_max": frame_statistics.maximum_luma,
                # Robust exposure: the 10th/90th luma percentiles, so one hot or
                # dead pixel cannot manufacture a defect.
                f"{topic}/luma_p10_mean": frame_statistics.tenth_percentile_luma_mean,
                f"{topic}/luma_p90_mean": frame_statistics.ninetieth_percentile_luma_mean,
                f"{topic}/clipped_highlight_pct": (
                    frame_statistics.clipped_highlight_frame_percent
                ),
                f"{topic}/crushed_shadow_pct": frame_statistics.crushed_shadow_frame_percent,
                # Black-pixel share over every frame, not only flagged frames.
                f"{topic}/black_pixel_share_mean": frame_statistics.black_pixel_share_mean,
                f"{topic}/black_pixel_share_max": frame_statistics.black_pixel_share_maximum,
                # Stillness, which is a different fact from a frozen feed: a
                # motionless scene reads near zero while freeze detection stays
                # silent.
                f"{topic}/frame_difference_mean": frame_statistics.frame_difference_mean,
                f"{topic}/frame_difference_max": frame_statistics.frame_difference_maximum,
                # Impulse noise and dropout streaks.
                f"{topic}/temporal_outlier_mean": frame_statistics.temporal_outlier_mean,
                f"{topic}/temporal_outlier_max": frame_statistics.temporal_outlier_maximum,
                f"{topic}/out_of_legal_range_mean": frame_statistics.out_of_legal_range_mean,
                f"{topic}/out_of_legal_range_max": frame_statistics.out_of_legal_range_maximum,
                "camera_signal_instrument": frame_statistics.provenance.ffmpeg_version,
                "camera_signal_measurement_definition": FRAME_STATISTICS_DEFINITION_VERSION,
            }
        )
    return CheckResult(measurements=measurements)


def action_integrity(
    episode: Episode,
    *,
    topic: str = "/joint_states",
    field: str | None = None,
    min_frozen_run_fraction: float = 0.05,
    min_unchanged_dimension_samples: int = 11,
) -> CheckResult:
    """Integrity of the recorded values on one action or state stream.

    The other state checks read timing and speed; this one reads the numbers.
    Three defects it finds that nothing else here would:

    - **Non-finite samples.** A NaN reaching a training set is silent: it
      compares False against every threshold, so a naive filter reads it as
      clean rather than rejecting it.
    - **Frozen runs.** Consecutive samples exactly equal across every
      dimension, which is a stalled publisher rather than a still robot -- real
      sensor noise does not repeat bit-for-bit. Runs at least
      ``min_frozen_run_fraction`` of the stream land as ``frozen:<topic>``
      intervals.
    - **Dead dimensions.** A joint that never moves while others do, which is
      usually a miswired or unpublished channel rather than a task that held
      one axis still.

    Equality is exact IEEE, deliberately: a tolerance would dissolve the very
    runs this looks for, and NaN != NaN correctly breaks a run rather than
    extending it. Run this on raw channels only -- a resampled channel carrying
    a hold policy manufactures exactly-repeated samples, which is a fabricated
    frozen run rather than a recorded one.
    """
    channel = episode.channel(topic)
    stamps_ns = channel.timestamps
    if len(stamps_ns) < 2:
        return CheckResult(measurements={f"{topic}/integrity_sample_count": len(stamps_ns)})

    samples = channel.to_numpy(field)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    sample_count, dimension_count = samples.shape
    measurements: dict[str, MeasurementValue] = {
        f"{topic}/integrity_sample_count": sample_count,
        f"{topic}/dimension_count": dimension_count,
        f"{topic}/nan_count": int(np.count_nonzero(np.isnan(samples))),
        f"{topic}/inf_count": int(np.count_nonzero(np.isinf(samples))),
        f"{topic}/frozen_run_min_fraction": min_frozen_run_fraction,
    }

    # A step is frozen when every dimension repeats exactly. NaN != NaN, so a
    # non-finite sample breaks the run instead of silently extending it.
    repeats_previous = np.all(samples[1:] == samples[:-1], axis=1)
    frozen_intervals = _mask_run_intervals(
        stamps_ns,
        repeats_previous,
        f"frozen:{topic}",
        min_duration_s=0.0,
    )
    minimum_run_steps = max(1, int(np.ceil(min_frozen_run_fraction * (sample_count - 1))))
    run_lengths = _mask_run_lengths(repeats_previous)
    reported_runs = [length for length in run_lengths if length >= minimum_run_steps]
    measurements[f"{topic}/frozen_run_count"] = len(reported_runs)
    measurements[f"{topic}/frozen_longest_run_fraction"] = (
        max(run_lengths) / (sample_count - 1) if run_lengths else 0.0
    )
    measurements[f"{topic}/frozen_step_pct"] = float(np.mean(repeats_previous) * 100.0)

    # Below the sample floor an unchanged fraction is noise, not evidence, so
    # emit nothing rather than a number that averages into garbage downstream.
    if sample_count >= min_unchanged_dimension_samples:
        dimension_ever_changed = np.any(samples[1:] != samples[:-1], axis=0)
        unchanged_dimensions = np.flatnonzero(~dimension_ever_changed)
        measurements[f"{topic}/unchanged_dimension_count"] = len(unchanged_dimensions)
        for dimension_index in unchanged_dimensions:
            measurements[f"{topic}/dim{int(dimension_index):02d}/unchanged"] = 1

    return CheckResult(
        measurements=measurements,
        intervals=[
            interval
            for interval, length in zip(frozen_intervals, run_lengths, strict=True)
            if length >= minimum_run_steps
        ],
    )


# Trajectory analysis constants. These define what the measurements MEAN, so
# they are fixed rather than exposed: a configurable sigma multiplier would make
# two corpora's "trajectory change" counts incomparable, which is the one thing
# a shared measurement has to avoid. Thresholds a user should own live in
# curation SQL instead.
_MEDIAN_ABSOLUTE_DEVIATION_TO_SIGMA = 1.482602218505602
_CHANGE_SIGMA_MULTIPLIER = 2.0
_PEAK_VELOCITY_FLAG_FACTOR = 3.0
_FINAL_POSE_UNSETTLED_FACTOR = 2.0


@dataclass(frozen=True)
class _TrajectoryProfile:
    """Velocity and curvature of one action stream, with invalid samples masked.

    ``speeds[i]`` is the speed over the step from ``stamps_ns[i]`` to
    ``stamps_ns[i + 1]``; ``curvatures[i]`` is how much the velocity vector
    turned between step ``i`` and step ``i + 1``, so it spans
    ``stamps_ns[i]`` through ``stamps_ns[i + 2]``. Both carry NaN where the
    underlying samples could not be used.
    """

    stamps_ns: np.ndarray
    step_durations_s: np.ndarray
    speeds: np.ndarray
    curvatures: np.ndarray
    non_finite_sample_count: int
    scale_source: str


def _trajectory_profile(
    episode: Episode,
    topic: str,
    field: str | None,
    dimension_scales: Sequence[float] | None,
) -> _TrajectoryProfile | None:
    """``None`` when the stream cannot support a velocity at all."""
    channel = episode.channel(topic)
    stamps_ns = channel.timestamps
    samples = channel.to_numpy(field)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    if len(stamps_ns) < 2:
        return None

    # Duplicate and backward log times are common in real recordings. Keeping
    # only strictly-increasing stamps is what stops a zero-length step from
    # dividing into an infinite speed.
    advancing = np.ones(len(stamps_ns), dtype=bool)
    advancing[1:] = np.diff(stamps_ns) > 0
    stamps_ns = stamps_ns[advancing]
    samples = samples[advancing]
    if len(stamps_ns) < 2:
        return None

    non_finite_sample_count = int(np.count_nonzero(~np.isfinite(samples)))
    # Whole-sample invalidation: one unusable component makes the entire sample
    # unusable, because the speed is a magnitude over every dimension at once.
    # Masking per column instead would keep the other dimensions and quietly
    # report a smaller motion than actually occurred.
    sample_is_usable = np.all(np.isfinite(samples), axis=1)

    scales = (
        np.asarray(dimension_scales, dtype=float)
        if dimension_scales is not None
        else np.ones(samples.shape[1])
    )
    if scales.shape != (samples.shape[1],):
        raise ValueError(
            f"dimension_scales has {scales.shape[0]} entries but {topic!r} carries "
            f"{samples.shape[1]} dimensions"
        )
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("every dimension_scales entry must be finite and positive")
    scaled = samples / scales

    step_durations_s = np.diff(stamps_ns) / 1e9
    step_is_usable = sample_is_usable[:-1] & sample_is_usable[1:]
    velocities = np.diff(scaled, axis=0) / step_durations_s[:, np.newaxis]
    speeds = np.where(step_is_usable, np.linalg.norm(velocities, axis=1), np.nan)

    if len(speeds) >= 2:
        curvature_is_usable = step_is_usable[:-1] & step_is_usable[1:]
        curvatures = np.where(
            curvature_is_usable,
            np.linalg.norm(np.diff(velocities, axis=0), axis=1),
            np.nan,
        )
    else:
        curvatures = np.empty(0)

    return _TrajectoryProfile(
        stamps_ns=stamps_ns,
        step_durations_s=step_durations_s,
        speeds=speeds,
        curvatures=curvatures,
        non_finite_sample_count=non_finite_sample_count,
        scale_source="user" if dimension_scales is not None else "raw",
    )


def _duration_weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The first value whose cumulative weight reaches half the total.

    Not ``np.median`` and not an interpolating weighted quantile: an
    interpolated value is one no sample actually took, which on a short stream
    shifts the change threshold below and so changes which spans flag.
    """
    order = np.argsort(values, kind="stable")
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    half = cumulative[-1] / 2.0
    return float(values[order][int(np.searchsorted(cumulative, half))])


def _mask_run_lengths(step_mask: np.ndarray) -> list[int]:
    """Lengths of each contiguous True run, in the same order
    :func:`_mask_run_intervals` yields its intervals."""
    lengths: list[int] = []
    run_length = 0
    for in_run in step_mask:
        if in_run:
            run_length += 1
        elif run_length:
            lengths.append(run_length)
            run_length = 0
    if run_length:
        lengths.append(run_length)
    return lengths


def camera_fps_conformance(
    episode: Episode,
    *,
    nominal_fps: dict[str, int] | None = None,
    cameras: Sequence[str] | None = None,
    max_plausible_fps: int = 240,
    downsample_tolerance_fps: int = 1,
) -> CheckResult:
    """Classify each camera's timestamp-derived rate against the rate declared.

    ``timestamp_regularity`` measures jitter against a stream's own median;
    this asks the different question of whether the stream ran at the rate the
    corpus says it should. The distinction that matters in practice is between
    a stream recorded at half rate -- common, recoverable, and visible as a
    clean 2x ratio -- and one whose clock is simply not believable.

    ``{topic}/fps_resolution`` is the classification, as text:

    - ``matches-nominal``: derived rate equals the declared rate.
    - ``downsample-2x``: derived rate is twice nominal within
      ``downsample_tolerance_fps`` (a true 2x source lands on 59 or 61 as
      often as 60, which is why the tolerance exists).
    - ``fallback-nominal``: derived rate exceeds ``max_plausible_fps``, so the
      timestamps are not believable and the declared rate is all there is.
    - ``unrecoverable``: a plausible rate that is neither nominal nor 2x, so
      nothing explains the difference.
    - ``insufficient-frames`` / ``non-advancing-clock`` / ``no-nominal-declared``:
      the question could not be asked.

    ``{topic}/fps_ratio`` carries the same fact numerically, because only
    numeric measurements pivot into the wide view. This check measures and
    classifies; it never rewrites the stream -- decimating an episode is a
    transform concern that would move episode identity.
    """
    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    measurements: dict[str, MeasurementValue] = {}
    for topic in selected_cameras:
        stamps_ns = episode.channel(topic).timestamps
        measurements[f"{topic}/fps_sample_count"] = len(stamps_ns)
        declared_fps = (nominal_fps or {}).get(topic)
        if declared_fps is not None:
            measurements[f"{topic}/nominal_fps"] = declared_fps
        if len(stamps_ns) < 2:
            measurements[f"{topic}/fps_resolution"] = "insufficient-frames"
            continue
        intervals_ns = np.diff(stamps_ns)
        measurements[f"{topic}/nonpositive_interval_count"] = int(np.sum(intervals_ns <= 0))
        advancing_ns = intervals_ns[intervals_ns > 0]
        if len(advancing_ns) == 0:
            measurements[f"{topic}/fps_resolution"] = "non-advancing-clock"
            continue
        median_interval_ns = float(np.median(advancing_ns))
        measurements[f"{topic}/median_frame_interval_ns"] = median_interval_ns
        derived_fps = round(1e9 / median_interval_ns)
        measurements[f"{topic}/derived_fps"] = derived_fps
        if declared_fps is None or declared_fps <= 0:
            measurements[f"{topic}/fps_resolution"] = "no-nominal-declared"
            continue
        measurements[f"{topic}/fps_ratio"] = derived_fps / declared_fps
        measurements[f"{topic}/fps_resolution"] = _classify_derived_fps(
            derived_fps=derived_fps,
            declared_fps=declared_fps,
            max_plausible_fps=max_plausible_fps,
            downsample_tolerance_fps=downsample_tolerance_fps,
        )
    return CheckResult(measurements=measurements)


def _classify_derived_fps(
    *,
    derived_fps: int,
    declared_fps: int,
    max_plausible_fps: int,
    downsample_tolerance_fps: int,
) -> str:
    """Branch order is the classification: equality first, then plausibility,
    then the 2x window. A declared rate at or above half the plausibility
    ceiling therefore shadows ``downsample-2x``, and an equal-but-absurd
    declared rate still reads as matching -- both deliberate, because a
    declared rate the operator stands behind outranks our suspicion of it.
    """
    if derived_fps == declared_fps:
        return "matches-nominal"
    if derived_fps > max_plausible_fps:
        return "fallback-nominal"
    if abs(derived_fps - 2 * declared_fps) <= downsample_tolerance_fps:
        return "downsample-2x"
    return "unrecoverable"


def _payload_starts_a_keyframe(payload: bytes) -> bool:
    """Whether one canonical video message carries an IDR access unit.

    Reuses the encoder's own Annex B scan rather than re-deriving keyframe
    syntax here, so the check and the writer can never disagree about what a
    keyframe is. A payload that is not a single decodable access unit (a
    non-canonical episode, or a codec this scan does not parse) is not counted:
    conservative in the same direction as reporting no keyframes at all.
    """
    try:
        access_units = split_annex_b_stream(payload)
    except ValueError:
        return False
    return len(access_units) == 1 and access_units[0].is_keyframe


def media_digest(episode: Episode, *, cameras: Sequence[str] | None = None) -> CheckResult:
    """Per-camera digest of the encoded footage alone, for redundancy hunts.

    SHA-256 over one camera channel's length-framed payload bytes, deliberately
    excluding log times -- so the same footage delivered twice identifies as the
    same footage even when the second copy was re-stamped or arrived with
    different telemetry around it. That is the case ``content_digest`` cannot
    see, because it hashes log times and every other channel too: use this to
    find redundant *footage*, that one to find redundant *recordings*.

    Thresholdless by construction: a group is a fact about the collection, not
    a judgment, so the reduction is a curation query::

        SELECT value_text AS digest, count(*) AS copies, list(episode_id)
        FROM measurements_latest
        WHERE key LIKE '%/media_digest'
        GROUP BY digest HAVING count(*) > 1

    Redundant footage is then ``sum(copies - 1)`` over that result, and
    ``{topic}/media_bytes`` weights it by what the duplication costs to store.
    Reads no pixels and runs no decode, so it is exact and cheap.
    """
    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    measurements: dict[str, MeasurementValue] = {}
    for topic in selected_cameras:
        payloads = episode.channel(topic).raw
        digest = hashlib.sha256()
        total_bytes = 0
        for payload in payloads:
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            total_bytes += len(payload)
        measurements[f"{topic}/media_digest"] = digest.hexdigest()
        measurements[f"{topic}/media_bytes"] = total_bytes
    return CheckResult(measurements=measurements)


@dataclass(frozen=True)
class _KeyframeIntervalPerCamera:
    """One camera channel's keyframe walk, computed once.

    The shared scan over ``channel.raw`` is the only fact the
    ``keyframe_interval`` key set depends on that the body would not
    already pay for; the body needs the index list to compute the
    measurements, and the key set needs the count (zero / one / many)
    to decide whether ``max_keyframe_gap_s`` and
    ``median_keyframe_interval_s`` apply. ``_keyframe_indices`` runs
    the scan and is the one source of that count.
    """

    frame_count: int
    keyframe_indices: tuple[int, ...]


def _keyframe_indices(channel) -> tuple[int, ...]:
    """Shared payload-scan helper for ``keyframe_interval``: which message
    indices in ``channel.raw`` are keyframes. The fact and the body call
    this; the cost is one Annex B scan per camera per call, which is the
    irreducible price the default pays to know its own key set. Caching
    the result on the channel would let the pre-decode supersession
    consult the fact without a second scan, but no caller needs that
    today -- the body is the only reader, and the fact is only consulted
    when the default will run anyway. Revisit if a future change makes
    the pre-decode check the hot path for this default.
    """
    return tuple(
        index
        for index, payload in enumerate(channel.raw)
        if _payload_starts_a_keyframe(payload)
    )


def keyframe_interval_keys(episode: Episode, *, cameras: Sequence[str] | None = None) -> set[str]:
    """The one statement of ``keyframe_interval``'s measurement key set.

    Per selected camera: ``scanned_frame_count`` and ``keyframe_count``
    and ``first_frame_is_keyframe`` always (when the camera has any
    frames); ``max_keyframe_gap_s`` once at least one keyframe is
    found; ``median_keyframe_interval_s`` only when at least two
    keyframes are found. ``App``'s pre-decode supersession reads this
    through the routing map, which only ever sees the automatic bare
    registration.
    """
    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    keys: set[str] = set()
    for topic in selected_cameras:
        channel = episode.channel(topic)
        frame_count = channel.timestamps.size
        keys.add(f"{topic}/scanned_frame_count")
        if frame_count == 0:
            continue
        keyframe_indices = _keyframe_indices(channel)
        keys.add(f"{topic}/keyframe_count")
        keys.add(f"{topic}/first_frame_is_keyframe")
        if not keyframe_indices:
            continue
        keys.add(f"{topic}/max_keyframe_gap_s")
        if len(keyframe_indices) >= 2:
            keys.add(f"{topic}/median_keyframe_interval_s")
    return keys


def _keyframe_interval_value(
    topic: str,
    name: str,
    inter: _KeyframeIntervalPerCamera,
    stamps_ns: np.ndarray,
) -> MeasurementValue:
    """Dispatcher for one ``keyframe_interval`` key. The body builds the
    intermediates struct once per camera and iterates the fact's keys
    through this, so the key set has one source of truth (#182).
    """
    if name == "scanned_frame_count":
        return inter.frame_count
    if name == "keyframe_count":
        return len(inter.keyframe_indices)
    if name == "first_frame_is_keyframe":
        return int(bool(inter.keyframe_indices) and inter.keyframe_indices[0] == 0)
    if name == "max_keyframe_gap_s":
        if not inter.keyframe_indices:
            raise ValueError(f"{name!r}: fact named for a keyframe-less camera")
        keyframe_stamps_ns = stamps_ns[list(inter.keyframe_indices)]
        gaps_ns = np.diff(np.append(keyframe_stamps_ns, stamps_ns[-1]))
        positive_gaps_ns = gaps_ns[gaps_ns > 0]
        return float(np.max(positive_gaps_ns) / 1e9) if len(positive_gaps_ns) else 0.0
    if name == "median_keyframe_interval_s":
        if len(inter.keyframe_indices) < 2:
            raise ValueError(f"{name!r}: fact named for a camera with fewer than two keyframes")
        keyframe_stamps_ns = stamps_ns[list(inter.keyframe_indices)]
        intervals_ns = np.diff(keyframe_stamps_ns)
        return float(np.median(intervals_ns) / 1e9)
    raise ValueError(f"keyframe_interval has no branch for the key {topic}/{name}")


def keyframe_interval(episode: Episode, *, cameras: Sequence[str] | None = None) -> CheckResult:
    """Keyframe cadence per camera: how seekable and cuttable the footage is.

    A keyframe is where a decoder can start, so the longest gap between them
    bounds both random access and frame-accurate cutting without re-encoding.
    Measured over true log time rather than the remuxed MP4's synthesized
    constant-rate clock, so a recording gap legitimately widens the reported
    gap -- a cut across that gap really does land somewhere else.

    Evidence only. The right bar depends on the read pattern, so it belongs in
    a curation query: HFlow's own encoder targets 1 s GOPs for VLA-style
    training and 6 s for world models, while frame-accurate stream-copy cutting
    wants well under a second. Compare against the GOP your corpus was written
    with rather than an absolute::

        SELECT episode_id FROM episodes
        WHERE "/wrist_cam/compressed/max_keyframe_gap_s" <= 1.5

    ``max_keyframe_gap_s`` is omitted when no keyframe was found at all (an
    open-GOP or intra-refresh source whose recovery points this scan does not
    count as keyframes), so a ``<=`` filter excludes those rather than reading
    them as perfect.
    """
    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    measurements: dict[str, MeasurementValue] = {}
    for topic in selected_cameras:
        channel = episode.channel(topic)
        stamps_ns = channel.timestamps
        if stamps_ns.size == 0:
            measurements[f"{topic}/scanned_frame_count"] = 0
            continue
        inter = _KeyframeIntervalPerCamera(
            frame_count=stamps_ns.size,
            keyframe_indices=_keyframe_indices(channel),
        )
        for key in sorted(keyframe_interval_keys(episode, cameras=selected_cameras)):
            if not key.startswith(f"{topic}/"):
                continue
            name = key[len(topic) + 1 :]
            measurements[key] = _keyframe_interval_value(topic, name, inter, stamps_ns)
    return CheckResult(measurements=measurements)


# The checks an App runs unless its pipeline says otherwise. Every episode
# gets a baseline of evidence without anyone opting in, because the answer to
# "was this recording sound?" should not depend on whether the pipeline author
# remembered to ask.
#
# Membership has three conditions, and each one excludes something:
#
# - **Registrable with no configuration.** `required_topics`, `action_rate`,
#   and `camera_fps_conformance` need a topic list or a nominal rate that is
#   the pipeline author's to supply, so they cannot be defaults.
# - **Meaningful on any corpus.** The joint and trajectory checks assume a
#   state stream; on human egocentric video there is none, and a default that
#   records nothing on a whole class of corpora is noise. A camera-less or
#   camera-only recording simply gets fewer keys from the checks below.
# - **Cheap enough to never think about.** `camera_frame_stats` costs one
#   ffmpeg decode per camera and earns it (blackout, freeze, exposure, and
#   the stored-versus-claimed frame count all come from that one pass).
#   `camera_signal_quality` would cost a SECOND decode for a deeper reading of
#   the same footage, so it stays opt-in, as does `camera_stability`, which
#   needs the `motion` extra the core install does not carry.
#
# Automatic defaults are still explicitly versioned; HFlow is their author,
# so it owns these promises on behalf of pipelines that accept the baseline.
# A pipeline-authored check always declares its version at registration.
_DEFAULT_CHECK_VERSION_BY_FUNCTION: dict[CheckFunction, str] = {
    episode_duration: "1",
    timestamp_regularity: "1",
    camera_frame_stats: "1",
    keyframe_interval: "1",
    content_digest: "1",
    media_digest: "1",
}

# Pass `hflow.App(default_checks=...)` to change the set; registering one of
# these yourself replaces the automatic copy rather than colliding with it.
# Derive the tuple from the version mapping so adding an automatic check cannot
# create an unversioned registration.
DEFAULT_CHECKS: tuple[CheckFunction, ...] = tuple(_DEFAULT_CHECK_VERSION_BY_FUNCTION)


def _default_check_version_for_automatic_registration(check_function: CheckFunction) -> str:
    """Return HFlow's explicit version for one automatic built-in check."""
    try:
        return _DEFAULT_CHECK_VERSION_BY_FUNCTION[check_function]
    except KeyError:
        check_name = getattr(check_function, "__name__", repr(check_function))
        raise ValueError(
            f"default_checks contains {check_name!r}, which has no HFlow-owned version; "
            "register pipeline-authored checks with app.check(version=...)"
        ) from None


# Key-set predictor for every default: what ``measurements`` keys would this
# function emit for a given episode, without actually running it.
#
# Used by ``hflow.app.App`` to decide whether a pipeline step has already
# covered a default's keys, so the default can be short-circuited before the
# ffmpeg decode it would otherwise pay for. The patterns mirror the
# ``measurements[...] = ...`` writes in the function bodies above, line for
# line -- a separate, internal contract that the drift-guard test in
# ``tests/test_default_checks.py`` enforces: a default whose actual key set
# diverges from its pattern is a regression in this engine, not in the
# default itself.
#
# The contract is between this registry and the function bodies in this file.
# It is not a public API: there is no ``keys=`` parameter to
# ``@app.check(version=...)``,
# no ``__emitted_keys__`` convention, no way for user code to register a
# pattern. A user-registered check has no pattern, so a user check that
# happens to overlap a default's keys falls back to the post-execution
# comparison in ``_yield_defaults_superseded_by_the_pipeline`` -- the same
# path the same-parameter wrapper case has always taken.
def _content_digest_keys(_episode: Episode) -> set[str]:
    return {"content_digest"}


def _media_digest_keys(episode: Episode) -> set[str]:
    keys: set[str] = set()
    for topic in episode.cameras:
        keys.add(f"{topic}/media_digest")
        keys.add(f"{topic}/media_bytes")
    return keys


# Internal: maps a default function to its key-set predictor. ``App`` reads
# this once at default-skip time. For the five still-mirrored defaults the
# predictor restates its body's writes, and the drift-guard test in
# ``tests/test_default_checks.py`` keeps the two in lockstep.
# ``camera_frame_stats`` maps to its own fact function
# (:func:`camera_frame_stats_keys`), whose output that body itself iterates --
# prediction and emission are the same statement there (#182).
_DEFAULT_KEY_PATTERNS: dict[CheckFunction, Callable[[Episode], set[str]]] = {
    episode_duration: episode_duration_keys,
    timestamp_regularity: timestamp_regularity_keys,
    camera_frame_stats: camera_frame_stats_keys,
    keyframe_interval: keyframe_interval_keys,
    content_digest: _content_digest_keys,
    media_digest: _media_digest_keys,
}
