"""Built-in checks, shipped in the same shape users write (evidence, not
verdicts; thresholds user-owned). Wrap them to register::

    @app.check()
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
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hflow.episode import Episode
from hflow.ffmpeg import ffmpeg_version, frame_stats
from hflow.steps import (
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
# passes it to `@app.check(gate=...)`. Copy it with your own numbers to tune,
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
    """
    infos = episode.topics
    selected = (
        list(topics)
        if topics is not None
        else sorted(topic for topic, info in infos.items() if info.message_count >= 2)
    )
    measurements: dict[str, MeasurementValue] = {}
    intervals: list[Interval] = []

    for topic in selected:
        stamps_ns = episode.channel(topic).timestamps
        if len(stamps_ns) < 2:
            measurements[f"{topic}/period_sample_count"] = len(stamps_ns)
            continue
        deltas_s = np.diff(stamps_ns) / 1e9
        declared_hz = (expected_hz or {}).get(topic)
        expected_period_s = 1.0 / declared_hz if declared_hz else float(np.median(deltas_s))
        violation_mask = np.abs(deltas_s - expected_period_s) > tolerance_s
        measurements[f"{topic}/median_dt_s"] = float(np.median(deltas_s))
        measurements[f"{topic}/period_violation_pct"] = float(np.mean(violation_mask) * 100.0)
        measurements[f"{topic}/max_gap_s"] = float(np.max(deltas_s))
        gap_indices: list[int] = np.flatnonzero(deltas_s > gap_factor * expected_period_s).tolist()
        intervals.extend(
            Interval(
                start_ns=int(stamps_ns[index]),
                end_ns=int(stamps_ns[index + 1]),
                label=f"gap:{topic}",
            )
            for index in gap_indices
        )

    camera_topics = [topic for topic in episode.cameras if topic in selected]
    state_topics = [
        topic
        for topic in selected
        if topic not in episode.cameras and infos[topic].message_count >= 2
    ]
    if camera_topics and state_topics:
        reference = max(state_topics, key=lambda topic: infos[topic].message_count)
        reference_stamps = episode.channel(reference).timestamps
        for camera in camera_topics:
            camera_stamps = episode.channel(camera).timestamps
            measurements[f"sync/{camera}~{reference}/start_offset_s"] = float(
                (camera_stamps[0] - reference_stamps[0]) / 1e9
            )
            measurements[f"sync/{camera}~{reference}/end_offset_s"] = float(
                (camera_stamps[-1] - reference_stamps[-1]) / 1e9
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

    Wraps the single-decode ffmpeg instrument (``hflow.ffmpeg.frame_stats``:
    blackframe + freezedetect + signalstats in one filter graph, one shared
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
    """
    selected_cameras = list(cameras) if cameras is not None else episode.cameras
    measurements: dict[str, MeasurementValue] = {}
    intervals: list[Interval] = []
    if selected_cameras:
        # Which ffmpeg produced these readings. The check's own version covers
        # its source and thresholds, but not the binary: different builds
        # genuinely measure differently, so a pin bump would otherwise move
        # every camera measurement in the corpus with nothing recording that it
        # had. Text, so it stays out of the wide view's numeric columns.
        measurements["camera_instrument"] = ffmpeg_version()
    for topic in selected_cameras:
        stamps_ns = episode.channel(topic).timestamps
        message_count = len(stamps_ns)
        measurements[f"{topic}/message_count"] = message_count
        if message_count >= 2:
            deltas_s = np.diff(stamps_ns) / 1e9
            declared_hz = (expected_hz or {}).get(topic)
            expected_period_s = 1.0 / declared_hz if declared_hz else float(np.median(deltas_s))
            span_s = float((stamps_ns[-1] - stamps_ns[0]) / 1e9)
            expected_frame_count = round(span_s / expected_period_s) + 1
            measurements[f"{topic}/expected_frame_count"] = expected_frame_count
            measurements[f"{topic}/frame_deficit_pct"] = float(
                100.0 * (expected_frame_count - message_count) / expected_frame_count
            )

        stats = frame_stats(
            episode.video(topic),
            black_frame_amount_pct=black_frame_amount_pct,
            black_pixel_threshold=black_pixel_threshold,
            freeze_noise_db=freeze_noise_db,
            freeze_min_duration_s=freeze_min_duration_s,
            bright_luma_threshold=bright_luma_threshold,
        )
        measurements[f"{topic}/decoded_frame_count"] = stats.frame_count
        measurements[f"{topic}/black_frame_pct"] = stats.black_frame_pct
        measurements[f"{topic}/overexposed_frame_pct"] = stats.overexposed_frame_pct
        measurements[f"{topic}/freeze_total_s"] = stats.freeze_total_s
        measurements[f"{topic}/luma_avg_mean"] = stats.luma_avg_mean
        measurements[f"{topic}/luma_avg_min"] = stats.luma_avg_min
        measurements[f"{topic}/luma_avg_max"] = stats.luma_avg_max
        if message_count:
            # Instrument times are seconds from the MP4 start, which is the
            # camera's first message; map freezes back onto the log clock.
            stream_start_ns = int(stamps_ns[0])
            intervals.extend(
                Interval(
                    start_ns=stream_start_ns + int(freeze_start_s * 1e9),
                    end_ns=stream_start_ns + int(freeze_end_s * 1e9),
                    label=f"freeze:{topic}",
                )
                for freeze_start_s, freeze_end_s in stats.freeze_intervals
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


def episode_duration(episode: Episode, *, topics: Sequence[str] | None = None) -> CheckResult:
    """Episode span and message volume, recorded for curation-side outlier cuts.

    An outlier is a corpus-relative judgment, so it cannot be decided inside
    a per-episode check without baking a threshold into the corpus; this
    check records the evidence and the cut is a curation query, e.g.::

        SELECT episode_id FROM episodes
        WHERE duration_s < 2 OR duration_s > 300
    """
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
    return CheckResult(
        measurements={
            "duration_s": duration_s,
            "message_count_total": message_count_total,
            "topic_count": len(selected),
        }
    )


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
    coverage denominator, and one content-hash version. Twenty-odd measurements
    under one name would mean a threshold on any of them gating all of them, and
    one changed constant re-versioning the lot. Both checks read the same cached
    remux, so the second costs a decode and no more.

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
        stats = frame_stats(
            episode.video(topic),
            black_pixel_threshold=black_pixel_threshold,
            freeze_noise_db=freeze_noise_db,
            freeze_min_duration_s=freeze_min_duration_s,
        )
        measurements.update(
            {
                f"{topic}/signal_frame_count": stats.frame_count,
                # Coding range, and the whole-scale extremes behind the verdict,
                # so a borderline call is auditable rather than asserted.
                f"{topic}/full_range_detected": int(stats.full_range_detected),
                f"{topic}/luma_min": stats.luma_min,
                f"{topic}/luma_max": stats.luma_max,
                # Robust exposure: the 10th/90th luma percentiles, so one hot or
                # dead pixel cannot manufacture a defect.
                f"{topic}/luma_p10_mean": stats.luma_p10_mean,
                f"{topic}/luma_p90_mean": stats.luma_p90_mean,
                f"{topic}/clipped_highlight_pct": stats.clipped_highlight_pct,
                f"{topic}/crushed_shadow_pct": stats.crushed_shadow_pct,
                # Black-pixel share over every frame, not only flagged frames.
                f"{topic}/black_pixel_share_mean": stats.black_pixel_share_mean,
                f"{topic}/black_pixel_share_max": stats.black_pixel_share_max,
                # Stillness, which is a different fact from a frozen feed: a
                # motionless scene reads near zero while freeze detection stays
                # silent.
                f"{topic}/frame_difference_mean": stats.frame_difference_mean,
                f"{topic}/frame_difference_max": stats.frame_difference_max,
                # Impulse noise and dropout streaks.
                f"{topic}/temporal_outlier_mean": stats.temporal_outlier_mean,
                f"{topic}/temporal_outlier_max": stats.temporal_outlier_max,
                f"{topic}/out_of_legal_range_mean": stats.out_of_legal_range_mean,
                f"{topic}/out_of_legal_range_max": stats.out_of_legal_range_max,
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
        measurements[f"{topic}/scanned_frame_count"] = len(stamps_ns)
        if len(stamps_ns) == 0:
            continue
        keyframe_indices = [
            index
            for index, payload in enumerate(channel.raw)
            if _payload_starts_a_keyframe(payload)
        ]
        measurements[f"{topic}/keyframe_count"] = len(keyframe_indices)
        measurements[f"{topic}/first_frame_is_keyframe"] = int(
            bool(keyframe_indices) and keyframe_indices[0] == 0
        )
        if not keyframe_indices:
            continue
        keyframe_stamps_ns = stamps_ns[keyframe_indices]
        # The tail matters: a long run after the last keyframe is just as
        # unseekable as a long run between two.
        gaps_ns = np.diff(np.append(keyframe_stamps_ns, stamps_ns[-1]))
        positive_gaps_ns = gaps_ns[gaps_ns > 0]
        measurements[f"{topic}/max_keyframe_gap_s"] = (
            float(np.max(positive_gaps_ns) / 1e9) if len(positive_gaps_ns) else 0.0
        )
        if len(keyframe_stamps_ns) >= 2:
            intervals_ns = np.diff(keyframe_stamps_ns)
            measurements[f"{topic}/median_keyframe_interval_s"] = float(
                np.median(intervals_ns) / 1e9
            )
    return CheckResult(measurements=measurements)
