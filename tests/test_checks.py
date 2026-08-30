"""Direct unit tests for the built-in checks (paths e2e only grazes)."""

from pathlib import Path

import numpy as np
import pytest
from mcap.writer import Writer as StockWriter

import hflow
from hflow.checks import (
    action_integrity,
    action_rate,
    camera_fps_conformance,
    camera_frame_stats,
    camera_signal_quality,
    camera_stability,
    content_digest,
    episode_duration,
    idle_fraction,
    joint_discontinuity,
    keyframe_interval,
    media_digest,
    required_topics,
    timestamp_regularity,
    trajectory_metrics,
    trajectory_segments,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode


def test_no_two_builtin_checks_claim_the_same_measurement_key(tmp_path: Path) -> None:
    """The catalog ranks measurement rows per (episode_id, key) and every step of
    one run shares that run's fingerprint and timestamp, so two checks emitting
    one key on one episode is a tie one of them silently loses. Registering the
    built-ins together is the documented path (examples/stress/synthetic.py), so
    their key namespaces must not overlap.
    """
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=3.0, cameras=("wrist_cam",), joint_jump_at_s=1.5),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        results_by_check = {
            "timestamp_regularity": timestamp_regularity(episode),
            "joint_discontinuity": joint_discontinuity(episode),
            "idle_fraction": idle_fraction(episode),
            "camera_frame_stats": camera_frame_stats(episode),
            "episode_duration": episode_duration(episode),
            "action_rate": action_rate(episode, topics=["/joint_states"]),
            "required_topics": required_topics(episode, topics=["/rig/required"]),
            "content_digest": content_digest(episode),
            "media_digest": media_digest(episode),
            "keyframe_interval": keyframe_interval(episode),
            "camera_fps_conformance": camera_fps_conformance(episode),
            "action_integrity": action_integrity(episode),
            "camera_signal_quality": camera_signal_quality(episode),
            "trajectory_metrics": trajectory_metrics(episode),
            "trajectory_segments": trajectory_segments(episode),
            "camera_stability": camera_stability(episode),
        }

    producers_by_key: dict[str, list[str]] = {}
    for check_name, result in results_by_check.items():
        for key in result.measurements:
            producers_by_key.setdefault(key, []).append(check_name)
    collisions = {key: names for key, names in producers_by_key.items() if len(names) > 1}
    assert collisions == {}


def test_every_builtin_check_registers_bare_and_runs(tmp_path: Path) -> None:
    """A built-in registers without HFlow inspecting its implementation.

    The pipeline owns the version, and the check name scopes it, so several
    checks may intentionally use the same simple revision string.
    """
    app = hflow.App("bare-registration", data_root=tmp_path / "data")
    for builtin in (
        timestamp_regularity,
        joint_discontinuity,
        idle_fraction,
        episode_duration,
        content_digest,
        media_digest,
        keyframe_interval,
        camera_fps_conformance,
        action_integrity,
        trajectory_metrics,
        trajectory_segments,
        camera_frame_stats,
        camera_signal_quality,
        camera_stability,
    ):
        app.check(version="1")(builtin)

    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",)),
    )
    report = app.test(source, verbose=False)
    assert not report.has_errors, [run.error for run in report.checks if run.error]
    assert {run.check.version for run in report.checks} == {"1"}
    # Every built-in is evidence-only. Asserted across the whole set, so a
    # built-in that starts returning a verdict is caught even where no
    # per-check test pins it.
    assert all(run.result is not None and run.result.verdict is None for run in report.checks)


@pytest.fixture(scope="module")
def jittery_episode(tmp_path_factory: pytest.TempPathFactory) -> hflow.Episode:
    """State-only episode with the +3ms timestamp-offset segment enabled...

    on the camera -- so use a camera-bearing spec but a short one; the offset
    segment lives on camera 0 per the fixture contract.
    """
    path = synthesize_episode(
        tmp_path_factory.mktemp("checks") / "episode.mcap",
        SyntheticEpisodeSpec(
            duration_s=4.0,
            timestamp_offset_segment=(2.0, 3.0),
            joint_jump_at_s=1.5,
        ),
    )
    return hflow.Episode(path)


def test_declared_rate_beats_median_inference(jittery_episode: hflow.Episode) -> None:
    # With a deliberately wrong declared rate, every delta violates.
    result = timestamp_regularity(
        jittery_episode,
        topics=["/joint_states"],
        expected_hz={"/joint_states": 50.0},  # actual: 100 Hz
        tolerance_s=0.001,
    )
    violation_pct = result.measurements["/joint_states/period_violation_pct"]
    assert isinstance(violation_pct, float)
    assert violation_pct == 100.0


def test_offset_segment_is_flagged_at_tight_tolerance(jittery_episode: hflow.Episode) -> None:
    camera_topic = "/wrist_cam/compressed"
    result = timestamp_regularity(jittery_episode, topics=[camera_topic], tolerance_s=0.001)
    violation_pct = result.measurements[f"{camera_topic}/period_violation_pct"]
    assert isinstance(violation_pct, float)
    # The +3ms offset produces exactly two anomalous deltas (entry and exit).
    assert violation_pct > 0.0


def test_gap_intervals_are_labeled(tmp_path: Path) -> None:
    # A source with a genuine gap: build via the writer-level fixture trick --
    # simplest is a synthetic episode read back and re-checked with a small
    # gap_factor so the offset deltas register as gaps.
    path = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, timestamp_offset_segment=(1.0, 1.5)),
    )
    with hflow.Episode(path) as episode:
        result = timestamp_regularity(episode, topics=["/wrist_cam/compressed"], gap_factor=1.02)
    assert result.intervals, "offset entry delta must exceed 1.02x the period"
    assert all(interval.label == "gap:/wrist_cam/compressed" for interval in result.intervals)


def test_cross_stream_sync_measurements_exist(jittery_episode: hflow.Episode) -> None:
    result = timestamp_regularity(jittery_episode)
    sync_keys = [key for key in result.measurements if key.startswith("sync/")]
    # Two cameras, each measured against the densest state topic, two bounds.
    assert len(sync_keys) == 4
    for key in sync_keys:
        assert "~/joint_states/" in key


def test_joint_discontinuity_finds_the_injected_jump(jittery_episode: hflow.Episode) -> None:
    result = joint_discontinuity(jittery_episode, velocity_limit=3.0)
    violation_count = result.measurements["/joint_states/violation_count"]
    assert isinstance(violation_count, int)
    assert violation_count >= 1
    assert result.intervals
    assert result.verdict is None  # evidence, not verdicts


def test_joint_discontinuity_high_limit_is_quiet(jittery_episode: hflow.Episode) -> None:
    result = joint_discontinuity(jittery_episode, velocity_limit=1e6)
    assert result.measurements["/joint_states/violation_count"] == 0
    assert result.intervals == []


def test_idle_fraction_is_time_weighted_and_bounded(jittery_episode: hflow.Episode) -> None:
    # The synthetic joints move continuously: a tiny epsilon finds almost no
    # idle time, a huge one classifies the whole span as idle.
    moving = idle_fraction(jittery_episode, velocity_epsilon=1e-9)
    idle = idle_fraction(jittery_episode, velocity_epsilon=1e9, min_interval_s=0.5)
    moving_fraction = moving.measurements["/joint_states/idle_fraction"]
    idle_fraction_value = idle.measurements["/joint_states/idle_fraction"]
    assert isinstance(moving_fraction, float) and isinstance(idle_fraction_value, float)
    assert moving_fraction < 0.05
    assert idle_fraction_value == pytest.approx(1.0)
    assert idle.intervals and all(
        interval.label == "idle:/joint_states" for interval in idle.intervals
    )
    assert moving.verdict is None  # evidence, not verdicts


def test_episode_duration_matches_the_synthesized_span(jittery_episode: hflow.Episode) -> None:
    result = episode_duration(jittery_episode)
    duration_s = result.measurements["duration_s"]
    assert isinstance(duration_s, float)
    # The 4 s spec spans ~4 s of log time (one sample period short per stream).
    assert duration_s == pytest.approx(4.0, abs=0.1)
    message_count_total = result.measurements["message_count_total"]
    assert isinstance(message_count_total, int) and message_count_total > 0


def test_required_topics_records_present_topic_inventory(
    jittery_episode: hflow.Episode,
) -> None:
    requested = ["/joint_states", "/wrist_cam/compressed"]
    expected_counts = {
        topic: sum(
            info.message_count for info in jittery_episode.channels.values() if info.topic == topic
        )
        for topic in requested
    }

    result = required_topics(jittery_episode, topics=requested)

    assert result.measurements == {
        "/joint_states/present": True,
        "/joint_states/message_count": expected_counts["/joint_states"],
        "/wrist_cam/compressed/present": True,
        "/wrist_cam/compressed/message_count": expected_counts["/wrist_cam/compressed"],
        "missing_topic_count": 0,
    }
    assert result.verdict is None


def test_required_topics_counts_one_missing_topic(jittery_episode: hflow.Episode) -> None:
    result = required_topics(jittery_episode, topics=["/joint_states", "/imu"])

    assert result.measurements["/joint_states/present"] is True
    assert result.measurements["/imu/present"] is False
    assert result.measurements["/imu/message_count"] == 0
    assert result.measurements["missing_topic_count"] == 1


def test_required_topics_records_a_topic_absent_from_the_file(
    jittery_episode: hflow.Episode,
) -> None:
    result = required_topics(jittery_episode, topics=["/not-recorded"])

    assert result.measurements == {
        "/not-recorded/present": False,
        "/not-recorded/message_count": 0,
        "missing_topic_count": 1,
    }


def test_required_topics_aggregates_multiple_channels_for_one_topic(tmp_path: Path) -> None:
    source = tmp_path / "multi-channel.mcap"
    with source.open("wb") as stream:
        writer = StockWriter(stream)
        writer.start(profile="", library="test")
        first_channel = writer.register_channel(
            topic="/status", message_encoding="json", schema_id=0
        )
        second_channel = writer.register_channel(
            topic="/status", message_encoding="json", schema_id=0
        )
        for index in range(2):
            writer.add_message(
                first_channel,
                log_time=index,
                publish_time=index,
                data=b"{}",
            )
        for index in range(3):
            writer.add_message(
                second_channel,
                log_time=10 + index,
                publish_time=10 + index,
                data=b"{}",
            )
        writer.finish()

    with hflow.Episode(source) as episode:
        result = required_topics(episode, topics=["/status"])

    assert result.measurements == {
        "/status/present": True,
        "/status/message_count": 5,
        "missing_topic_count": 0,
    }


def test_action_rate_matches_the_synthesized_rate(jittery_episode: hflow.Episode) -> None:
    result = action_rate(jittery_episode, topics=["/joint_states"])
    rate_hz = result.measurements["/joint_states/message_rate_hz"]
    assert isinstance(rate_hz, float)
    # the synthetic joint stream runs at 100 Hz by SyntheticEpisodeSpec default
    assert rate_hz == pytest.approx(100.0, abs=0.5)


def test_action_rate_reports_each_topic_at_its_own_rate(
    jittery_episode: hflow.Episode,
) -> None:
    """Pooling several topics into one figure reported their sum as if it were
    a rate any single stream ran at. Each topic now answers for itself, and the
    pooled throughput is named as the different quantity it is.
    """
    camera_topic = "/wrist_cam/compressed"
    result = action_rate(jittery_episode, topics=["/joint_states", camera_topic])

    joint_rate_hz = result.measurements["/joint_states/message_rate_hz"]
    camera_rate_hz = result.measurements[f"{camera_topic}/message_rate_hz"]
    pooled_rate_hz = result.measurements["pooled_message_rate_hz"]
    assert isinstance(joint_rate_hz, float)
    assert isinstance(camera_rate_hz, float)
    assert isinstance(pooled_rate_hz, float)

    # Adding a 15 Hz camera must not inflate what the 100 Hz joint stream reports.
    assert joint_rate_hz == pytest.approx(100.0, abs=0.5)
    assert camera_rate_hz == pytest.approx(15.0, abs=0.5)
    assert pooled_rate_hz > joint_rate_hz


def test_content_digest_identifies_duplicate_content(tmp_path: Path) -> None:
    # Digest behavior is independent of camera encoding. A tiny state stream
    # keeps this contract focused on message content instead of fixture cost.
    spec = SyntheticEpisodeSpec(
        duration_s=0.2,
        cameras=(),
        joint_hz=10.0,
        joint_count=1,
        black_segment=None,
        joint_jump_at_s=None,
        timestamp_offset_segment=None,
    )
    first = synthesize_episode(tmp_path / "a.mcap", spec)
    duplicate = synthesize_episode(tmp_path / "b.mcap", spec)
    different = synthesize_episode(
        tmp_path / "c.mcap",
        SyntheticEpisodeSpec(
            duration_s=0.2,
            cameras=(),
            joint_hz=10.0,
            joint_count=1,
            black_segment=None,
            joint_jump_at_s=None,
            timestamp_offset_segment=None,
            seed=1,
        ),
    )
    with (
        hflow.Episode(first) as ep_a,
        hflow.Episode(duplicate) as ep_b,
        hflow.Episode(different) as ep_c,
    ):
        digest_a = content_digest(ep_a).measurements["content_digest"]
        digest_b = content_digest(ep_b).measurements["content_digest"]
        digest_c = content_digest(ep_c).measurements["content_digest"]
    assert digest_a == digest_b
    assert digest_a != digest_c


def test_camera_frame_stats_sees_the_injected_black_segment(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=4.0, cameras=("wrist_cam",), black_segment=(1.0, 2.0)),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        result = camera_frame_stats(episode)
    black_frame_pct = result.measurements[f"{camera_topic}/black_frame_pct"]
    assert isinstance(black_frame_pct, float)
    # 1 s of 4 s is black; decode boundaries make the exact count fuzzy.
    assert 10.0 < black_frame_pct < 50.0
    # Non-bright fixture: overexposed_frame_pct should be 0.0 at default threshold.
    overexposed_frame_pct = result.measurements[f"{camera_topic}/overexposed_frame_pct"]
    assert isinstance(overexposed_frame_pct, float)
    assert overexposed_frame_pct == 0.0
    message_count = result.measurements[f"{camera_topic}/message_count"]
    decoded_frame_count = result.measurements[f"{camera_topic}/decoded_frame_count"]
    assert message_count == decoded_frame_count
    assert result.measurements[f"{camera_topic}/decode_deficit_pct"] == pytest.approx(0.0)
    # No dropped frames were injected: the stored count matches the rate.
    assert result.measurements[f"{camera_topic}/frame_deficit_pct"] == pytest.approx(0.0)
    assert result.measurements[f"{camera_topic}/expected_frame_count"] == message_count
    # Which binary measured this: a pinned-build bump moves readings without
    # moving the check's version, so the instrument names itself in the row.
    instrument = result.measurements["camera_instrument"]
    assert isinstance(instrument, str) and "ffmpeg version" in instrument


def test_media_digest_matches_the_same_footage_under_different_telemetry(
    tmp_path: Path,
) -> None:
    """The case content_digest cannot see: identical footage, everything else
    different. Camera frames depend only on the camera's own spec fields, so
    varying the seed changes the joint stream and leaves the footage alone.
    """
    first = synthesize_episode(
        tmp_path / "first.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",), seed=0),
    )
    second = synthesize_episode(
        tmp_path / "second.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",), seed=7),
    )
    with hflow.Episode(first) as episode:
        camera_topic = episode.cameras[0]
        first_media = media_digest(episode).measurements[f"{camera_topic}/media_digest"]
        first_content = content_digest(episode).measurements["content_digest"]
        media_bytes = media_digest(episode).measurements[f"{camera_topic}/media_bytes"]
    with hflow.Episode(second) as episode:
        second_media = media_digest(episode).measurements[f"{camera_topic}/media_digest"]
        second_content = content_digest(episode).measurements["content_digest"]

    assert first_media == second_media, "same footage must digest the same"
    assert first_content != second_content, "differing telemetry must change content_digest"
    assert isinstance(media_bytes, int) and media_bytes > 0


def test_media_digest_separates_different_footage(tmp_path: Path) -> None:
    plain = synthesize_episode(
        tmp_path / "plain.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",), black_segment=None),
    )
    blacked_out = synthesize_episode(
        tmp_path / "blacked.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",), black_segment=(0.5, 1.5)),
    )
    digests = []
    for source in (plain, blacked_out):
        with hflow.Episode(source) as episode:
            camera_topic = episode.cameras[0]
            digests.append(media_digest(episode).measurements[f"{camera_topic}/media_digest"])
    assert digests[0] != digests[1]


def test_keyframe_interval_reports_the_encoders_gop(tmp_path: Path) -> None:
    """The canonical encoder writes a keyframe every gop_seconds, so the
    measured cadence is the writer's own contract read back off the stream.
    """
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=4.0, cameras=("wrist_cam",), black_segment=None),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        result = keyframe_interval(episode)

    keyframe_count = result.measurements[f"{camera_topic}/keyframe_count"]
    assert isinstance(keyframe_count, int) and keyframe_count >= 2
    assert result.measurements[f"{camera_topic}/first_frame_is_keyframe"] == 1
    max_gap_s = result.measurements[f"{camera_topic}/max_keyframe_gap_s"]
    assert isinstance(max_gap_s, float)
    # The VLA preset targets 1 s GOPs; allow the fps-estimate rounding the
    # encoder applies when converting seconds to a frame count.
    assert 0.5 < max_gap_s < 1.6
    assert result.verdict is None


def test_fps_conformance_classifies_matching_and_half_rate_streams(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",), image_hz=10.0),
    )
    with hflow.Episode(source) as episode:
        camera_topic = episode.cameras[0]
        matching = camera_fps_conformance(episode, nominal_fps={camera_topic: 10})
        # The corpus says 5 Hz; this stream ran at twice that.
        doubled = camera_fps_conformance(episode, nominal_fps={camera_topic: 5})
        implausible = camera_fps_conformance(episode, nominal_fps={camera_topic: 300})
        undeclared = camera_fps_conformance(episode)

    assert matching.measurements[f"{camera_topic}/derived_fps"] == 10
    assert matching.measurements[f"{camera_topic}/fps_resolution"] == "matches-nominal"
    assert matching.measurements[f"{camera_topic}/fps_ratio"] == pytest.approx(1.0)
    assert doubled.measurements[f"{camera_topic}/fps_resolution"] == "downsample-2x"
    assert doubled.measurements[f"{camera_topic}/fps_ratio"] == pytest.approx(2.0)
    assert implausible.measurements[f"{camera_topic}/fps_resolution"] == "unrecoverable"
    assert undeclared.measurements[f"{camera_topic}/fps_resolution"] == "no-nominal-declared"


def test_action_integrity_finds_the_injected_frozen_run(tmp_path: Path) -> None:
    """A stalled publisher repeats samples bit-for-bit; a still robot does not.
    The fixture holds every joint for 1 s of a 4 s stream.
    """
    source = synthesize_episode(
        tmp_path / "frozen.mcap",
        SyntheticEpisodeSpec(
            duration_s=4.0,
            cameras=(),
            joint_jump_at_s=None,
            joint_freeze_segment=(1.0, 2.0),
        ),
    )
    with hflow.Episode(source) as episode:
        result = action_integrity(episode)

    assert result.measurements["/joint_states/nan_count"] == 0
    assert result.measurements["/joint_states/inf_count"] == 0
    assert result.measurements["/joint_states/frozen_run_count"] == 1
    longest_run_fraction = result.measurements["/joint_states/frozen_longest_run_fraction"]
    assert isinstance(longest_run_fraction, float)
    # 1 s held out of 4 s recorded.
    assert longest_run_fraction == pytest.approx(0.25, abs=0.02)

    frozen_intervals = [i for i in result.intervals if i.label == "frozen:/joint_states"]
    assert len(frozen_intervals) == 1
    held_duration_s = (frozen_intervals[0].end_ns - frozen_intervals[0].start_ns) / 1e9
    assert held_duration_s == pytest.approx(1.0, abs=0.05)
    assert result.verdict is None


def test_action_integrity_reports_a_clean_stream_as_clean(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "clean.mcap",
        SyntheticEpisodeSpec(duration_s=3.0, cameras=(), joint_jump_at_s=None),
    )
    with hflow.Episode(source) as episode:
        result = action_integrity(episode)

    assert result.measurements["/joint_states/nan_count"] == 0
    assert result.measurements["/joint_states/frozen_run_count"] == 0
    assert result.measurements["/joint_states/frozen_step_pct"] == 0.0
    assert result.measurements["/joint_states/unchanged_dimension_count"] == 0
    assert result.intervals == []


def test_camera_signal_quality_measures_range_exposure_and_stillness(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=3.0, cameras=("wrist_cam",), black_segment=None),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        result = camera_signal_quality(episode)

    # The coding range is a measured fact, recorded with the evidence behind it.
    full_range = result.measurements[f"{camera_topic}/full_range_detected"]
    luma_min = result.measurements[f"{camera_topic}/luma_min"]
    luma_max = result.measurements[f"{camera_topic}/luma_max"]
    assert full_range in (0, 1)
    assert isinstance(luma_min, float) and isinstance(luma_max, float)
    assert 0.0 <= luma_min <= luma_max <= 255.0
    # The verdict must be consistent with its own evidence.
    assert bool(full_range) == (luma_min < 16.0 or luma_max > 235.0)

    # The test pattern moves, so stillness is well above zero.
    frame_difference_mean = result.measurements[f"{camera_topic}/frame_difference_mean"]
    assert isinstance(frame_difference_mean, float) and frame_difference_mean > 0.0

    # A clean synthetic encode carries no impulse noise.
    assert result.measurements[f"{camera_topic}/temporal_outlier_max"] == pytest.approx(
        0.0, abs=1e-3
    )
    assert result.verdict is None


def test_camera_signal_quality_sees_a_blacked_out_segment(tmp_path: Path) -> None:
    """Assertions here are deliberately build-robust.

    The suite runs against whatever ffmpeg is on PATH (see tests/conftest.py),
    and builds genuinely disagree about absolute luma: on this same fixture,
    ffmpeg 6.1 range-scales the black frames to a floor of 8 while the pinned
    8.1 preserves 0. That moves every exposure percentile and so the
    range-gated shares with it. The gate arithmetic is pinned exactly on
    synthetic instrument text in tests/test_ffmpeg.py; what must hold on any
    build is the threshold-based evidence and the ordering between signals.
    """
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=4.0, cameras=("wrist_cam",), black_segment=(1.0, 3.0)),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    with hflow.Episode(canonical) as episode:
        camera_topic = episode.cameras[0]
        result = camera_signal_quality(episode)

    # Black pixels are counted against a fixed threshold of 17, which both
    # builds' black frames sit under, so the share is stable across them.
    black_share_max = result.measurements[f"{camera_topic}/black_pixel_share_max"]
    black_share_mean = result.measurements[f"{camera_topic}/black_pixel_share_mean"]
    assert isinstance(black_share_max, float) and black_share_max > 90.0
    # 2 s of 4 s blacked out; decode boundaries make the exact count fuzzy.
    assert isinstance(black_share_mean, float) and 25.0 < black_share_mean < 75.0

    # Half the clip near-black must drag the low percentile far below the high
    # one, whatever scale the decoder put them on.
    luma_p10_mean = result.measurements[f"{camera_topic}/luma_p10_mean"]
    luma_p90_mean = result.measurements[f"{camera_topic}/luma_p90_mean"]
    assert isinstance(luma_p10_mean, float) and isinstance(luma_p90_mean, float)
    assert luma_p10_mean < luma_p90_mean / 2.0


def test_registering_both_camera_checks_caches_the_instrument(tmp_path: Path) -> None:
    """Issue #173: registering ``camera_frame_stats`` AND
    ``camera_signal_quality`` against the same episode must run the ffmpeg
    instrument pass once per camera, not twice. The cache lives beside the
    workdir MP4 ``Episode.video()`` produces, exactly the same place the
    MP4 remux cache lives, and the second check reads it without invoking
    ffmpeg again.
    """
    source = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=("wrist_cam",)),
    )
    canonical = tmp_path / "episode.canonical.mcap"
    write_canonical_episode(source, canonical, TransformConfig())
    workdir = tmp_path / "workdir"

    with hflow.Episode(canonical, workdir=workdir) as episode:
        camera_topic = episode.cameras[0]
        first = camera_frame_stats(episode)
        video_path = episode.video(camera_topic)
        cache_files_after_first_check = list(workdir.glob(f"{video_path.stem}.instrument.*.txt"))
        assert len(cache_files_after_first_check) == 1

        # If the second check attempts another decode, this no longer-valid MP4
        # makes it fail. A successful result proves it read the cached stream.
        video_path.write_bytes(b"not a video anymore")
        second = camera_signal_quality(episode)

    assert list(workdir.glob(f"{video_path.stem}.instrument.*.txt")) == (
        cache_files_after_first_check
    )
    # Both checks produced measurements for the same camera.
    assert f"{camera_topic}/decoded_frame_count" in first.measurements
    assert f"{camera_topic}/signal_frame_count" in second.measurements


def test_trajectory_metrics_finds_the_injected_hold(tmp_path: Path) -> None:
    """A held publisher is motionless, and the fraction is time-weighted over
    the span actually measured rather than the episode span.
    """
    source = synthesize_episode(
        tmp_path / "held.mcap",
        SyntheticEpisodeSpec(
            duration_s=4.0,
            cameras=(),
            joint_jump_at_s=None,
            joint_freeze_segment=(1.0, 2.0),
        ),
    )
    with hflow.Episode(source) as episode:
        result = trajectory_metrics(episode)

    motionless_fraction = result.measurements["/joint_states/motionless_fraction"]
    assert isinstance(motionless_fraction, float)
    # 1 s held out of 4 s measured.
    assert motionless_fraction == pytest.approx(0.25, abs=0.03)
    assert result.measurements["/joint_states/non_finite_sample_count"] == 0
    assert result.measurements["/joint_states/scale_source"] == "raw"
    peak = result.measurements["/joint_states/peak_velocity"]
    mean = result.measurements["/joint_states/mean_velocity"]
    assert isinstance(peak, float) and isinstance(mean, float) and peak >= mean > 0.0
    assert result.verdict is None


def test_trajectory_metrics_moving_stream_is_not_motionless(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "moving.mcap",
        SyntheticEpisodeSpec(duration_s=3.0, cameras=(), joint_jump_at_s=None),
    )
    with hflow.Episode(source) as episode:
        result = trajectory_metrics(episode)
    assert result.measurements["/joint_states/motionless_fraction"] == pytest.approx(0.0)


def test_trajectory_metrics_dimension_scales_are_recorded_and_validated(
    tmp_path: Path,
) -> None:
    """Scaling is opt-in and its provenance is recorded, because a normalized
    number and a raw one are not comparable and nothing else would say which.
    """
    source = synthesize_episode(
        tmp_path / "scaled.mcap",
        SyntheticEpisodeSpec(duration_s=2.0, cameras=(), joint_count=7),
    )
    with hflow.Episode(source) as episode:
        raw = trajectory_metrics(episode)
        scaled = trajectory_metrics(episode, dimension_scales=[2.0] * 7)
        with pytest.raises(ValueError, match="dimension_scales has 3 entries"):
            trajectory_metrics(episode, dimension_scales=[1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="finite and positive"):
            trajectory_metrics(episode, dimension_scales=[0.0] * 7)

    assert scaled.measurements["/joint_states/scale_source"] == "user"
    raw_peak = raw.measurements["/joint_states/peak_velocity"]
    scaled_peak = scaled.measurements["/joint_states/peak_velocity"]
    assert isinstance(raw_peak, float) and isinstance(scaled_peak, float)
    # Halving every dimension halves the magnitude over all of them.
    assert scaled_peak == pytest.approx(raw_peak / 2.0)


def test_trajectory_segments_localizes_the_hold(tmp_path: Path) -> None:
    source = synthesize_episode(
        tmp_path / "held.mcap",
        SyntheticEpisodeSpec(
            duration_s=4.0,
            cameras=(),
            joint_jump_at_s=None,
            joint_freeze_segment=(1.0, 2.0),
        ),
    )
    with hflow.Episode(source) as episode:
        result = trajectory_segments(episode)

    motionless = [i for i in result.intervals if i.label == "motionless:/joint_states"]
    assert len(motionless) == 1
    held_s = (motionless[0].end_ns - motionless[0].start_ns) / 1e9
    assert held_s == pytest.approx(1.0, abs=0.05)
    assert result.measurements["/joint_states/motionless_span_count"] == 1
    assert result.verdict is None


def test_trajectory_segments_flags_the_injected_jump_as_a_change(tmp_path: Path) -> None:
    """The jump is a step discontinuity, so it is the sharpest curvature in the
    episode and must clear a threshold derived from the episode's own spread.
    """
    source = synthesize_episode(
        tmp_path / "jump.mcap",
        SyntheticEpisodeSpec(duration_s=4.0, cameras=(), joint_jump_at_s=2.0),
    )
    with hflow.Episode(source) as episode:
        result = trajectory_segments(episode)

    changes = [i for i in result.intervals if i.label == "trajectory_change:/joint_states"]
    assert changes, "the injected jump must register as a trajectory change"
    jump_ns = min(changes, key=lambda i: i.start_ns)
    episode_start_ns = int(episode.channel("/joint_states").timestamps[0])
    assert (jump_ns.start_ns - episode_start_ns) / 1e9 == pytest.approx(2.0, abs=0.1)
    # A curvature sample spans three stamps, so a reported span is never empty.
    assert all(i.end_ns > i.start_ns for i in changes)

    # The episode maximum is owned by trajectory_metrics, so read it there
    # rather than duplicating the key across two checks.
    threshold = result.measurements["/joint_states/trajectory_change_threshold"]
    with hflow.Episode(source) as episode:
        max_change = trajectory_metrics(episode).measurements["/joint_states/max_trajectory_change"]
    assert isinstance(threshold, float) and isinstance(max_change, float)
    assert max_change > threshold


def test_trajectory_change_threshold_uses_a_true_weighted_median(tmp_path: Path) -> None:
    """The threshold must come from a value a sample actually took. An
    interpolating quantile invents one, which shifts which spans flag.
    """
    from hflow.checks import _duration_weighted_median

    values = np.array([1.0, 2.0, 100.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert _duration_weighted_median(values, weights) == 2.0
    # Weight concentrated on the smallest value moves the median onto it.
    assert _duration_weighted_median(values, np.array([10.0, 1.0, 1.0])) == 1.0


def test_trajectory_metrics_emits_unsettled_ratio_for_a_moving_episode(
    tmp_path: Path,
) -> None:
    """A moving episode has mean_velocity > 0, so the unsettled ratio is
    defined and must be emitted alongside final_pose_speed.
    """
    source = synthesize_episode(
        tmp_path / "moving.mcap",
        SyntheticEpisodeSpec(duration_s=3.0, cameras=(), joint_jump_at_s=None),
    )
    with hflow.Episode(source) as episode:
        result = trajectory_metrics(episode)

    assert "/joint_states/final_pose_speed" in result.measurements
    assert "/joint_states/final_pose_unsettled_ratio" in result.measurements
    ratio = result.measurements["/joint_states/final_pose_unsettled_ratio"]
    assert isinstance(ratio, float)


def test_trajectory_metrics_omits_unsettled_ratio_for_a_motionless_episode(
    tmp_path: Path,
) -> None:
    """A fully frozen episode has mean_velocity == 0.0, so dividing by it
    is undefined. final_pose_speed is still recorded; the ratio is not.
    """
    source = synthesize_episode(
        tmp_path / "frozen.mcap",
        SyntheticEpisodeSpec(
            duration_s=3.0,
            cameras=(),
            joint_jump_at_s=None,
            joint_freeze_segment=(0.0, 3.0),
        ),
    )
    with hflow.Episode(source) as episode:
        result = trajectory_metrics(episode)

    assert "/joint_states/final_pose_speed" in result.measurements
    assert "/joint_states/final_pose_unsettled_ratio" not in result.measurements
