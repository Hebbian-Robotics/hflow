"""Direct unit tests for the built-in checks (paths e2e only grazes)."""

from pathlib import Path

import pytest

import hflow
from hflow.checks import (
    action_integrity,
    action_rate,
    camera_fps_conformance,
    camera_frame_stats,
    content_digest,
    episode_duration,
    idle_fraction,
    joint_discontinuity,
    keyframe_interval,
    media_digest,
    timestamp_regularity,
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
            "content_digest": content_digest(episode),
            "media_digest": media_digest(episode),
            "keyframe_interval": keyframe_interval(episode),
            "camera_fps_conformance": camera_fps_conformance(episode),
            "action_integrity": action_integrity(episode),
        }

    producers_by_key: dict[str, list[str]] = {}
    for check_name, result in results_by_check.items():
        for key in result.measurements:
            producers_by_key.setdefault(key, []).append(check_name)
    collisions = {key: names for key, names in producers_by_key.items() if len(names) > 1}
    assert collisions == {}


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
    # No dropped frames were injected: the stored count matches the rate.
    assert result.measurements[f"{camera_topic}/frame_deficit_pct"] == pytest.approx(0.0)
    assert result.measurements[f"{camera_topic}/expected_frame_count"] == message_count


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
