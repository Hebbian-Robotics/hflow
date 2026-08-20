"""Direct unit tests for the built-in checks (paths e2e only grazes)."""

from pathlib import Path

import pytest

import hflow
from hflow.checks import (
    action_rate,
    camera_frame_stats,
    content_digest,
    episode_duration,
    idle_fraction,
    joint_discontinuity,
    required_topics,
    timestamp_regularity,
)
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import TransformConfig, write_canonical_episode


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
    violation_pct = result.measurements["/joint_states/violation_pct"]
    assert isinstance(violation_pct, float)
    assert violation_pct == 100.0


def test_offset_segment_is_flagged_at_tight_tolerance(jittery_episode: hflow.Episode) -> None:
    camera_topic = "/wrist_cam/compressed"
    result = timestamp_regularity(jittery_episode, topics=[camera_topic], tolerance_s=0.001)
    violation_pct = result.measurements[f"{camera_topic}/violation_pct"]
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
    action_rate_hz = result.measurements["action_rate_hz"]
    assert isinstance(action_rate_hz, float)
    # the synthetic joint stream runs at 100 Hz by SyntheticEpisodeSpec default
    assert action_rate_hz == pytest.approx(100.0, abs=0.5)


def test_required_topics_all_present(jittery_episode: hflow.Episode) -> None:
    result = required_topics(
        jittery_episode, topics=["/joint_states", "/wrist_cam/compressed"]
    ).measurements
    assert result["/joint_states/present"] is True
    assert result["/wrist_cam/compressed/present"] is True
    joint_states_count = result["/joint_states/message_count"]
    wrist_cam_count = result["/wrist_cam/compressed/message_count"]
    assert isinstance(joint_states_count, int) and joint_states_count > 0
    assert isinstance(wrist_cam_count, int) and wrist_cam_count > 0
    assert result["missing_topic_count"] == 0


def test_required_topics_reports_a_missing_topic(jittery_episode: hflow.Episode) -> None:
    result = required_topics(
        jittery_episode, topics=["/joint_states", "/no_such_topic"]
    ).measurements
    assert result["/joint_states/present"] is True
    assert result["/no_such_topic/present"] is False
    assert result["/no_such_topic/message_count"] == 0
    assert result["missing_topic_count"] == 1


def test_content_digest_identifies_duplicate_content(tmp_path: Path) -> None:
    spec = SyntheticEpisodeSpec(duration_s=2.0)
    first = synthesize_episode(tmp_path / "a.mcap", spec)
    duplicate = synthesize_episode(tmp_path / "b.mcap", spec)
    different = synthesize_episode(
        tmp_path / "c.mcap", SyntheticEpisodeSpec(duration_s=2.0, joint_jump_at_s=1.0)
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
