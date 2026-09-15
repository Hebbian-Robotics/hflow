"""#546: unmeasurable joint steps must fail CLOSED.

``_joint_motion_profile`` yields NaN velocity where a step has a duplicate
timestamp or a NaN position. A NaN comparison is False, so the old code
counted those steps as compliant and not-idle while their duration stayed in
the percentage denominators: a stream whose every step duplicates its
timestamp reported 0.0 violations, and an all-NaN channel reported
``idle_fraction = 0.0``. These tests pin the measurable-mask discipline:
metrics over measurable steps only, counts and refusal when nothing is
measurable, and no NaN ever leaving a check.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from mcap.writer import Writer

from hflow import Episode
from hflow.checks import idle_fraction, joint_discontinuity
from hflow.format import METADATA_RECORD_EPISODE
from hflow.steps import CheckResult
from hflow.testing import (
    _JOINT_STATE_SCHEMA_TEXT,
    JOINT_STATE_SCHEMA_NAME,
    _encode_joint_state,
)

TOPIC = "/joint_states"
NS_PER_S = 1_000_000_000


def _joint_states_episode(path: Path, stamps_ns: list[int], positions: list[float]) -> Episode:
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="ros2", library="test-546")
        schema_id = writer.register_schema(
            name=JOINT_STATE_SCHEMA_NAME,
            encoding="ros2msg",
            data=_JOINT_STATE_SCHEMA_TEXT.encode("utf-8"),
        )
        channel_id = writer.register_channel(
            topic=TOPIC, message_encoding="cdr", schema_id=schema_id
        )
        writer.add_metadata(
            METADATA_RECORD_EPISODE, {"task": "nan-failopen", "embodiment": "probe"}
        )
        for sequence, (stamp_ns, position) in enumerate(zip(stamps_ns, positions, strict=True)):
            payload = _encode_joint_state(
                stamp_ns,
                "probe",
                ["j0"],
                [position],
                [],
                [],
            )
            writer.add_message(
                channel_id=channel_id,
                log_time=stamp_ns,
                publish_time=stamp_ns,
                data=payload,
                sequence=sequence,
            )
        writer.finish()
    return Episode(path)


def _assert_all_measurements_finite(*results: CheckResult) -> None:
    for result in results:
        for key, value in result.measurements.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{key} left the check non-finite"


def test_duplicate_stamp_stream_refuses_instead_of_reporting_zero_violations(
    tmp_path: Path,
) -> None:
    """Stream A (all duplicate stamps, big position hops) must abstain;
    Stream B (same positions, advancing stamps) reports 100.0. Old code:
    A reported 0.0 -- a gate-wide fail-open."""
    stamps = [0, 0, 0, 0, 0]
    positions = [0.0, 5.0, 10.0, 15.0, 20.0]
    with (
        _joint_states_episode(tmp_path / "a.mcap", stamps, positions) as stream_a,
        _joint_states_episode(
            tmp_path / "b.mcap",
            [i * NS_PER_S for i in range(5)],
            positions,
        ) as stream_b,
    ):
        refused = joint_discontinuity(stream_a, velocity_limit=3.0)
        measured = joint_discontinuity(stream_b, velocity_limit=3.0)
        _assert_all_measurements_finite(refused, measured)
        assert f"{TOPIC}/violation_pct" not in refused.measurements
        assert refused.measurements[f"{TOPIC}/velocity_measurable_step_count"] == 0
        assert refused.measurements[f"{TOPIC}/velocity_sample_count"] == 5
        assert refused.measurements[f"{TOPIC}/nonpositive_dt_count"] == 4
        assert measured.measurements[f"{TOPIC}/violation_pct"] == pytest.approx(100.0)
        assert measured.measurements[f"{TOPIC}/nonpositive_dt_count"] == 0


def test_all_nan_channel_withholds_every_percentage(tmp_path: Path) -> None:
    """An all-NaN position channel is not a 0%-violation, 0%-idle stream;
    both checks must report counts and abstain (#546's dead-channel case).
    Note idle_nonpositive_dt_count == 0: the timestamps advanced, only the
    positions were unmeasurable."""
    stamps = [i * NS_PER_S for i in range(4)]
    with _joint_states_episode(tmp_path / "nan.mcap", stamps, [float("nan")] * 4) as episode:
        velocity = joint_discontinuity(episode, velocity_limit=3.0)
        idle = idle_fraction(episode, velocity_epsilon=0.05)
        _assert_all_measurements_finite(velocity, idle)
        for key in ("violation_pct", "violation_count", "max_abs_velocity"):
            assert f"{TOPIC}/{key}" not in velocity.measurements
        assert velocity.measurements[f"{TOPIC}/velocity_measurable_step_count"] == 0
        assert f"{TOPIC}/idle_fraction" not in idle.measurements
        assert idle.measurements[f"{TOPIC}/idle_sample_count"] == 4
        assert idle.measurements[f"{TOPIC}/idle_measurable_step_count"] == 0
        assert idle.measurements[f"{TOPIC}/idle_nonpositive_dt_count"] == 0


def test_unmeasurable_steps_do_not_dilate_percentage_denominators(
    tmp_path: Path,
) -> None:
    """1 violation and 2 hidden violations among 3 measurable steps must be
    33.3%, not 20% (a revert of the mask in the denominator flips this red).
    The idle case needs no NaN at all to dilute: a step with positive dt but
    NaN positions left the old denominator, under-reporting idle."""
    with (
        _joint_states_episode(
            tmp_path / "mixed_v.mcap",
            [0, 0, NS_PER_S, 2 * NS_PER_S, 2 * NS_PER_S, 3 * NS_PER_S],
            [0.0, 5.0, 10.0, 10.0, 15.0, 15.0],
        ) as velocity_episode,
        _joint_states_episode(
            tmp_path / "mixed_i.mcap",
            [0, NS_PER_S, 2 * NS_PER_S],
            [5.0, 5.0, float("nan")],
        ) as idle_episode,
    ):
        velocity = joint_discontinuity(velocity_episode, velocity_limit=3.0)
        assert velocity.measurements[f"{TOPIC}/violation_pct"] == pytest.approx(100.0 / 3.0)
        assert velocity.measurements[f"{TOPIC}/velocity_measurable_step_count"] == 3
        assert velocity.measurements[f"{TOPIC}/nonpositive_dt_count"] == 2
        idle = idle_fraction(idle_episode, velocity_epsilon=0.05)
        assert idle.measurements[f"{TOPIC}/idle_fraction"] == pytest.approx(1.0)
        assert idle.measurements[f"{TOPIC}/idle_measurable_step_count"] == 1
        _assert_all_measurements_finite(velocity, idle)


def test_every_percentage_names_the_time_it_could_not_measure(
    tmp_path: Path,
) -> None:
    """#546's visibility rule: wherever this profile feeds a percentage, the
    unmeasurable-step counts ride along on the same row."""
    with _joint_states_episode(
        tmp_path / "mixed.mcap",
        [0, 0, NS_PER_S],
        [0.0, 5.0, 5.0],
    ) as episode:
        velocity = joint_discontinuity(episode, velocity_limit=3.0)
        idle = idle_fraction(episode, velocity_epsilon=0.05)
        assert velocity.measurements[f"{TOPIC}/nonpositive_dt_count"] == 1
        assert f"{TOPIC}/violation_pct" in velocity.measurements
        assert idle.measurements[f"{TOPIC}/idle_nonpositive_dt_count"] == 1
        assert f"{TOPIC}/idle_fraction" in idle.measurements
        _assert_all_measurements_finite(velocity, idle)
