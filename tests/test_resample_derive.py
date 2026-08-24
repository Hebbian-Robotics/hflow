"""Grid resampling, derived signals, @app.transform, and camera-less ffmpeg provenance."""

import json
from pathlib import Path

import numpy as np
import pytest
from mcap.reader import make_reader

from hflow.app import App
from hflow.episode import ChannelData, Episode
from hflow.ffmpeg import FFMPEG_ENV_VAR, ffmpeg_path, ffmpeg_version
from hflow.format import (
    DERIVED_SCHEMA_NAME,
    FFMPEG_VERSION_NOT_USED,
    METADATA_RECORD_PROVENANCE,
    PROVENANCE_KEY_DERIVED_PREFIX,
    PROVENANCE_KEY_RESAMPLE_POLICY_VERSION,
    RESAMPLE_POLICY_VERSION,
)
from hflow.reader import TopicInfo
from hflow.resample import DerivedSeries, to_grid
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import (
    EpisodeStamps,
    TransformConfig,
    compute_pipeline_version,
    write_canonical_episode,
)

# A camera-less source: exercises derived signals over state channels quickly
# AND the "no video work -> no ffmpeg resolution" provenance rule.
CAMERALESS_SPEC = SyntheticEpisodeSpec(
    duration_s=2.0,
    cameras=(),
    joint_hz=50.0,
    black_segment=None,
    joint_jump_at_s=None,
    timestamp_offset_segment=None,
)


@pytest.fixture(scope="module")
def cameraless_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(
        tmp_path_factory.mktemp("cameraless") / "episode.mcap", CAMERALESS_SPEC
    )


def make_json_channel(
    timestamps_ns: list[int], payload_objects: list[dict[str, object]]
) -> ChannelData:
    """A ChannelData constructed directly (no file): JSON-decoded messages."""
    raw_payloads = [json.dumps(payload_object).encode() for payload_object in payload_objects]
    log_times = np.asarray(timestamps_ns, dtype=np.int64)
    info = TopicInfo(
        topic="/synthetic",
        channel_id=1,
        schema_name="",
        schema_encoding="",
        message_encoding="json",
        message_count=len(raw_payloads),
        schema_data=b"",
        has_schema=False,
    )
    return ChannelData(
        topic="/synthetic",
        channel_id=1,
        info=info,
        log_times=log_times,
        publish_times=log_times.copy(),
        raw=raw_payloads,
        decoder=lambda payload: json.loads(payload.decode()),
    )


def ramp_channel() -> ChannelData:
    """Scalar ramp: value 0/10/20 at 1.25s/2.25s/3.25s (epoch offsets in ns)."""
    return make_json_channel(
        [1_250_000_000, 2_250_000_000, 3_250_000_000],
        [{"value": 0.0}, {"value": 10.0}, {"value": 20.0}],
    )


def test_to_grid_interpolate_linear_ramp() -> None:
    series = to_grid(ramp_channel(), 1.0, policy="interpolate")
    assert series.timestamps_ns.tolist() == [2_000_000_000, 3_000_000_000]
    np.testing.assert_allclose(series.values["value"], [7.5, 17.5])


def test_to_grid_nearest_selects_closest_sample() -> None:
    series = to_grid(ramp_channel(), 1.0, policy="nearest")
    assert series.timestamps_ns.tolist() == [2_000_000_000, 3_000_000_000]
    # 2.0s is 0.75s from the 1.25s sample and 0.25s from the 2.25s sample.
    np.testing.assert_array_equal(series.values["value"], [10.0, 20.0])


def test_to_grid_grids_are_epoch_aligned_across_channels() -> None:
    """Two channels starting at different times share grid timestamps."""
    early_channel = make_json_channel(
        [1_100_000_000, 2_100_000_000, 3_100_000_000],
        [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}],
    )
    late_channel = make_json_channel(
        [1_600_000_000, 2_600_000_000, 3_600_000_000],
        [{"value": 4.0}, {"value": 5.0}, {"value": 6.0}],
    )
    step_ns = 500_000_000  # 2 Hz
    early_series = to_grid(early_channel, 2.0, policy="interpolate")
    late_series = to_grid(late_channel, 2.0, policy="interpolate")
    assert np.all(early_series.timestamps_ns % step_ns == 0)
    assert np.all(late_series.timestamps_ns % step_ns == 0)
    overlap = set(early_series.timestamps_ns.tolist()) & set(late_series.timestamps_ns.tolist())
    assert overlap, "same-rate grids must share timestamps on overlapping spans"


def test_to_grid_vector_field_becomes_named_columns() -> None:
    channel = make_json_channel(
        [1_000_000_000, 2_000_000_000, 3_000_000_000],
        [{"position": [0.0, 0.0]}, {"position": [10.0, 100.0]}, {"position": [20.0, 200.0]}],
    )
    series = to_grid(channel, 2.0, policy="interpolate", field="position")
    assert sorted(series.values) == ["position_0", "position_1"]
    assert series.timestamps_ns.tolist() == [
        1_000_000_000,
        1_500_000_000,
        2_000_000_000,
        2_500_000_000,
        3_000_000_000,
    ]
    np.testing.assert_allclose(series.values["position_0"], [0.0, 5.0, 10.0, 15.0, 20.0])
    np.testing.assert_allclose(series.values["position_1"], [0.0, 50.0, 100.0, 150.0, 200.0])


def quaternion_channel(timestamps_ns: list[int], quaternions: list[list[float]]) -> ChannelData:
    return make_json_channel(
        timestamps_ns,
        [{"orientation": quaternion} for quaternion in quaternions],
    )


def quaternion_values(series: DerivedSeries) -> np.ndarray:
    return np.column_stack(
        [series.values[f"orientation_{component_index}"] for component_index in range(4)]
    )


def test_to_grid_slerp_interpolates_180_degree_rotation() -> None:
    assert RESAMPLE_POLICY_VERSION == "2"
    source = quaternion_channel(
        [0, 2_000_000_000],
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    )

    series = to_grid(source, 1.0, policy="slerp", field="orientation")
    quaternions = quaternion_values(series)

    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)
    np.testing.assert_allclose(quaternions[1], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12)
    assert abs(np.dot(quaternions[-1], np.array([0.0, 0.0, 1.0, 0.0]))) == pytest.approx(1.0)


def test_to_grid_slerp_aligns_antipodal_sign_flip() -> None:
    rotation = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    source = quaternion_channel(
        [0, 1_000_000_000, 2_000_000_000],
        [[0.0, 0.0, 0.0, 1.0], rotation.tolist(), (-rotation).tolist()],
    )

    series = to_grid(source, 2.0, policy="slerp", field="orientation")
    quaternions = quaternion_values(series)

    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)
    assert abs(np.dot(quaternions[-1], -rotation)) == pytest.approx(1.0)
    assert abs(np.dot(quaternions[-2], rotation)) == pytest.approx(1.0)


def test_to_grid_slerp_near_parallel_fallback_stays_normalized() -> None:
    tiny_angle = 1e-5
    nearly_identical = [0.0, 0.0, np.sin(tiny_angle / 2.0), np.cos(tiny_angle / 2.0)]
    source = quaternion_channel(
        [0, 2_000_000_000],
        [[0.0, 0.0, 0.0, 2.0], nearly_identical],
    )

    series = to_grid(source, 1.0, policy="slerp", field="orientation")
    quaternions = quaternion_values(series)

    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)
    expected_midpoint = np.array([0.0, 0.0, np.sin(tiny_angle / 4.0), np.cos(tiny_angle / 4.0)])
    assert abs(np.dot(quaternions[1], expected_midpoint)) == pytest.approx(1.0)


def test_to_grid_slerp_rejects_non_quaternion_width() -> None:
    channel = make_json_channel(
        [0, 1_000_000_000],
        [{"rotation": [1.0, 0.0, 0.0]}, {"rotation": [0.0, 1.0, 0.0]}],
    )

    with pytest.raises(ValueError, match="exactly 4 columns, got 3"):
        to_grid(channel, 1.0, policy="slerp", field="rotation")


def test_to_grid_rejects_span_shorter_than_one_step() -> None:
    with pytest.raises(ValueError, match="shorter than one grid step"):
        to_grid(ramp_channel(), 0.01, policy="interpolate")


def test_to_grid_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="rate_hz must be positive"):
        to_grid(ramp_channel(), 0.0, policy="interpolate")


def test_derived_series_validates_column_lengths() -> None:
    with pytest.raises(ValueError, match="2 values"):
        DerivedSeries(
            timestamps_ns=np.asarray([1, 2, 3], dtype=np.int64),
            values={"a": np.asarray([1.0, 2.0])},
        )


def test_derived_channel_lands_in_canonical_file(cameraless_source: Path, tmp_path: Path) -> None:
    with Episode(cameraless_source) as source_episode:
        series = to_grid(
            source_episode.channel("/joint_states"), 10.0, policy="interpolate", field="position"
        )
    derived_topic = "/joints_10hz"
    derived_version = "abcdef123456"
    output = tmp_path / "out.mcap"
    stamps = write_canonical_episode(
        cameraless_source, output, derived=[(derived_topic, series, derived_version)]
    )

    with output.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        assert summary is not None
        derived_mcap_channel = next(
            channel for channel in summary.channels.values() if channel.topic == derived_topic
        )
        schema = summary.schemas[derived_mcap_channel.schema_id]
        assert schema.name == DERIVED_SCHEMA_NAME
        assert schema.encoding == "jsonschema"
        assert derived_mcap_channel.message_encoding == "json"
        decoded_messages = [
            (message.log_time, json.loads(message.data.decode()))
            for _schema, _channel, message in reader.iter_messages(topics=[derived_topic])
        ]
        provenance = {record.name: dict(record.metadata) for record in reader.iter_metadata()}[
            METADATA_RECORD_PROVENANCE
        ]

    assert [log_time for log_time, _payload in decoded_messages] == series.timestamps_ns.tolist()
    first_payload = decoded_messages[0][1]
    assert sorted(first_payload) == sorted(series.values)
    for column_name, column_values in series.values.items():
        assert first_payload[column_name] == pytest.approx(float(column_values[0]))

    assert provenance[PROVENANCE_KEY_RESAMPLE_POLICY_VERSION] == RESAMPLE_POLICY_VERSION
    assert provenance[f"{PROVENANCE_KEY_DERIVED_PREFIX}{derived_topic}"] == derived_version
    # Derived topics+versions are part of the pipeline_version content hash.
    assert stamps.pipeline_version != compute_pipeline_version(TransformConfig())
    assert stamps.pipeline_version == compute_pipeline_version(
        TransformConfig(), {derived_topic: derived_version}
    )


def test_derived_topic_colliding_with_source_topic_errors(
    cameraless_source: Path, tmp_path: Path
) -> None:
    series = DerivedSeries(
        timestamps_ns=np.asarray([1_755_000_000_500_000_000], dtype=np.int64),
        values={"value": np.asarray([1.0])},
    )
    with pytest.raises(ValueError, match="collides with a source"):
        write_canonical_episode(
            cameraless_source, tmp_path / "out.mcap", derived=[("/joint_states", series, "v1")]
        )


def test_cameraless_transform_never_resolves_ffmpeg(
    cameraless_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No camera channels -> ffmpeg is never resolved and provenance says so.

    The override points at a nonexistent path with the resolution caches
    cleared: any attempt to resolve ffmpeg would raise, so a passing transform
    proves no resolution happened.
    """
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(tmp_path / "missing-ffmpeg"))
    ffmpeg_path.cache_clear()
    ffmpeg_version.cache_clear()
    output = tmp_path / "out.mcap"
    try:
        stamps = write_canonical_episode(cameraless_source, output)
    finally:
        # Nothing resolved under the bad override (or the test raised), so
        # clearing lets later tests re-resolve under the restored env.
        ffmpeg_path.cache_clear()
        ffmpeg_version.cache_clear()
    assert stamps.ffmpeg_version == FFMPEG_VERSION_NOT_USED
    with output.open("rb") as stream:
        provenance = {
            record.name: dict(record.metadata) for record in make_reader(stream).iter_metadata()
        }[METADATA_RECORD_PROVENANCE]
    assert provenance["ffmpeg_version"] == FFMPEG_VERSION_NOT_USED


def test_app_derive_end_to_end(cameraless_source: Path, tmp_path: Path) -> None:
    app = App("derive-test", data_root=tmp_path / "data")

    @app.derive("/joints_5hz")
    def joints_5hz(ep: Episode) -> DerivedSeries:
        return to_grid(ep.channel("/joint_states"), 5.0, policy="nearest", field="position")

    report = app.test(cameraless_source, verbose=False)
    assert report.stamps.ffmpeg_version == FFMPEG_VERSION_NOT_USED
    with Episode(report.canonical_path) as canonical_episode:
        derived_channel = canonical_episode.channel("/joints_5hz")
        assert len(derived_channel) > 0
        step_ns = 200_000_000  # 5 Hz grid, epoch-anchored
        assert all(int(timestamp) % step_ns == 0 for timestamp in derived_channel.timestamps)
        first_message = derived_channel.messages[0]
        expected_columns = {f"position_{index}" for index in range(CAMERALESS_SPEC.joint_count)}
        assert set(first_message) == expected_columns
        derived_provenance_version = canonical_episode.metadata[
            f"{PROVENANCE_KEY_DERIVED_PREFIX}/joints_5hz"
        ]
        assert len(derived_provenance_version) == 12
        assert (
            canonical_episode.metadata[PROVENANCE_KEY_RESAMPLE_POLICY_VERSION]
            == RESAMPLE_POLICY_VERSION
        )


def test_derive_topic_collision_errors(tmp_path: Path) -> None:
    app = App("collision-test", data_root=tmp_path / "data")

    @app.derive("/gripper_open")
    def first_derivation(ep: Episode) -> DerivedSeries:
        return to_grid(ep.channel("/joint_states"), 5.0, policy="nearest")

    with pytest.raises(ValueError, match="already registered"):

        @app.derive("/gripper_open")
        def second_derivation(ep: Episode) -> DerivedSeries:
            return to_grid(ep.channel("/joint_states"), 10.0, policy="nearest")


def test_transform_override_invoked_and_stamps_propagate(
    cameraless_source: Path, tmp_path: Path
) -> None:
    app = App("override-test", data_root=tmp_path / "data")
    override_calls: list[tuple[Path, Path]] = []

    @app.derive("/never_computed")
    def never_computed(ep: Episode) -> DerivedSeries:
        raise AssertionError("derive functions must not run under a transform override")

    @app.transform
    def custom_transform(source: Path, output: Path, config: TransformConfig) -> EpisodeStamps:
        override_calls.append((source, output))
        # The override contract: still end by calling write_canonical_episode.
        return write_canonical_episode(source, output, config, source_uri="custom://override")

    report = app.test(cameraless_source, verbose=False)
    assert override_calls == [(cameraless_source, report.canonical_path)]
    with Episode(report.canonical_path) as canonical_episode:
        assert canonical_episode.metadata["source_uri"] == "custom://override"
        source_and_derived_topics = {info.topic for info in canonical_episode.channels.values()}
        assert "/never_computed" not in source_and_derived_topics
    assert report.stamps.pipeline_version == compute_pipeline_version(TransformConfig())


def test_second_transform_override_errors(tmp_path: Path) -> None:
    app = App("double-override-test", data_root=tmp_path / "data")

    @app.transform
    def first_override(source: Path, output: Path, config: TransformConfig) -> EpisodeStamps:
        return write_canonical_episode(source, output, config)

    with pytest.raises(ValueError, match="already registered"):

        @app.transform
        def second_override(source: Path, output: Path, config: TransformConfig) -> EpisodeStamps:
            return write_canonical_episode(source, output, config)


def test_interpolate_refuses_quaternion_shaped_fields() -> None:
    """Componentwise lerp corrupts rotations; the guard points to slerp."""
    from typing import cast

    import numpy as np

    from hflow.episode import ChannelData
    from hflow.reader import TopicInfo
    from hflow.resample import to_grid

    class _FakeQuaternionChannel:
        topic = "/imu"
        info = TopicInfo(
            topic="/imu",
            channel_id=1,
            schema_name="sensor_msgs/msg/Imu",
            schema_encoding="ros2msg",
            message_encoding="cdr",
            message_count=3,
            schema_data=b"",
        )
        timestamps = np.array([0, 10**9, 2 * 10**9], dtype=np.int64)

        def to_numpy(self, field: str | None = None) -> np.ndarray:
            return np.array([[0.0, 0.0, 0.0, 1.0]] * 3)

    # Duck-typed test double smuggled past the checker, like untyped user code.
    channel = cast(ChannelData, _FakeQuaternionChannel())
    with pytest.raises(ValueError, match="policy='slerp'"):
        to_grid(channel, 10.0, policy="interpolate", field="orientation")
    # nearest copies whole samples and stays quaternion-safe.
    series = to_grid(channel, 10.0, policy="nearest", field="orientation")
    assert len(series.values) == 4
