"""Round-trip tests for the synthetic episode generator.

The fixture MCAP is decoded with the stock ``mcap`` + ``mcap_ros2`` readers so
these tests referee the hand-encoded CDR, the ros2msg schema texts, and the
deterministic content formulas.
"""

import errno
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import pytest
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from hflow.ffmpeg import ffmpeg_path
from hflow.testing import (
    JOINT_SINE_MAX_FREQUENCY_HZ,
    JOINT_SINE_MIN_FREQUENCY_HZ,
    SyntheticEpisodeSpec,
    VideoEpisodeSpec,
    joint_sine_parameters,
    synthesize_episode,
    write_video_episode,
)

DEFAULT_SPEC = SyntheticEpisodeSpec()

# (log_time_ns, decoded_message) per topic, in log-time order.
DecodedMessagesByTopic = dict[str, list[tuple[int, Any]]]


@pytest.fixture(scope="module")
def default_episode_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(tmp_path_factory.mktemp("synthetic") / "episode.mcap")


@pytest.fixture(scope="module")
def decoded_messages_by_topic(default_episode_path: Path) -> DecodedMessagesByTopic:
    messages_by_topic: DecodedMessagesByTopic = {}
    with default_episode_path.open("rb") as episode_stream:
        reader = make_reader(episode_stream, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded_message in reader.iter_decoded_messages():
            messages_by_topic.setdefault(channel.topic, []).append(
                (message.log_time, decoded_message)
            )
    return messages_by_topic


def expected_log_time_ns(message_index: int, rate_hz: float) -> int:
    return DEFAULT_SPEC.start_time_ns + round(message_index * 1_000_000_000 / rate_hz)


def expected_joint_position(joint_index: int, phase_rad: float, time_s: float) -> float:
    """Mirror of the generator's documented formula, recomputed independently."""
    frequency_hz = 0.2 + 0.3 * joint_index / (DEFAULT_SPEC.joint_count - 1)
    position_rad = 0.5 * math.sin(2.0 * math.pi * frequency_hz * time_s + phase_rad)
    assert DEFAULT_SPEC.joint_jump_at_s is not None
    if time_s >= DEFAULT_SPEC.joint_jump_at_s:
        position_rad += DEFAULT_SPEC.joint_jump_rad
    return position_rad


def test_joint_sine_parameters_are_deterministic_and_span_frequencies() -> None:
    joint_count = 5
    first_parameters = joint_sine_parameters(SyntheticEpisodeSpec(seed=42, joint_count=joint_count))
    second_parameters = joint_sine_parameters(
        SyntheticEpisodeSpec(seed=42, joint_count=joint_count, duration_s=1.0, cameras=())
    )

    assert first_parameters == second_parameters

    frequencies_hz = [frequency_hz for frequency_hz, _phase_rad in first_parameters]
    expected_frequency_step_hz = (JOINT_SINE_MAX_FREQUENCY_HZ - JOINT_SINE_MIN_FREQUENCY_HZ) / (
        joint_count - 1
    )
    assert frequencies_hz[0] == JOINT_SINE_MIN_FREQUENCY_HZ
    assert frequencies_hz[-1] == JOINT_SINE_MAX_FREQUENCY_HZ
    assert [
        frequencies_hz[index + 1] - frequencies_hz[index] for index in range(joint_count - 1)
    ] == pytest.approx([expected_frequency_step_hz] * (joint_count - 1))
    assert all(0.0 <= phase_rad < 2.0 * math.pi for _frequency_hz, phase_rad in first_parameters)


def test_joint_sine_parameters_single_joint_uses_minimum_frequency() -> None:
    parameters = joint_sine_parameters(SyntheticEpisodeSpec(seed=42, joint_count=1))

    assert len(parameters) == 1
    frequency_hz, phase_rad = parameters[0]
    assert frequency_hz == JOINT_SINE_MIN_FREQUENCY_HZ
    assert 0.0 <= phase_rad < 2.0 * math.pi


def test_joint_states_round_trip(decoded_messages_by_topic: DecodedMessagesByTopic) -> None:
    joint_messages = decoded_messages_by_topic["/joint_states"]
    expected_message_count = round(DEFAULT_SPEC.duration_s * DEFAULT_SPEC.joint_hz)
    assert len(joint_messages) == expected_message_count

    phase_rng = random.Random(DEFAULT_SPEC.seed)
    phases_rad = [phase_rng.uniform(0.0, 2.0 * math.pi) for _ in range(DEFAULT_SPEC.joint_count)]

    assert DEFAULT_SPEC.joint_jump_at_s is not None
    jump_message_index = math.ceil(DEFAULT_SPEC.joint_jump_at_s * DEFAULT_SPEC.joint_hz)
    for message_index, (log_time_ns, joint_state) in enumerate(joint_messages):
        assert log_time_ns == expected_log_time_ns(message_index, DEFAULT_SPEC.joint_hz)
        assert joint_state.header.frame_id == "base"
        stamp_ns = joint_state.header.stamp.sec * 1_000_000_000 + joint_state.header.stamp.nanosec
        assert stamp_ns == log_time_ns
        assert list(joint_state.name) == [
            f"joint_{joint_index}" for joint_index in range(DEFAULT_SPEC.joint_count)
        ]
        time_s = message_index / DEFAULT_SPEC.joint_hz
        expected_positions = [
            expected_joint_position(joint_index, phases_rad[joint_index], time_s)
            for joint_index in range(DEFAULT_SPEC.joint_count)
        ]
        assert list(joint_state.position) == pytest.approx(expected_positions, abs=1e-12)
        # Exactly one step: the sine-subtracted residual is 0 before the jump
        # index and exactly joint_jump_rad from it onward.
        expected_residual = (
            DEFAULT_SPEC.joint_jump_rad if message_index >= jump_message_index else 0.0
        )
        for joint_index, position_rad in enumerate(joint_state.position):
            pure_sine_rad = 0.5 * math.sin(
                2.0 * math.pi * (0.2 + 0.3 * joint_index / (DEFAULT_SPEC.joint_count - 1)) * time_s
                + phases_rad[joint_index]
            )
            assert position_rad - pure_sine_rad == pytest.approx(expected_residual, abs=1e-12)


def test_camera_images_round_trip(decoded_messages_by_topic: DecodedMessagesByTopic) -> None:
    expected_frame_count = round(DEFAULT_SPEC.duration_s * DEFAULT_SPEC.image_hz)
    timestamp_offset_ns = round(DEFAULT_SPEC.timestamp_offset_s * 1_000_000_000)
    assert DEFAULT_SPEC.timestamp_offset_segment is not None
    offset_start_s, offset_end_s = DEFAULT_SPEC.timestamp_offset_segment
    for camera_index, camera_name in enumerate(DEFAULT_SPEC.cameras):
        image_messages = decoded_messages_by_topic[f"/{camera_name}/compressed"]
        assert len(image_messages) == expected_frame_count
        for frame_index, (log_time_ns, compressed_image) in enumerate(image_messages):
            frame_time_s = frame_index / DEFAULT_SPEC.image_hz
            expected_ns = expected_log_time_ns(frame_index, DEFAULT_SPEC.image_hz)
            if camera_index == 0 and offset_start_s <= frame_time_s < offset_end_s:
                expected_ns += timestamp_offset_ns
            assert log_time_ns == expected_ns
            assert compressed_image.header.frame_id == camera_name
            stamp_ns = (
                compressed_image.header.stamp.sec * 1_000_000_000
                + compressed_image.header.stamp.nanosec
            )
            assert stamp_ns == log_time_ns
            assert compressed_image.format == "jpeg"
            jpeg_bytes = bytes(compressed_image.data)
            assert jpeg_bytes[:2] == b"\xff\xd8"
            assert jpeg_bytes[-2:] == b"\xff\xd9"


def test_metadata_and_attachment(default_episode_path: Path) -> None:
    with default_episode_path.open("rb") as episode_stream:
        reader = make_reader(episode_stream)
        metadata_records = {record.name: record.metadata for record in reader.iter_metadata()}
        attachments = list(reader.iter_attachments())
    assert metadata_records["episode/v1"] == {
        "task": DEFAULT_SPEC.task,
        "operator": DEFAULT_SPEC.operator,
        "success": "true",
        "robot_software_version": DEFAULT_SPEC.robot_software_version,
    }
    assert len(attachments) == 1
    calibration_attachment = attachments[0]
    assert calibration_attachment.name == "calibration.json"
    assert calibration_attachment.media_type == "application/json"
    calibration = json.loads(calibration_attachment.data)
    assert set(calibration["cameras"]) == set(DEFAULT_SPEC.cameras)


def test_synthesis_is_deterministic(tmp_path: Path) -> None:
    # Small spec keeps the double generation fast; the faults still fall
    # inside the shortened duration.
    small_spec = SyntheticEpisodeSpec(
        duration_s=2.0,
        image_hz=5.0,
        joint_hz=20.0,
        black_segment=(0.4, 0.8),
        joint_jump_at_s=1.0,
        timestamp_offset_segment=(1.2, 1.6),
    )
    first_path = synthesize_episode(tmp_path / "first.mcap", small_spec)
    second_path = synthesize_episode(tmp_path / "second.mcap", small_spec)
    first_digest = hashlib.sha256(first_path.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second_path.read_bytes()).hexdigest()
    assert first_digest == second_digest


def test_black_segment_frames(decoded_messages_by_topic: DecodedMessagesByTopic) -> None:
    first_camera_topic = f"/{DEFAULT_SPEC.cameras[0]}/compressed"
    image_messages = decoded_messages_by_topic[first_camera_topic]
    assert DEFAULT_SPEC.black_segment is not None
    black_start_s, black_end_s = DEFAULT_SPEC.black_segment
    black_frame_jpegs = [
        bytes(compressed_image.data)
        for frame_index, (_log_time_ns, compressed_image) in enumerate(image_messages)
        if black_start_s <= frame_index / DEFAULT_SPEC.image_hz < black_end_s
    ]
    assert len(black_frame_jpegs) == round((black_end_s - black_start_s) * DEFAULT_SPEC.image_hz)
    assert all(frame_jpeg == black_frame_jpegs[0] for frame_jpeg in black_frame_jpegs)
    first_normal_frame_jpeg = bytes(image_messages[0][1].data)
    assert first_normal_frame_jpeg != black_frame_jpegs[0]


def test_video_episode_adapter_round_trip_and_faults(tmp_path: Path) -> None:
    source_video_path = tmp_path / "source.mp4"
    completed = subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=10",
            "-frames:v",
            "20",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source_video_path),
        ],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")

    output_path = write_video_episode(
        source_video_path,
        tmp_path / "adapted.mcap",
        VideoEpisodeSpec(
            source_start_s=0.2,
            duration_s=1.0,
            image_hz=5.0,
            image_width=160,
            image_height=90,
            black_segment=(0.2, 0.4),
            freeze_segment=(0.6, 1.0),
            metadata=(("source_dataset", "test-video"),),
        ),
    )

    with output_path.open("rb") as episode_stream:
        reader = make_reader(episode_stream, decoder_factories=[DecoderFactory()])
        decoded_messages = list(reader.iter_decoded_messages())
        metadata_records = {record.name: record.metadata for record in reader.iter_metadata()}

    assert len(decoded_messages) == 5
    assert {channel.topic for _schema, channel, _message, _decoded in decoded_messages} == {
        "/head_camera/compressed"
    }
    decoded_frame_jpegs = [
        bytes(decoded_message.data)
        for _schema, _channel, _message, decoded_message in decoded_messages
    ]
    assert decoded_frame_jpegs[0] != decoded_frame_jpegs[1]
    assert decoded_frame_jpegs[1] != decoded_frame_jpegs[2]
    assert decoded_frame_jpegs[3] == decoded_frame_jpegs[4]
    assert metadata_records["episode/v1"]["source_dataset"] == "test-video"


def test_video_episode_missing_source_names_the_path_with_errno(tmp_path: Path) -> None:
    missing_source = tmp_path / "absent.mp4"
    with pytest.raises(FileNotFoundError) as excinfo:
        write_video_episode(missing_source, tmp_path / "adapted.mcap")
    assert excinfo.value.errno == errno.ENOENT
    assert excinfo.value.filename == str(missing_source)
    assert "No such file or directory" in str(excinfo.value)
    assert str(missing_source) in str(excinfo.value)


@pytest.mark.parametrize(
    ("spec_kwargs", "match"),
    [
        pytest.param(
            {"source_start_s": True},
            r"source_start_s must be an int or float, got bool",
            id="source_start_s-bool",
        ),
        pytest.param(
            {"source_start_s": float("nan")},
            r"source_start_s must be finite",
            id="source_start_s-nan",
        ),
        pytest.param(
            {"source_start_s": -1.0},
            r"source_start_s must be >= 0",
            id="source_start_s-negative",
        ),
        pytest.param(
            {"duration_s": True},
            r"duration_s must be an int or float, got bool",
            id="duration_s-bool",
        ),
        pytest.param(
            {"duration_s": float("nan")},
            r"duration_s must be finite",
            id="duration_s-nan",
        ),
        pytest.param({"duration_s": 0.0}, r"duration_s must be > 0", id="duration_s-zero"),
        pytest.param(
            {"image_hz": True},
            r"image_hz must be an int or float, got bool",
            id="image_hz-bool",
        ),
        pytest.param({"image_hz": float("inf")}, r"image_hz must be finite", id="image_hz-inf"),
        pytest.param({"image_hz": 0.0}, r"image_hz must be > 0", id="image_hz-zero"),
        pytest.param(
            {"image_width": True},
            r"image_width must be an int, got bool",
            id="image_width-bool",
        ),
        pytest.param(
            {"image_width": float("inf")},
            r"image_width must be an int, got float",
            id="image_width-inf",
        ),
        pytest.param(
            {"image_width": 640.0},
            r"image_width must be an int, got float",
            id="image_width-float",
        ),
        pytest.param(
            {"image_width": 0}, r"image dimensions must be positive", id="image_width-zero"
        ),
        pytest.param({"image_width": 639}, r"image dimensions must be even", id="image_width-odd"),
        pytest.param(
            {"image_height": True},
            r"image_height must be an int, got bool",
            id="image_height-bool",
        ),
        pytest.param(
            {"image_height": float("nan")},
            r"image_height must be an int, got float",
            id="image_height-nan",
        ),
        pytest.param(
            {"image_height": 360.0},
            r"image_height must be an int, got float",
            id="image_height-float",
        ),
        pytest.param(
            {"image_height": 0}, r"image dimensions must be positive", id="image_height-zero"
        ),
        pytest.param(
            {"image_height": 359}, r"image dimensions must be even", id="image_height-odd"
        ),
    ],
)
def test_write_video_episode_refuses_invalid_spec(
    tmp_path: Path, spec_kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        write_video_episode(
            tmp_path / "source-is-not-read.mp4",
            tmp_path / "refused.mcap",
            VideoEpisodeSpec(**spec_kwargs),  # ty: ignore
        )


@pytest.mark.parametrize(
    ("spec_kwargs", "match"),
    [
        pytest.param({"duration_s": 0.0}, r"duration_s must be > 0", id="duration_s-zero"),
        pytest.param({"duration_s": -1.0}, r"duration_s must be > 0", id="duration_s-negative"),
        pytest.param({"joint_hz": 0.0}, r"joint_hz must be > 0", id="joint_hz-zero"),
        pytest.param({"image_hz": 0.0}, r"image_hz must be > 0", id="image_hz-zero"),
        pytest.param(
            {"duration_s": float("nan")}, r"duration_s must be finite", id="duration_s-nan"
        ),
        pytest.param({"image_hz": float("inf")}, r"image_hz must be finite", id="image_hz-inf"),
        pytest.param({"joint_hz": float("nan")}, r"joint_hz must be finite", id="joint_hz-nan"),
        pytest.param(
            {"joint_count": True}, r"joint_count must be an int, got bool", id="joint_count-bool"
        ),
        pytest.param(
            {"duration_s": True},
            r"duration_s must be an int or float, got bool",
            id="duration_s-bool",
        ),
        pytest.param(
            {"image_width": True}, r"image_width must be an int, got bool", id="image_width-bool"
        ),
        pytest.param(
            {"image_width": 0}, r"image dimensions must be positive", id="image_width-zero"
        ),
        pytest.param(
            {"image_height": 0}, r"image dimensions must be positive", id="image_height-zero"
        ),
        pytest.param({"joint_count": 0}, r"joint_count must be > 0", id="joint_count-zero"),
        pytest.param(
            {"duration_s": 1.0, "black_segment": (5.0, 9.0)},
            r"black_segment must satisfy 0 <= start < end <= duration_s",
            id="black_segment-past-end",
        ),
        pytest.param(
            {"duration_s": 1.0, "joint_freeze_segment": (5.0, 9.0)},
            r"joint_freeze_segment must satisfy 0 <= start < end <= duration_s",
            id="joint_freeze_segment-past-end",
        ),
        pytest.param(
            {"duration_s": 1.0, "timestamp_offset_segment": (5.0, 9.0)},
            r"timestamp_offset_segment must satisfy 0 <= start < end <= duration_s",
            id="timestamp_offset_segment-past-end",
        ),
    ],
)
def test_synthesize_episode_refuses_invalid_spec(
    tmp_path: Path, spec_kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        synthesize_episode(
            tmp_path / "refused.mcap",
            SyntheticEpisodeSpec(cameras=(), **spec_kwargs),  # ty: ignore
        )


def test_a_default_fault_segment_past_a_shortened_duration_still_writes(tmp_path: Path) -> None:
    """The exemption the rest of the suite stands on.

    ``black_segment`` and ``timestamp_offset_segment`` default to spans sized
    for ``duration_s=8.0``, and most callers shorten the episode without
    clearing them. Validating defaults the way an explicit segment is validated
    refuses specs that have always worked: it fails 78 tests and errors 112
    more. This pins the exemption so tightening it is a deliberate choice with
    a red test, not a tidy-up.
    """
    spec = SyntheticEpisodeSpec(duration_s=1.0, cameras=())

    assert spec.black_segment == (2.0, 3.0)
    assert spec.timestamp_offset_segment == (6.0, 7.0)

    written = synthesize_episode(tmp_path / "short.mcap", spec)

    assert written.is_file()


def test_an_explicit_fault_segment_past_the_end_is_still_refused(tmp_path: Path) -> None:
    """The exemption is scoped to defaults, so a chosen segment stays checked."""
    with pytest.raises(ValueError, match=r"black_segment must satisfy"):
        synthesize_episode(
            tmp_path / "refused.mcap",
            SyntheticEpisodeSpec(duration_s=1.0, cameras=(), black_segment=(2.0, 4.0)),
        )
