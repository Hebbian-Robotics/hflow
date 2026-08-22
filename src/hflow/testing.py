"""Synthesize a small, fully redistributable input episode for tests and demos.

Produces an *input-shaped* MCAP (what a robot would record): default
chunking, ROS 2 CDR channels -- deliberately NOT canonical, so it exercises
the transform. Contents:

- One ``sensor_msgs/msg/JointState`` channel (``/joint_states``) at
  ``joint_hz``: per-joint sine waves with a deterministic discontinuity
  injected at ``joint_jump_at_s`` (exercises the joint check).
- One ``sensor_msgs/msg/CompressedImage`` JPEG channel per camera at
  ``image_hz`` (topics ``/<name>/compressed``), frames generated with ffmpeg
  test sources (a different pattern per camera). The first camera gets a
  black segment (exercises the blackout instrument) and a +3 ms timestamp
  offset segment (exercises the timestamp check).
- An ``episode/v1`` Metadata record (task, operator, success,
  robot_software_version) and one small ``calibration.json`` Attachment
  (exercise metadata/attachment passthrough).

Implementation notes:

- Written with the stock ``mcap.writer.Writer`` (this module must not depend
  on the canonical writer -- the fixture is its test input, and input files
  come from arbitrary recorders).
- CDR payloads are hand-encoded (little-endian XCDR1: 4-byte encapsulation
  header ``00 01 00 00``, 4/8-byte alignment relative to the byte after the
  header, strings length-prefixed and NUL-terminated, sequences
  count-prefixed). Correctness is pinned by round-trip decoding with
  ``mcap_ros2.decoder`` in tests.
- Schema records carry rosbag2-style concatenated message definitions
  (``ros2msg`` encoding, dependencies separated by ``===`` lines with
  ``MSG: pkg/msg/Type`` headers) so any ROS 2-aware reader decodes them.
- Everything is deterministic: fixed ``start_time_ns``, seeded noise, no
  wall-clock reads.
"""

import json
import math
import random
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mcap.writer import Writer

from hflow.ffmpeg import ffmpeg_path
from hflow.format import (
    EPISODE_KEY_ROBOT_SOFTWARE_VERSION,
    METADATA_RECORD_EPISODE,
    NANOSECONDS_PER_SECOND,
)

JOINT_STATES_TOPIC = "/joint_states"
JOINT_STATE_SCHEMA_NAME = "sensor_msgs/msg/JointState"
COMPRESSED_IMAGE_SCHEMA_NAME = "sensor_msgs/msg/CompressedImage"

# Joint sine parameters: position_j(t) = 0.5 * sin(2*pi*f_j*t + phase_j), with
# f_j spread linearly over [0.2, 0.5] Hz and phases drawn from the seed.
JOINT_SINE_AMPLITUDE_RAD = 0.5
JOINT_SINE_MIN_FREQUENCY_HZ = 0.2
JOINT_SINE_MAX_FREQUENCY_HZ = 0.5

_CDR_ENCAPSULATION_HEADER = bytes([0x00, 0x01, 0x00, 0x00])
_ROS2MSG_SEPARATOR = "=" * 80

_JOINT_STATE_SCHEMA_TEXT = "\n".join(
    [
        "std_msgs/Header header",
        "string[] name",
        "float64[] position",
        "float64[] velocity",
        "float64[] effort",
        _ROS2MSG_SEPARATOR,
        "MSG: std_msgs/Header",
        "builtin_interfaces/Time stamp",
        "string frame_id",
        _ROS2MSG_SEPARATOR,
        "MSG: builtin_interfaces/Time",
        "int32 sec",
        "uint32 nanosec",
    ]
)

_COMPRESSED_IMAGE_SCHEMA_TEXT = "\n".join(
    [
        "std_msgs/Header header",
        "string format",
        "uint8[] data",
        _ROS2MSG_SEPARATOR,
        "MSG: std_msgs/Header",
        "builtin_interfaces/Time stamp",
        "string frame_id",
        _ROS2MSG_SEPARATOR,
        "MSG: builtin_interfaces/Time",
        "int32 sec",
        "uint32 nanosec",
    ]
)


@dataclass(frozen=True)
class SyntheticEpisodeSpec:
    duration_s: float = 8.0
    cameras: tuple[str, ...] = ("wrist_cam", "overhead_cam")
    image_hz: float = 15.0
    image_width: int = 320
    image_height: int = 240
    joint_hz: float = 100.0
    joint_count: int = 7
    # Segment of near-black frames on the first camera, seconds from start.
    black_segment: tuple[float, float] | None = (2.0, 3.0)
    # Instantaneous joint-position jump (radians) injected at this time.
    joint_jump_at_s: float | None = 5.0
    joint_jump_rad: float = 0.8
    # Segment on the first camera whose image timestamps are offset by +3 ms.
    timestamp_offset_segment: tuple[float, float] | None = (6.0, 7.0)
    timestamp_offset_s: float = 0.003
    start_time_ns: int = 1_755_000_000_000_000_000
    task: str = "fold_napkin"
    operator: str = "synthetic"
    success: bool = True
    robot_software_version: str = "sim-0.1.0"
    seed: int = 0


@dataclass(frozen=True)
class VideoEpisodeSpec:
    """Configuration for adapting a video excerpt into an input-shaped MCAP.

    The output intentionally uses ``sensor_msgs/msg/CompressedImage`` JPEG
    messages so it exercises the normal canonical transform. Fault segments
    are seconds from the beginning of the excerpt, not the source video.
    """

    source_start_s: float = 0.0
    duration_s: float = 20.0
    image_hz: float = 10.0
    image_width: int = 640
    image_height: int = 360
    camera_name: str = "head_camera"
    black_segment: tuple[float, float] | None = None
    freeze_segment: tuple[float, float] | None = None
    start_time_ns: int = 1_755_000_000_000_000_000
    task: str = "human_egocentric_activity"
    operator: str = "anonymous"
    success: bool | None = None
    robot_software_version: str = "external-video"
    metadata: tuple[tuple[str, str], ...] = ()


class _CdrEncoder:
    """Little-endian XCDR1 encoder; alignment is measured from the body start."""

    def __init__(self) -> None:
        self._body = bytearray()

    def align(self, alignment: int) -> None:
        remainder = len(self._body) % alignment
        if remainder:
            self._body.extend(b"\x00" * (alignment - remainder))

    def write_u32(self, value: int) -> None:
        self.align(4)
        self._body += struct.pack("<I", value)

    def write_i32(self, value: int) -> None:
        self.align(4)
        self._body += struct.pack("<i", value)

    def write_f64(self, value: float) -> None:
        self.align(8)
        self._body += struct.pack("<d", value)

    def write_string(self, value: str) -> None:
        encoded_value = value.encode("utf-8")
        # The uint32 length prefix includes the trailing NUL terminator.
        self.write_u32(len(encoded_value) + 1)
        self._body += encoded_value + b"\x00"

    def write_bytes(self, data: bytes) -> None:
        self._body += data

    def payload(self) -> bytes:
        return _CDR_ENCAPSULATION_HEADER + bytes(self._body)


def _write_header(encoder: _CdrEncoder, stamp_ns: int, frame_id: str) -> None:
    encoder.write_i32(stamp_ns // NANOSECONDS_PER_SECOND)
    encoder.write_u32(stamp_ns % NANOSECONDS_PER_SECOND)
    encoder.write_string(frame_id)


def _encode_joint_state(
    stamp_ns: int,
    frame_id: str,
    joint_names: list[str],
    positions: list[float],
    velocities: list[float],
    efforts: list[float],
) -> bytes:
    encoder = _CdrEncoder()
    _write_header(encoder, stamp_ns, frame_id)
    encoder.write_u32(len(joint_names))
    for joint_name in joint_names:
        encoder.write_string(joint_name)
    for float_sequence in (positions, velocities, efforts):
        encoder.write_u32(len(float_sequence))
        for element in float_sequence:
            encoder.write_f64(element)
    return encoder.payload()


def _encode_compressed_image(stamp_ns: int, frame_id: str, jpeg_bytes: bytes) -> bytes:
    encoder = _CdrEncoder()
    _write_header(encoder, stamp_ns, frame_id)
    encoder.write_string("jpeg")
    encoder.write_u32(len(jpeg_bytes))
    encoder.write_bytes(jpeg_bytes)
    return encoder.payload()


def joint_sine_parameters(spec: SyntheticEpisodeSpec) -> list[tuple[float, float]]:
    """Per-joint ``(frequency_hz, phase_rad)`` pairs, derived only from the spec."""
    phase_rng = random.Random(spec.seed)
    frequency_span_hz = JOINT_SINE_MAX_FREQUENCY_HZ - JOINT_SINE_MIN_FREQUENCY_HZ
    parameters: list[tuple[float, float]] = []
    for joint_index in range(spec.joint_count):
        frequency_hz = JOINT_SINE_MIN_FREQUENCY_HZ + frequency_span_hz * joint_index / max(
            spec.joint_count - 1, 1
        )
        phase_rad = phase_rng.uniform(0.0, 2.0 * math.pi)
        parameters.append((frequency_hz, phase_rad))
    return parameters


def _message_log_time_ns(spec: SyntheticEpisodeSpec, message_index: int, rate_hz: float) -> int:
    return spec.start_time_ns + round(message_index * NANOSECONDS_PER_SECOND / rate_hz)


def _time_is_in_segment(time_s: float, segment: tuple[float, float] | None) -> bool:
    return segment is not None and segment[0] <= time_s < segment[1]


def _run_ffmpeg(arguments: list[str]) -> None:
    command = [str(ffmpeg_path()), "-hide_banner", "-loglevel", "error", "-y", *arguments]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({' '.join(command)}): {completed.stderr.decode(errors='replace')}"
        )


def _render_camera_frames(
    spec: SyntheticEpisodeSpec, camera_index: int, frame_count: int, work_dir: Path
) -> list[bytes]:
    """Render one camera's JPEG frames with a single ffmpeg run."""
    frames_dir = work_dir / f"camera_{camera_index}"
    frames_dir.mkdir()
    size_argument = f"size={spec.image_width}x{spec.image_height}"
    rate_argument = f"rate={spec.image_hz:g}"
    if camera_index == 0:
        lavfi_source = f"testsrc2={size_argument}:{rate_argument}"
        video_filters: list[str] = []
    else:
        # Vary hue (and invert every other camera) so cameras are visually distinct.
        lavfi_source = f"smptebars={size_argument}:{rate_argument}"
        video_filters = [f"hue=h={(90 * camera_index) % 360}"]
        if camera_index % 2 == 0:
            video_filters.append("negate")
    arguments = ["-f", "lavfi", "-i", lavfi_source, "-frames:v", str(frame_count)]
    if video_filters:
        arguments += ["-vf", ",".join(video_filters)]
    arguments += [
        "-pix_fmt",
        "yuvj420p",
        "-c:v",
        "mjpeg",
        "-q:v",
        "5",
        "-f",
        "image2",
        str(frames_dir / "frame_%05d.jpg"),
    ]
    _run_ffmpeg(arguments)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) != frame_count:
        raise RuntimeError(
            f"expected {frame_count} frames from ffmpeg for camera {camera_index}, "
            f"got {len(frame_paths)}"
        )
    return [frame_path.read_bytes() for frame_path in frame_paths]


def _render_black_frame(spec: SyntheticEpisodeSpec, work_dir: Path) -> bytes:
    black_frame_path = work_dir / "black.jpg"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=black:size={spec.image_width}x{spec.image_height}",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            "5",
            str(black_frame_path),
        ]
    )
    return black_frame_path.read_bytes()


def _validate_video_fault_segment(
    segment_name: str,
    segment: tuple[float, float] | None,
    duration_s: float,
) -> None:
    if segment is None:
        return
    segment_start_s, segment_end_s = segment
    if segment_start_s < 0.0 or segment_end_s <= segment_start_s or segment_end_s > duration_s:
        raise ValueError(
            f"{segment_name} must satisfy 0 <= start < end <= duration_s; "
            f"got {segment!r} for duration_s={duration_s}"
        )


def _validate_video_episode_spec(spec: VideoEpisodeSpec) -> None:
    if spec.source_start_s < 0.0:
        raise ValueError(f"source_start_s must be >= 0, got {spec.source_start_s}")
    if spec.duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {spec.duration_s}")
    if spec.image_hz <= 0.0:
        raise ValueError(f"image_hz must be > 0, got {spec.image_hz}")
    if spec.image_width <= 0 or spec.image_height <= 0:
        raise ValueError(
            f"image dimensions must be positive, got {spec.image_width}x{spec.image_height}"
        )
    if spec.image_width % 2 != 0 or spec.image_height % 2 != 0:
        raise ValueError(
            "image dimensions must be even so the canonical H.264 transform can use yuv420p; "
            f"got {spec.image_width}x{spec.image_height}"
        )
    if not spec.camera_name.strip():
        raise ValueError(f"camera_name must not be empty, got {spec.camera_name!r}")
    _validate_video_fault_segment("black_segment", spec.black_segment, spec.duration_s)
    _validate_video_fault_segment("freeze_segment", spec.freeze_segment, spec.duration_s)
    if spec.black_segment is not None and spec.freeze_segment is not None:
        black_start_s, black_end_s = spec.black_segment
        freeze_start_s, freeze_end_s = spec.freeze_segment
        if max(black_start_s, freeze_start_s) < min(black_end_s, freeze_end_s):
            raise ValueError(
                f"black_segment and freeze_segment must not overlap, "
                f"got {spec.black_segment!r} and {spec.freeze_segment!r}"
            )


def _render_source_video_frames(
    source_video_path: Path,
    spec: VideoEpisodeSpec,
    work_dir: Path,
) -> list[bytes]:
    frame_count = round(spec.duration_s * spec.image_hz)
    if frame_count < 1:
        raise ValueError(
            f"duration_s * image_hz must produce at least one frame, got {frame_count}"
        )
    frames_dir = work_dir / "source_video"
    frames_dir.mkdir()
    video_filter = (
        f"fps={spec.image_hz:g},"
        f"scale={spec.image_width}:{spec.image_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={spec.image_width}:{spec.image_height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    _run_ffmpeg(
        [
            "-ss",
            f"{spec.source_start_s:g}",
            "-i",
            str(source_video_path),
            "-an",
            "-vf",
            video_filter,
            "-frames:v",
            str(frame_count),
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            "5",
            "-f",
            "image2",
            str(frames_dir / "frame_%05d.jpg"),
        ]
    )
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_paths) != frame_count:
        raise RuntimeError(
            f"expected {frame_count} frames from {source_video_path}, got {len(frame_paths)}; "
            "the requested excerpt may extend past the source video"
        )
    return [frame_path.read_bytes() for frame_path in frame_paths]


def _render_video_black_frame(spec: VideoEpisodeSpec, work_dir: Path) -> bytes:
    black_frame_path = work_dir / "video_black.jpg"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=black:size={spec.image_width}x{spec.image_height}",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            "5",
            str(black_frame_path),
        ]
    )
    return black_frame_path.read_bytes()


def _build_video_camera_messages(
    source_video_path: Path,
    spec: VideoEpisodeSpec,
    work_dir: Path,
) -> list[tuple[int, bytes]]:
    frame_jpegs = _render_source_video_frames(source_video_path, spec, work_dir)
    black_frame_jpeg = (
        _render_video_black_frame(spec, work_dir) if spec.black_segment is not None else None
    )
    frozen_frame_jpeg: bytes | None = None
    if spec.freeze_segment is not None:
        freeze_start_s, _freeze_end_s = spec.freeze_segment
        frozen_frame_index = min(round(freeze_start_s * spec.image_hz), len(frame_jpegs) - 1)
        frozen_frame_jpeg = frame_jpegs[frozen_frame_index]

    messages: list[tuple[int, bytes]] = []
    for frame_index, original_frame_jpeg in enumerate(frame_jpegs):
        frame_time_s = frame_index / spec.image_hz
        frame_jpeg = original_frame_jpeg
        if black_frame_jpeg is not None and _time_is_in_segment(frame_time_s, spec.black_segment):
            frame_jpeg = black_frame_jpeg
        elif frozen_frame_jpeg is not None and _time_is_in_segment(
            frame_time_s, spec.freeze_segment
        ):
            frame_jpeg = frozen_frame_jpeg
        log_time_ns = spec.start_time_ns + round(
            frame_index * NANOSECONDS_PER_SECOND / spec.image_hz
        )
        messages.append(
            (
                log_time_ns,
                _encode_compressed_image(
                    stamp_ns=log_time_ns,
                    frame_id=spec.camera_name,
                    jpeg_bytes=frame_jpeg,
                ),
            )
        )
    return messages


def _video_episode_metadata(spec: VideoEpisodeSpec) -> dict[str, str]:
    metadata = {
        "task": spec.task,
        "operator": spec.operator,
        EPISODE_KEY_ROBOT_SOFTWARE_VERSION: spec.robot_software_version,
    }
    if spec.success is not None:
        metadata["success"] = "true" if spec.success else "false"
    reserved_metadata_keys = {
        "task",
        "operator",
        "success",
        EPISODE_KEY_ROBOT_SOFTWARE_VERSION,
    }
    additional_metadata_keys: set[str] = set()
    for metadata_key, metadata_value in spec.metadata:
        if metadata_key in reserved_metadata_keys:
            raise ValueError(f"metadata key {metadata_key!r} is reserved")
        if metadata_key in additional_metadata_keys:
            raise ValueError(f"metadata contains duplicate key {metadata_key!r}")
        additional_metadata_keys.add(metadata_key)
        metadata[metadata_key] = metadata_value
    return metadata


def write_video_episode(
    source_video: Path | str,
    output: Path | str,
    spec: VideoEpisodeSpec | None = None,
) -> Path:
    """Adapt a video excerpt into a ROS 2-shaped MCAP for demos and tests.

    This is a source adapter, not the canonical transform: it writes JPEG
    ``CompressedImage`` messages that :func:`write_canonical_episode` later
    converts to in-band H.264 ``foxglove.CompressedVideo`` messages.
    """
    resolved_spec = spec if spec is not None else VideoEpisodeSpec()
    _validate_video_episode_spec(resolved_spec)
    source_video_path = Path(source_video)
    if not source_video_path.is_file():
        raise FileNotFoundError(source_video_path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="video-episode-") as work_dir_name:
        camera_messages = _build_video_camera_messages(
            source_video_path, resolved_spec, Path(work_dir_name)
        )

    with output_path.open("wb") as output_stream:
        writer = Writer(output_stream)
        writer.start(profile="ros2", library="hflow video episode adapter")
        schema_id = writer.register_schema(
            name=COMPRESSED_IMAGE_SCHEMA_NAME,
            encoding="ros2msg",
            data=_COMPRESSED_IMAGE_SCHEMA_TEXT.encode("utf-8"),
        )
        channel_id = writer.register_channel(
            topic=f"/{resolved_spec.camera_name}/compressed",
            message_encoding="cdr",
            schema_id=schema_id,
        )
        writer.add_metadata(
            name=METADATA_RECORD_EPISODE, data=_video_episode_metadata(resolved_spec)
        )
        for sequence, (log_time_ns, payload) in enumerate(camera_messages):
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time_ns,
                data=payload,
                publish_time=log_time_ns,
                sequence=sequence,
            )
        writer.finish()
    return output_path


def _build_joint_messages(spec: SyntheticEpisodeSpec) -> list[tuple[int, bytes]]:
    """(log_time_ns, cdr_payload) per JointState message, in time order."""
    sine_parameters = joint_sine_parameters(spec)
    joint_names = [f"joint_{joint_index}" for joint_index in range(spec.joint_count)]
    message_count = round(spec.duration_s * spec.joint_hz)
    messages: list[tuple[int, bytes]] = []
    for message_index in range(message_count):
        time_s = message_index / spec.joint_hz
        jump_offset_rad = (
            spec.joint_jump_rad
            if spec.joint_jump_at_s is not None and time_s >= spec.joint_jump_at_s
            else 0.0
        )
        positions: list[float] = []
        velocities: list[float] = []
        for frequency_hz, phase_rad in sine_parameters:
            angle_rad = 2.0 * math.pi * frequency_hz * time_s + phase_rad
            positions.append(JOINT_SINE_AMPLITUDE_RAD * math.sin(angle_rad) + jump_offset_rad)
            velocities.append(
                JOINT_SINE_AMPLITUDE_RAD * 2.0 * math.pi * frequency_hz * math.cos(angle_rad)
            )
        log_time_ns = _message_log_time_ns(spec, message_index, spec.joint_hz)
        payload = _encode_joint_state(
            stamp_ns=log_time_ns,
            frame_id="base",
            joint_names=joint_names,
            positions=positions,
            velocities=velocities,
            efforts=[0.0] * spec.joint_count,
        )
        messages.append((log_time_ns, payload))
    return messages


def _build_camera_messages(
    spec: SyntheticEpisodeSpec, camera_index: int, camera_name: str, work_dir: Path
) -> list[tuple[int, bytes]]:
    """(log_time_ns, cdr_payload) per CompressedImage message, in time order."""
    frame_count = round(spec.duration_s * spec.image_hz)
    frame_jpegs = _render_camera_frames(spec, camera_index, frame_count, work_dir)
    is_faulted_camera = camera_index == 0
    black_frame_jpeg: bytes | None = None
    if is_faulted_camera and spec.black_segment is not None:
        black_frame_jpeg = _render_black_frame(spec, work_dir)
    timestamp_offset_ns = round(spec.timestamp_offset_s * NANOSECONDS_PER_SECOND)
    messages: list[tuple[int, bytes]] = []
    for frame_index, frame_jpeg in enumerate(frame_jpegs):
        time_s = frame_index / spec.image_hz
        log_time_ns = _message_log_time_ns(spec, frame_index, spec.image_hz)
        if is_faulted_camera and _time_is_in_segment(time_s, spec.timestamp_offset_segment):
            log_time_ns += timestamp_offset_ns
        if (
            is_faulted_camera
            and black_frame_jpeg is not None
            and _time_is_in_segment(time_s, spec.black_segment)
        ):
            frame_jpeg = black_frame_jpeg
        payload = _encode_compressed_image(
            stamp_ns=log_time_ns, frame_id=camera_name, jpeg_bytes=frame_jpeg
        )
        messages.append((log_time_ns, payload))
    return messages


def _calibration_attachment_json(spec: SyntheticEpisodeSpec) -> bytes:
    camera_calibrations = {
        camera_name: {
            "width": spec.image_width,
            "height": spec.image_height,
            "fx": 300.0,
            "fy": 300.0,
            "cx": spec.image_width / 2.0,
            "cy": spec.image_height / 2.0,
        }
        for camera_name in spec.cameras
    }
    return json.dumps({"cameras": camera_calibrations}, sort_keys=True).encode("utf-8")


def synthesize_episode(output: Path | str, spec: SyntheticEpisodeSpec | None = None) -> Path:
    """Write a synthetic input episode to ``output`` and return its path."""
    resolved_spec = spec if spec is not None else SyntheticEpisodeSpec()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # (log_time_ns, channel_ordinal, sequence, payload) for a global log-time
    # sort; ordinal and sequence break ties deterministically.
    pending_messages: list[tuple[int, int, int, bytes]] = []
    channel_topics: list[tuple[str, str]] = [(JOINT_STATES_TOPIC, JOINT_STATE_SCHEMA_NAME)]
    with tempfile.TemporaryDirectory(prefix="synthetic-episode-") as work_dir_name:
        work_dir = Path(work_dir_name)
        for sequence, (log_time_ns, payload) in enumerate(_build_joint_messages(resolved_spec)):
            pending_messages.append((log_time_ns, 0, sequence, payload))
        for camera_index, camera_name in enumerate(resolved_spec.cameras):
            channel_ordinal = 1 + camera_index
            channel_topics.append((f"/{camera_name}/compressed", COMPRESSED_IMAGE_SCHEMA_NAME))
            camera_messages = _build_camera_messages(
                resolved_spec, camera_index, camera_name, work_dir
            )
            for sequence, (log_time_ns, payload) in enumerate(camera_messages):
                pending_messages.append((log_time_ns, channel_ordinal, sequence, payload))
    pending_messages.sort(key=lambda message: (message[0], message[1], message[2]))

    with output_path.open("wb") as output_stream:
        writer = Writer(output_stream)
        writer.start(profile="ros2", library="hflow synthetic episode")
        schema_ids = {
            JOINT_STATE_SCHEMA_NAME: writer.register_schema(
                name=JOINT_STATE_SCHEMA_NAME,
                encoding="ros2msg",
                data=_JOINT_STATE_SCHEMA_TEXT.encode("utf-8"),
            ),
            COMPRESSED_IMAGE_SCHEMA_NAME: writer.register_schema(
                name=COMPRESSED_IMAGE_SCHEMA_NAME,
                encoding="ros2msg",
                data=_COMPRESSED_IMAGE_SCHEMA_TEXT.encode("utf-8"),
            ),
        }
        channel_ids = [
            writer.register_channel(
                topic=topic, message_encoding="cdr", schema_id=schema_ids[schema_name]
            )
            for topic, schema_name in channel_topics
        ]
        writer.add_metadata(
            name=METADATA_RECORD_EPISODE,
            data={
                "task": resolved_spec.task,
                "operator": resolved_spec.operator,
                "success": "true" if resolved_spec.success else "false",
                EPISODE_KEY_ROBOT_SOFTWARE_VERSION: resolved_spec.robot_software_version,
            },
        )
        writer.add_attachment(
            create_time=resolved_spec.start_time_ns,
            log_time=resolved_spec.start_time_ns,
            name="calibration.json",
            media_type="application/json",
            data=_calibration_attachment_json(resolved_spec),
        )
        for log_time_ns, channel_ordinal, sequence, payload in pending_messages:
            writer.add_message(
                channel_id=channel_ids[channel_ordinal],
                log_time=log_time_ns,
                data=payload,
                publish_time=log_time_ns,
                sequence=sequence,
            )
        writer.finish()
    return output_path
