"""Stream one MCAP camera to MP4 while preserving irregular log-time intervals.

Requires the optional ``video`` extra. Decoding is synchronous; callers own
process isolation when a hard native-decoder deadline is required.
"""

from __future__ import annotations

import base64
import math
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import chain
from pathlib import Path

import av
import numpy as np
from av.container import OutputContainer
from av.stream import Stream
from av.video.codeccontext import VideoCodecContext
from av.video.stream import VideoStream

from hflow import Episode, TopicInfo
from hflow.format import PASSTHROUGH_VIDEO_SCHEMA_NAMES
from hflow.media import VideoLimits
from hflow.video import (
    _annex_b_nal_offsets_and_types,
    ensure_access_unit_delimiter,
    scan_picture_coding_types,
    split_annex_b_stream,
)

MAXIMUM_FRAME_BYTES = 32 * 1024 * 1024
NANOSECONDS_PER_SECOND = 1_000_000_000
VIDEO_TICKS_PER_SECOND = 1_000_000
NANOSECONDS_PER_VIDEO_TICK = NANOSECONDS_PER_SECOND // VIDEO_TICKS_PER_SECOND
MAXIMUM_VIDEO_INTERVAL_TICKS = 2**31 - 1
VIDEO_TIME_BASE = Fraction(1, VIDEO_TICKS_PER_SECOND)
COMPRESSED_IMAGE_SCHEMAS = frozenset(
    {"foxglove.CompressedImage", "sensor_msgs/msg/CompressedImage"}
)
RAW_IMAGE_SCHEMAS = frozenset({"foxglove.RawImage", "sensor_msgs/msg/Image"})
RAW_PIXEL_FORMATS = {
    "rgb8": ("rgb24", 3),
    "bgr8": ("bgr24", 3),
    "rgba8": ("rgba", 4),
    "bgra8": ("bgra", 4),
    "mono8": ("gray", 1),
}


class McapMediaError(RuntimeError):
    """The selected camera could not be prepared within the supported profile."""


@dataclass(frozen=True, slots=True)
class PreparedMcapVideo:
    video_path: Path = field(repr=False)
    duration_millis: int
    camera_topic: str = field(repr=False)
    start_timestamp_ns: int


@dataclass(frozen=True, slots=True)
class CameraFrameInterval:
    timestamp_ns: int
    duration_ns: int
    message: object = field(repr=False)


def _field(message: object, name: str) -> object:
    return message.get(name) if isinstance(message, Mapping) else getattr(message, name, None)


def _frame_bytes(message: object) -> bytes:
    payload = _field(message, "data")
    if isinstance(payload, str):
        if len(payload) > MAXIMUM_FRAME_BYTES * 2:
            raise McapMediaError()
        payload = base64.b64decode(payload, validate=True)
    if not isinstance(payload, bytes | bytearray | memoryview | list | np.ndarray):
        raise McapMediaError()
    if not 0 < len(payload) <= MAXIMUM_FRAME_BYTES:
        raise McapMediaError()
    return bytes(payload)


def _select_camera(episode: Episode, camera_topic: str | None) -> TopicInfo:
    cameras = episode.cameras
    if camera_topic is None:
        if len(cameras) != 1:
            raise McapMediaError()
        camera_topic = cameras[0]
    if camera_topic not in cameras:
        raise McapMediaError()
    channels = [channel for channel in episode.channels.values() if channel.topic == camera_topic]
    if len(channels) != 1:
        raise McapMediaError()
    return channels[0]


def _camera_messages(
    episode: Episode, channel: TopicInfo, limits: VideoLimits
) -> Iterator[tuple[int, object]]:
    maximum_frames = math.ceil(limits.maximum_duration_seconds * limits.maximum_frames_per_second)
    frame_count = 0
    for batch in episode.iter_decoded_batches(
        channel_ids=[channel.channel_id], batch_max_messages=32, batch_max_bytes=8 * 1024 * 1024
    ):
        for timestamp, message in zip(batch.log_times, batch.messages, strict=True):
            frame_count += 1
            if frame_count > maximum_frames:
                raise McapMediaError()
            yield int(timestamp), message


def _frame_intervals(
    messages: Iterator[tuple[int, object]], limits: VideoLimits
) -> Iterator[CameraFrameInterval]:
    previous_timestamp, previous_message = next(messages)
    start_timestamp = previous_timestamp
    previous_interval = 0
    frame_count = 1
    maximum_duration_ns = int(limits.maximum_duration_seconds * NANOSECONDS_PER_SECOND)
    for timestamp, message in messages:
        interval = timestamp - previous_timestamp
        if interval <= 0 or timestamp - start_timestamp > maximum_duration_ns:
            raise McapMediaError()
        frame_count += 1
        yield CameraFrameInterval(previous_timestamp - start_timestamp, interval, previous_message)
        previous_timestamp, previous_message, previous_interval = timestamp, message, interval
    final_duration = previous_timestamp - start_timestamp + previous_interval
    if previous_interval == 0 or not (
        limits.minimum_duration_seconds * NANOSECONDS_PER_SECOND
        <= final_duration
        <= maximum_duration_ns
    ):
        raise McapMediaError()
    # Log timestamps measure arrival time and can bunch together during capture.
    # Bound sustained density while preserving those positive timestamp gaps.
    if frame_count * NANOSECONDS_PER_SECOND > limits.maximum_frames_per_second * final_duration:
        raise McapMediaError()
    yield CameraFrameInterval(
        previous_timestamp - start_timestamp, previous_interval, previous_message
    )


def _require_dimensions(width: int, height: int, limits: VideoLimits) -> None:
    if width <= 0 or height <= 0 or width * height > limits.maximum_frame_pixels:
        raise McapMediaError()


def _decode_image(message: object, schema_name: str, limits: VideoLimits) -> av.VideoFrame:
    payload = _frame_bytes(message)
    if schema_name in COMPRESSED_IMAGE_SCHEMAS:
        encoding = _field(message, "format")
        if not isinstance(encoding, str):
            raise McapMediaError()
        if "jpeg" in encoding.lower() or "jpg" in encoding.lower():
            codec_name = "mjpeg"
        elif "png" in encoding.lower():
            codec_name = "png"
        else:
            raise McapMediaError()
        decoder = av.CodecContext.create(codec_name, "r")
        decoder.thread_count = 1
        frames = decoder.decode(av.Packet(payload))
        if len(frames) != 1 or not isinstance(frames[0], av.VideoFrame):
            raise McapMediaError()
        frame = frames[0]
        _require_dimensions(frame.width, frame.height, limits)
        return frame.reformat(format="rgb24")
    if schema_name not in RAW_IMAGE_SCHEMAS:
        raise McapMediaError()
    encoding = _field(message, "encoding")
    width, height, step = (_field(message, name) for name in ("width", "height", "step"))
    if (
        not isinstance(encoding, str)
        or encoding not in RAW_PIXEL_FORMATS
        or type(width) is not int
        or type(height) is not int
        or type(step) is not int
    ):
        raise McapMediaError()
    _require_dimensions(width, height, limits)
    pixel_format, channels = RAW_PIXEL_FORMATS[encoding]
    if step < width * channels or len(payload) != step * height:
        raise McapMediaError()
    pixels = np.frombuffer(payload, dtype=np.uint8).reshape(height, step)[:, : width * channels]
    shape = (height, width) if channels == 1 else (height, width, channels)
    return av.VideoFrame.from_ndarray(
        np.ascontiguousarray(pixels.reshape(shape)), format=pixel_format
    ).reformat(format="rgb24")


def _h264_packet(
    message: object, limits: VideoLimits, parameter_sets: dict[int, bytes]
) -> tuple[av.Packet, tuple[int, int] | None]:
    if _field(message, "format") != "h264":
        raise McapMediaError()
    payload = ensure_access_unit_delimiter(_frame_bytes(message))
    coding = scan_picture_coding_types(payload)
    units = split_annex_b_stream(payload)
    if coding.picture_count != 1 or coding.b_picture_count or len(units) != 1:
        raise McapMediaError()
    # Codec configuration persists across access units. Later IDRs may omit
    # SPS/PPS, and an SPS-only update must still pass our dimension limits.
    parameter_sets_changed = False
    nal_offsets = _annex_b_nal_offsets_and_types(payload)
    for index, (start_offset, nal_type) in enumerate(nal_offsets):
        if nal_type in (7, 8):
            end_offset = nal_offsets[index + 1][0] if index + 1 < len(nal_offsets) else len(payload)
            parameter_sets[nal_type] = payload[start_offset:end_offset]
            parameter_sets_changed = True
    if set(parameter_sets) != {7, 8}:
        raise McapMediaError()
    dimensions = None
    if parameter_sets_changed:
        parser = av.CodecContext.create("h264", "r")
        parser.parse(parameter_sets[7] + parameter_sets[8] + payload)
        parser.parse(None)
        _require_dimensions(parser.width, parser.height, limits)
        dimensions = (parser.width, parser.height)
    packet = av.Packet(payload)
    packet.is_keyframe = units[0].is_keyframe
    return packet, dimensions


def _write_camera_video(
    output: OutputContainer,
    intervals: Iterator[CameraFrameInterval],
    channel: TopicInfo,
    limits: VideoLimits,
) -> int:
    stream: Stream | None = None
    parameter_sets: dict[int, bytes] = {}
    source_dimensions: tuple[int, int] | None = None
    duration_ns = 0
    timestamp_ticks = 0
    for interval in intervals:
        duration_ns = interval.timestamp_ns + interval.duration_ns
        # Round each absolute endpoint once, then share it with the next PTS.
        # Accumulating rounded intervals would drift on irregular camera timing.
        end_timestamp_ticks = (
            duration_ns + NANOSECONDS_PER_VIDEO_TICK // 2
        ) // NANOSECONDS_PER_VIDEO_TICK
        duration_ticks = end_timestamp_ticks - timestamp_ticks
        if not 0 < duration_ticks <= MAXIMUM_VIDEO_INTERVAL_TICKS:
            raise McapMediaError()
        if channel.schema_name in PASSTHROUGH_VIDEO_SCHEMA_NAMES:
            packet, dimensions = _h264_packet(interval.message, limits, parameter_sets)
            if stream is None:
                if not packet.is_keyframe or dimensions is None:
                    raise McapMediaError()
                source_dimensions = dimensions
                stream = output.add_mux_stream(
                    "h264",
                    width=dimensions[0],
                    height=dimensions[1],
                    time_base=VIDEO_TIME_BASE,
                )
            elif dimensions is not None and dimensions != source_dimensions:
                raise McapMediaError()
            packet.pts = packet.dts = timestamp_ticks
            packet.time_base = VIDEO_TIME_BASE
            packet.stream = stream
        else:
            frame = _decode_image(interval.message, channel.schema_name, limits)
            if stream is None:
                stream = output.add_stream(
                    "libx264rgb",
                    rate=30,
                    width=frame.width,
                    height=frame.height,
                    pix_fmt="rgb24",
                    time_base=VIDEO_TIME_BASE,
                    options={
                        "crf": "0",
                        "preset": "ultrafast",
                        "tune": "zerolatency",
                        "x264-params": "bframes=0:repeat-headers=1:aud=1",
                    },
                )
                stream.codec_context.thread_count = 1
            if not isinstance(stream, VideoStream) or (frame.width, frame.height) != (
                stream.width,
                stream.height,
            ):
                raise McapMediaError()
            frame.pts, frame.time_base = timestamp_ticks, VIDEO_TIME_BASE
            packets = stream.encode(frame)
            if len(packets) != 1:
                raise McapMediaError()
            packet = packets[0]
        packet.duration = duration_ticks
        output.mux(packet)
        timestamp_ticks = end_timestamp_ticks
    if (
        isinstance(stream, VideoStream)
        and isinstance(stream.codec_context, VideoCodecContext)
        and stream.encode(None)
    ):
        raise McapMediaError()
    return (duration_ns + 999_999) // 1_000_000


def _export_camera(
    source_path: Path, output_path: Path, camera_topic: str | None, limits: VideoLimits
) -> PreparedMcapVideo:
    with Episode(source_path) as episode:
        channel = _select_camera(episode, camera_topic)
        messages = _camera_messages(episode, channel, limits)
        first_message = next(messages)
        with av.open(
            output_path,
            "w",
            format="mp4",
            options={"video_track_timescale": str(VIDEO_TICKS_PER_SECOND)},
        ) as output:
            duration_millis = _write_camera_video(
                output,
                _frame_intervals(chain((first_message,), messages), limits),
                channel,
                limits,
            )
    return PreparedMcapVideo(output_path, duration_millis, channel.topic, first_message[0])


def export_mcap_camera(
    source_path: Path,
    output_path: Path,
    *,
    camera_topic: str | None = None,
    limits: VideoLimits = VideoLimits(),
) -> PreparedMcapVideo:
    """Export a selected camera without modifying source or existing destination.

    PTS are MCAP log times relative to the first selected frame, rounded to the
    nearest microsecond. The last frame repeats the preceding positive interval.
    Requires at least two frames with distinct rounded times and gaps no larger
    than 2,147.483647 seconds. H.264 without B-frames is remuxed with lossless AUD
    repair. JPEG/PNG and supported 8-bit raw images are encoded as lossless RGB
    H.264. Metadata includes the selected topic and original first log timestamp.
    """
    source_path = source_path.resolve(strict=True)
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise McapMediaError("MCAP source must be a nonempty local file")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".camera-export-") as directory:
        staged_path = Path(directory) / "video.mp4"
        try:
            result = _export_camera(source_path, staged_path, camera_topic, limits)
        except (StopIteration, RuntimeError, ValueError) as error:
            raise McapMediaError(
                "MCAP camera cannot be exported with the requested profile"
            ) from error
        os.link(staged_path, output_path)
        return replace(result, video_path=output_path)
