"""Import a local video excerpt as an input episode for the processing engine."""

import errno
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from mcap.writer import Writer
from mcap_protobuf.schema import build_file_descriptor_set

from hflow._pinned_asset import sha256_hex_of_file
from hflow.ffmpeg import ffmpeg_path, ffmpeg_version, ffprobe_path
from hflow.format import METADATA_RECORD_EPISODE, NANOSECONDS_PER_SECOND

_IMPORT_METADATA_RECORD = "video_import/v1"
_MAXIMUM_TIMESTAMP_NS = (1 << 64) - 1


@dataclass(frozen=True)
class VideoImportConfig:
    """A fixed-rate excerpt from the first video stream of a local file.

    Samples cover ``[source_start_s, source_start_s + duration_s)`` on a
    regular ``image_hz`` grid, producing ``ceil(duration_s * image_hz)``
    frames. FFmpeg resamples the source using the frame covering each sample
    time, duplicating frames when necessary.
    Images are resized with aspect ratio preserved and black letterboxing.
    Episode timestamps start at ``start_time_ns``, independent of the source
    offset, and are rounded to the nearest nanosecond. No recording date,
    task, operator, or success label is inferred; ``metadata`` supplies only
    the episode fields the caller actually knows.
    """

    duration_s: float
    source_start_s: float = 0.0
    image_hz: float = 10.0
    image_width: int = 640
    image_height: int = 360
    camera_name: str = "camera"
    start_time_ns: int = 0
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("duration_s", self.duration_s),
            ("source_start_s", self.source_start_s),
            ("image_hz", self.image_hz),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.duration_s <= 0 or self.source_start_s < 0:
            raise ValueError("duration_s must be positive and source_start_s nonnegative")
        if not 0 < self.image_hz <= NANOSECONDS_PER_SECOND:
            raise ValueError("image_hz must be positive and no greater than 1 GHz")
        if not math.isfinite(self.source_start_s + self.duration_s):
            raise ValueError("the excerpt end must be finite")
        for name, value in (
            ("image_width", self.image_width),
            ("image_height", self.image_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2:
                raise ValueError(f"{name} must be a positive even integer")
        if (
            isinstance(self.start_time_ns, bool)
            or not isinstance(self.start_time_ns, int)
            or not 0 <= self.start_time_ns <= _MAXIMUM_TIMESTAMP_NS
        ):
            raise ValueError("start_time_ns must be an unsigned 64-bit integer")
        if self.frame_count > (1 << 32):
            raise ValueError("the excerpt exceeds the MCAP sequence number range")
        final_timestamp_ns = _sample_timestamp_ns(self, self.frame_count - 1)
        if final_timestamp_ns > _MAXIMUM_TIMESTAMP_NS:
            raise ValueError("the excerpt exceeds the MCAP timestamp range")
        if (
            not isinstance(self.camera_name, str)
            or not self.camera_name
            or self.camera_name.strip("/") != self.camera_name
            or any(character.isspace() for character in self.camera_name)
        ):
            raise ValueError("camera_name must be nonempty without edge slashes or whitespace")
        metadata_keys: set[str] = set()
        for metadata_key, metadata_value in self.metadata:
            if not isinstance(metadata_key, str) or not metadata_key:
                raise ValueError("metadata keys must be nonempty strings")
            if not isinstance(metadata_value, str):
                raise ValueError("metadata values must be strings")
            if metadata_key in metadata_keys:
                raise ValueError(f"metadata contains duplicate key {metadata_key!r}")
            metadata_keys.add(metadata_key)

    @property
    def frame_count(self) -> int:
        """Number of samples on the excerpt's half-open sampling grid."""
        # Decimal rates agree with the values passed to FFmpeg. Binary float
        # multiplication would invent an extra sample for e.g. 0.14 s at 100 Hz.
        return math.ceil(Fraction(str(self.duration_s)) * Fraction(str(self.image_hz)))


def _sample_timestamp_ns(config: VideoImportConfig, frame_index: int) -> int:
    return config.start_time_ns + round(
        Fraction(frame_index * NANOSECONDS_PER_SECOND) / Fraction(str(config.image_hz))
    )


def _require_excerpt_duration(source_video: Path, config: VideoImportConfig) -> None:
    completed = subprocess.run(
        [
            str(ffprobe_path()),
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(source_video),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "could not inspect source video: " + completed.stderr.decode(errors="replace").strip()
        )
    source_information = json.loads(completed.stdout)
    streams = source_information.get("streams", [])
    if not streams:
        raise ValueError("source file has no video stream")
    try:
        video_duration_s = float(streams[0]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source video stream must declare its duration") from error
    if not math.isfinite(video_duration_s) or video_duration_s <= 0:
        raise ValueError("source video stream must declare a finite positive duration")
    if config.source_start_s + config.duration_s > video_duration_s + 1e-6:
        raise ValueError("the requested excerpt extends past the source video")


def _render_frames(source_video: Path, config: VideoImportConfig, working_directory: Path) -> None:
    # Keep the preceding keyframe's negative, excerpt-relative timestamps.
    # Accurate seeking would discard the frame covering a non-frame-aligned
    # start, letting fps pad the excerpt with a later (potentially different)
    # frame. Resample before trimming so that preceding frame stays available.
    video_filter = (
        f"fps={config.image_hz}:start_time=0:round=up:eof_action=pass,"
        f"trim=duration={config.duration_s},"
        f"scale={config.image_width}:{config.image_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={config.image_width}:{config.image_height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    completed = subprocess.run(
        [
            str(ffmpeg_path()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-xerror",
            "-protocol_whitelist",
            "file",
            "-noaccurate_seek",
            "-ss",
            str(config.source_start_s),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            video_filter,
            "-frames:v",
            str(config.frame_count),
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            "5",
            "-f",
            "image2",
            str(working_directory / "frame_%010d.jpg"),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "could not decode source video: " + completed.stderr.decode(errors="replace").strip()
        )


def import_video_episode(
    source_video: Path | str, output: Path | str, config: VideoImportConfig
) -> Path:
    """Import a local excerpt into an MCAP for :meth:`hflow.App.process`.

    The first video stream must declare its duration (as MP4 video streams
    do). Missing, incomplete, corrupt, or out-of-range excerpts raise without
    publishing an output. URL inputs and network references are not read.
    FFmpeg uses HFlow's usual managed-binary policy.

    JPEG ``foxglove.CompressedImage`` messages are source data, not canonical
    output; the normal processing engine still owns transformation and QC.
    Only caller-supplied fields enter ``episode/v1``. ``video_import/v1``
    records source SHA-256, import settings, and the FFmpeg version.

    Conversion uses temporary disk beside ``output`` and reads one JPEG at
    a time into the MCAP writer. The complete output is published atomically
    without overwriting an existing path, including a concurrent publisher.
    Temporary files are cleaned on success and exception; the caller owns
    the published output and any later processing workspace.
    """
    source_video_path = Path(source_video).resolve()
    if not source_video_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(source_video_path))
    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(output_path))
    _require_excerpt_duration(source_video_path, config)
    source_sha256 = sha256_hex_of_file(source_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".video-import-") as directory:
        working_directory = Path(directory)
        _render_frames(source_video_path, config, working_directory)
        staged_episode = working_directory / "episode.mcap"
        with staged_episode.open("wb") as output_stream:
            writer = Writer(output_stream)
            writer.start(library="hflow video importer")
            schema_id = writer.register_schema(
                name="foxglove.CompressedImage",
                encoding="protobuf",
                data=build_file_descriptor_set(CompressedImage).SerializeToString(),
            )
            channel_id = writer.register_channel(
                topic=f"/{config.camera_name}/compressed",
                message_encoding="protobuf",
                schema_id=schema_id,
            )
            writer.add_metadata(name=METADATA_RECORD_EPISODE, data=dict(config.metadata))
            writer.add_metadata(
                name=_IMPORT_METADATA_RECORD,
                data={
                    "importer_version": "1",
                    "source_sha256": source_sha256,
                    "source_start_s": str(config.source_start_s),
                    "duration_s": str(config.duration_s),
                    "image_hz": str(config.image_hz),
                    "image_width": str(config.image_width),
                    "image_height": str(config.image_height),
                    "camera_name": config.camera_name,
                    "start_time_ns": str(config.start_time_ns),
                    "frame_count": str(config.frame_count),
                    "ffmpeg_version": ffmpeg_version(),
                },
            )
            for frame_index in range(config.frame_count):
                frame_path = working_directory / f"frame_{frame_index + 1:010d}.jpg"
                if not frame_path.is_file():
                    raise RuntimeError(
                        f"expected {config.frame_count} video samples, decoded only {frame_index}"
                    )
                timestamp_ns = _sample_timestamp_ns(config, frame_index)
                message = CompressedImage()
                message.timestamp.seconds, message.timestamp.nanos = divmod(
                    timestamp_ns, NANOSECONDS_PER_SECOND
                )
                message.frame_id = config.camera_name
                message.format = "jpeg"
                message.data = frame_path.read_bytes()
                writer.add_message(
                    channel_id=channel_id,
                    log_time=timestamp_ns,
                    publish_time=timestamp_ns,
                    sequence=frame_index,
                    data=message.SerializeToString(),
                )
                frame_path.unlink()
            writer.finish()
        # A hard link is an atomic create-if-absent on the same filesystem;
        # rename/replace would overwrite a concurrent caller's finished file.
        os.link(staged_episode, output_path)
    return output_path
