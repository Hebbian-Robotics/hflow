"""Import a local video excerpt as an input episode for the processing engine."""

import errno
import math
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap.writer import Writer
from mcap_protobuf.schema import build_file_descriptor_set

from hflow._field_guards import (
    require_finite_float,
    require_int_in_range,
    require_positive_int,
)
from hflow._pinned_asset import sha256_hex_of_file
from hflow.ffmpeg import ffmpeg_path, ffmpeg_version, media_input_was_rejected, run_media_command
from hflow.format import (
    CANONICAL_VIDEO_SCHEMA_NAME,
    GOP_SECONDS,
    METADATA_RECORD_EPISODE,
    NANOSECONDS_PER_SECOND,
)
from hflow.media import UnreadableVideo, UnsupportedVideo, VideoLimits, VideoProperties, probe_video
from hflow.transform import TransformConfig
from hflow.video import (
    AccessUnit,
    VideoEncodeError,
    _enforce_encode_guarantees,
    canonical_x264_parameters,
    split_annex_b_stream,
    write_access_units_to_mp4,
)

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
    ``maximum_encoded_bytes`` bounds the buffered Annex-B excerpt (default
    64 MiB, exclusive). Larger excerpts must be split or given an explicit
    higher budget. This is an encoded-byte limit, not a process RSS limit.
    """

    duration_s: float
    source_start_s: float = 0.0
    image_hz: float = 10.0
    image_width: int = 640
    image_height: int = 360
    camera_name: str = "camera"
    start_time_ns: int = 0
    metadata: tuple[tuple[str, str], ...] = ()
    maximum_encoded_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        require_positive_int(self.maximum_encoded_bytes, "maximum_encoded_bytes")
        for name, value in (
            ("duration_s", self.duration_s),
            ("source_start_s", self.source_start_s),
            ("image_hz", self.image_hz),
        ):
            require_finite_float(value, name)
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
            require_positive_int(value, name)
            # H.264 chroma subsampling needs even dimensions. Evenness is a
            # domain rule about this config, not a shared numeric invariant,
            # so it stays here with its own message instead of moving into
            # _field_guards (see #508).
            if value % 2:
                raise ValueError(f"{name} must be an even integer, got {value}")
        require_int_in_range(
            self.start_time_ns, "start_time_ns", minimum=0, maximum=_MAXIMUM_TIMESTAMP_NS
        )
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


class _UnreadableImport(RuntimeError):
    pass


class _UnsupportedExcerpt(ValueError):
    pass


def _require_excerpt_duration(properties: VideoProperties, config: VideoImportConfig) -> None:
    if config.source_start_s + config.duration_s > float(properties.duration_seconds) + 1e-6:
        raise _UnsupportedExcerpt("the requested excerpt extends past the source video")


def _excerpt_video_filter(config: VideoImportConfig) -> str:
    # Keep the preceding keyframe's negative, excerpt-relative timestamps.
    # Accurate seeking would discard the frame covering a non-frame-aligned
    # start, letting fps pad the excerpt with a later (potentially different)
    # frame. Resample before trimming so that preceding frame stays available.
    return (
        f"fps={config.image_hz}:start_time=0:round=up:eof_action=pass,"
        f"trim=duration={config.duration_s},"
        f"scale={config.image_width}:{config.image_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={config.image_width}:{config.image_height}:(ow-iw)/2:(oh-ih)/2:black"
    )


def _ffmpeg_excerpt_command(
    source_video: Path,
    config: VideoImportConfig,
    *,
    output_flags: list[str],
) -> list[str]:
    return [
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
        _excerpt_video_filter(config),
        "-frames:v",
        str(config.frame_count),
        *output_flags,
    ]


def _render_h264_access_units(
    source_video: Path,
    config: VideoImportConfig,
    working_directory: Path,
    limits: VideoLimits,
    *,
    transform_config: TransformConfig = TransformConfig(),
) -> list[AccessUnit]:
    """Transcode the excerpt to canonical in-band H.264 access units.

    One FFmpeg pass resample+scales the source and encodes libx264 with the
    same AUD / keyframe / no-B-frame contract as
    :func:`hflow.video.encode_images_to_h264`, so
    :func:`hflow.transform.write_canonical_episode` can pass the messages
    through without a JPEG intermediate.
    """
    fps = config.image_hz if config.frame_count > 1 else 1.0
    gop_seconds = (
        transform_config.gop_seconds
        if transform_config.gop_seconds is not None
        else GOP_SECONDS[transform_config.gop_preset]
    )
    gop_frames = max(1, round(gop_seconds * fps))
    x264_params = canonical_x264_parameters(gop_frames)
    annex_b_path = working_directory / "excerpt.h264"
    output_flags = [
        # Set encoder/VUI timing too: remux -r alone cannot change the
        # duration embedded in a single-picture H.264 stream.
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(transform_config.crf),
        "-x264-params",
        x264_params,
        "-f",
        "h264",
        # FFmpeg may overshoot by one packet; reject before reading it into
        # Python. The input/output dimension and timeout limits still apply.
        "-fs",
        str(config.maximum_encoded_bytes),
        str(annex_b_path),
    ]

    def run_encode(extra_output_flags: list[str]) -> None:
        if annex_b_path.exists():
            annex_b_path.unlink()
        completed = run_media_command(
            _ffmpeg_excerpt_command(
                source_video, config, output_flags=[*extra_output_flags, *output_flags]
            ),
            timeout_seconds=limits.timeout_seconds,
            maximum_output_bytes=limits.maximum_probe_bytes,
        )
        if media_input_was_rejected(completed):
            raise _UnreadableImport("could not decode source video")

    def read_units() -> list[AccessUnit]:
        if not annex_b_path.is_file():
            raise _UnreadableImport("could not decode source video")
        encoded_size = annex_b_path.stat().st_size
        if encoded_size >= config.maximum_encoded_bytes:
            raise _UnsupportedExcerpt("encoded video reaches maximum_encoded_bytes")
        with annex_b_path.open("rb") as stream:
            encoded = stream.read(encoded_size + 1)
        if len(encoded) >= config.maximum_encoded_bytes:
            raise _UnsupportedExcerpt("encoded video reaches maximum_encoded_bytes")
        try:
            return split_annex_b_stream(encoded)
        except ValueError as error:
            raise _UnreadableImport("could not decode source video") from error

    run_encode([])
    access_units = read_units()
    if len(access_units) != config.frame_count:
        # Some frame-rate conversions duplicate or drop frames to hit CFR;
        # passthrough forces one output frame per filtered sample.
        run_encode(["-fps_mode", "passthrough"])
        access_units = read_units()
    if len(access_units) != config.frame_count:
        raise _UnreadableImport(
            f"expected {config.frame_count} video samples, decoded only {len(access_units)}"
        )
    try:
        _enforce_encode_guarantees(
            access_units, expected_frame_count=config.frame_count, gop_frames=gop_frames
        )
    except VideoEncodeError as error:
        raise _UnreadableImport("could not decode source video") from error
    return access_units


def import_video_episode(
    source_video: Path | str,
    output: Path | str,
    config: VideoImportConfig,
    *,
    limits: VideoLimits = VideoLimits(),
    transform_config: TransformConfig = TransformConfig(),
) -> Path:
    """Import a local excerpt into an MCAP for :meth:`hflow.App.process`.

    The first video stream must have a known duration from stream metadata,
    a duration tag, or an unambiguous single-stream container. Missing,
    incomplete, corrupt, or out-of-range excerpts raise without
    publishing an output. URL inputs and network references are not read.
    FFmpeg uses HFlow's usual managed-binary policy.

    Landing messages are in-band H.264 ``foxglove.CompressedVideo`` on
    ``/{camera_name}/compressed``, encoded to the same Annex-B access-unit
    contract the canonical transform validates for pass-through (AUD, SPS/PPS
    on keyframes, no B-frames, first message is a keyframe). Container and
    codec variety stop at this importer: ``write_canonical_episode`` does not
    need a JPEG intermediate. The processing engine still owns provenance,
    grouping, and QC. Only caller-supplied fields enter ``episode/v1``.
    ``video_import/v1`` records source SHA-256, import settings, and the
    FFmpeg version.

    Pass the same ``transform_config`` to import and canonical processing:
    CRF and effective GOP seconds are committed here, recorded in import
    metadata, and checked before canonical pass-through. Incompatible settings
    require re-import from the source, not a second lossy transcode. Grouping
    and other non-encoding transform settings are still applied at SYNC.
    A single-picture stream is encoded at 1 Hz regardless of sampling rate.
    Encoded excerpts reaching ``config.maximum_encoded_bytes`` are unsupported.

    Conversion uses temporary disk beside ``output``. The complete output is
    published atomically without overwriting an existing path, including a
    concurrent publisher. Temporary files are cleaned on success and
    exception; the caller owns the published output and any later processing
    workspace.
    """
    source_video_path = Path(source_video).resolve()
    if not source_video_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(source_video_path))
    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(output_path))
    inspection = probe_video(source_video_path, limits=limits)
    match inspection:
        case UnreadableVideo():
            raise _UnreadableImport("could not inspect source video")
        case UnsupportedVideo():
            raise _UnsupportedExcerpt("source video exceeds supported limits")
        case VideoProperties():
            return _import_inspected_video(
                source_video_path, output_path, config, limits, inspection, transform_config
            )


def _import_inspected_video(
    source_video_path: Path,
    output_path: Path,
    config: VideoImportConfig,
    limits: VideoLimits,
    properties: VideoProperties,
    transform_config: TransformConfig,
) -> Path:
    if (
        config.image_width * config.image_height > limits.maximum_frame_pixels
        or config.image_hz > limits.maximum_frames_per_second
        or config.duration_s > limits.maximum_duration_seconds
    ):
        raise _UnsupportedExcerpt("requested video output exceeds supported limits")
    _require_excerpt_duration(properties, config)
    source_sha256 = sha256_hex_of_file(source_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".video-import-") as directory:
        working_directory = Path(directory)
        access_units = _render_h264_access_units(
            source_video_path, config, working_directory, limits, transform_config=transform_config
        )
        staged_episode = working_directory / "episode.mcap"
        with staged_episode.open("wb") as output_stream:
            writer = Writer(output_stream)
            writer.start(library="hflow video importer")
            schema_id = writer.register_schema(
                name=CANONICAL_VIDEO_SCHEMA_NAME,
                encoding="protobuf",
                data=build_file_descriptor_set(CompressedVideo).SerializeToString(),
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
                    "importer_version": "2",
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
                    "landing_format": "h264",
                    "crf": str(transform_config.crf),
                    "gop_seconds": str(
                        transform_config.gop_seconds
                        if transform_config.gop_seconds is not None
                        else GOP_SECONDS[transform_config.gop_preset]
                    ),
                    "maximum_encoded_bytes": str(config.maximum_encoded_bytes),
                },
            )
            for frame_index, access_unit in enumerate(access_units):
                timestamp_ns = _sample_timestamp_ns(config, frame_index)
                message = CompressedVideo()
                message.timestamp.seconds, message.timestamp.nanos = divmod(
                    timestamp_ns, NANOSECONDS_PER_SECOND
                )
                message.frame_id = config.camera_name
                message.format = "h264"
                message.data = access_unit.data
                writer.add_message(
                    channel_id=channel_id,
                    log_time=timestamp_ns,
                    publish_time=timestamp_ns,
                    sequence=frame_index,
                    data=message.SerializeToString(),
                )
            writer.finish()
        # A hard link is an atomic create-if-absent on the same filesystem;
        # rename/replace would overwrite a concurrent caller's finished file.
        os.link(staged_episode, output_path)
    return output_path


@dataclass(frozen=True)
class ImportedVideoEpisode:
    path: Path


def prepare_video_episode(
    source_video: Path,
    output: Path,
    config: VideoImportConfig,
    *,
    limits: VideoLimits = VideoLimits(),
    transform_config: TransformConfig = TransformConfig(),
) -> ImportedVideoEpisode | UnreadableVideo | UnsupportedVideo:
    """Import supported media with explicit rejection outcomes.

    Unlike rejection outcomes, operational errors propagate to the caller.
    Output and sampling semantics are identical to import_video_episode.
    """
    inspection = probe_video(source_video, limits=limits)
    if not isinstance(inspection, VideoProperties):
        return inspection
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError("episode output already exists")
        return ImportedVideoEpisode(
            _import_inspected_video(
                source_video.resolve(strict=True),
                output,
                config,
                limits,
                inspection,
                transform_config,
            )
        )
    except _UnreadableImport:
        return UnreadableVideo()
    except _UnsupportedExcerpt:
        return UnsupportedVideo()


def prepare_model_video(
    source_video: Path,
    output: Path,
    config: VideoImportConfig,
    *,
    limits: VideoLimits = VideoLimits(),
    transform_config: TransformConfig = TransformConfig(),
) -> Path | UnreadableVideo | UnsupportedVideo:
    """Prepare canonical model-input pixels directly, without an intermediate MCAP.

    Shares the importer's direct source-to-H.264 recipe (same filters, GOP, and
    CRF as :func:`import_video_episode` with the same transform settings), so
    decoded pixels match a SYNC of the imported landing episode. No catalog,
    episode or persistent workspace is created. Output is published atomically
    without replacing any existing path; the caller owns its lifetime.
    """
    source_video = source_video.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inspection = probe_video(source_video, limits=limits)
    if not isinstance(inspection, VideoProperties):
        return inspection
    if (
        config.image_width * config.image_height > limits.maximum_frame_pixels
        or config.image_hz > limits.maximum_frames_per_second
        or config.duration_s > limits.maximum_duration_seconds
    ):
        return UnsupportedVideo()
    try:
        _require_excerpt_duration(inspection, config)
    except _UnsupportedExcerpt:
        return UnsupportedVideo()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".model-video-") as directory:
        work_directory = Path(directory)
        try:
            access_units = _render_h264_access_units(
                source_video,
                config,
                work_directory,
                limits,
                transform_config=transform_config,
            )
        except _UnreadableImport:
            return UnreadableVideo()
        except _UnsupportedExcerpt:
            return UnsupportedVideo()
        frames_per_second = config.image_hz if len(access_units) >= 2 else 1.0
        staged_video = write_access_units_to_mp4(
            (access_unit.data for access_unit in access_units),
            fps=frames_per_second,
            output=work_directory / "video.mp4",
        )
        os.link(staged_video, output)
    return output


def prepare_model_frames(
    source_video: Path,
    output_directory: Path,
    config: VideoImportConfig,
    frame_indices: tuple[int, ...],
    *,
    limits: VideoLimits = VideoLimits(),
    transform_config: TransformConfig = TransformConfig(),
    jpeg_quality: int = 2,
) -> tuple[Path, ...] | UnreadableVideo | UnsupportedVideo:
    """Select JPEGs by index from the canonical model-video encoding.

    Indices address the fixed-rate prepared video, not source codec frames.
    The intermediate H.264 video remains task-local and is removed before return.
    No output directory is published for an unreadable or unsupported source.
    """
    if (
        not frame_indices
        or any(
            type(index) is not int or index < 0 or index >= config.frame_count
            for index in frame_indices
        )
        or tuple(sorted(set(frame_indices))) != frame_indices
    ):
        raise ValueError("frame indices must be distinct, ordered, and within the excerpt")
    if type(jpeg_quality) is not int or not 2 <= jpeg_quality <= 31:
        raise ValueError("JPEG quality must be between 2 and 31")
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_directory.parent, prefix=".model-frames-"
    ) as temporary_directory:
        work_directory = Path(temporary_directory)
        prepared_video = prepare_model_video(
            source_video,
            work_directory / "model.mp4",
            config,
            limits=limits,
            transform_config=transform_config,
        )
        if not isinstance(prepared_video, Path):
            return prepared_video
        frame_directory = work_directory / "frames"
        frame_directory.mkdir()
        selection_expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
        command_result = run_media_command(
            [
                str(ffmpeg_path()),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-xerror",
                "-protocol_whitelist",
                "file",
                "-i",
                str(prepared_video),
                "-vf",
                f"select={selection_expression}",
                "-fps_mode",
                "vfr",
                "-frames:v",
                str(len(frame_indices)),
                "-q:v",
                str(jpeg_quality),
                "-start_number",
                "0",
                str(frame_directory / "frame_%06d.jpg"),
            ],
            timeout_seconds=limits.timeout_seconds,
            maximum_output_bytes=limits.maximum_probe_bytes,
        )
        if command_result.returncode != 0:
            raise RuntimeError(f"Model frame extraction failed (exit {command_result.returncode})")
        frame_names = tuple(f"frame_{index:06d}.jpg" for index in range(len(frame_indices)))
        if not all((frame_directory / name).is_file() for name in frame_names):
            raise RuntimeError("Model frame extraction omitted selected frames")
        frame_directory.rename(output_directory)
    return tuple(output_directory / name for name in frame_names)
