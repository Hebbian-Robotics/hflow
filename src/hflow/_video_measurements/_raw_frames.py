"""Stream decoded frames without resampling or image re-encoding."""

import hashlib
import json
import subprocess
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from ._toolchain import VideoMeasurementToolchain


class RawFrameError(RuntimeError):
    """A video could not be probed or decoded into complete raw frames."""


LUMA_FRAME_FILTER_GRAPH = "format=pix_fmts=gray"


@dataclass(frozen=True)
class SampledRGBFrame:
    """One sampled RGB frame and the source positions that identify it."""

    sample_index: int
    source_frame_index: int
    pixels: np.ndarray


def _probe_frame_shape(video: Path, toolchain: VideoMeasurementToolchain) -> tuple[int, int]:
    """Return the coded ``(height, width)`` of the first video stream."""
    completed_process = subprocess.run(
        [
            str(toolchain.ffprobe_executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode != 0:
        raise RawFrameError(f"ffprobe failed for {video}: {completed_process.stderr.strip()}")
    # CSV also appends stream side data (for example display rotation). Read
    # named fields so that metadata cannot change the dimension record's shape.
    try:
        stream = json.loads(completed_process.stdout)["streams"][0]
        frame_width, frame_height = stream["width"], stream["height"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        raise RawFrameError(f"ffprobe reported no video dimensions for {video}") from None
    if (
        type(frame_width) is not int
        or type(frame_height) is not int
        or frame_width <= 0
        or frame_height <= 0
    ):
        raise RawFrameError(f"ffprobe reported invalid video dimensions for {video}")
    return frame_height, frame_width


def _probe_frame_rate(video: Path, toolchain: VideoMeasurementToolchain) -> float:
    """Return the stream's declared constant frame rate."""
    completed_process = subprocess.run(
        [
            str(toolchain.ffprobe_executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode != 0:
        raise RawFrameError(f"ffprobe failed for {video}: {completed_process.stderr.strip()}")
    try:
        stream = json.loads(completed_process.stdout)["streams"][0]
        rates = [Fraction(stream[key]) for key in ("avg_frame_rate", "r_frame_rate")]
        rate = next(candidate for candidate in rates if candidate > 0)
    except (
        json.JSONDecodeError,
        KeyError,
        IndexError,
        StopIteration,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        raise RawFrameError(f"ffprobe reported no constant frame rate for {video}") from None
    if rate <= 0:
        raise RawFrameError(f"ffprobe reported no constant frame rate for {video}")
    return float(rate)


def _sample_source_frame_index(sample_index: int, source_fps: float, sample_fps: float) -> int:
    """Map an output tick to the source frame visible at that tick."""
    return int(sample_index * source_fps / sample_fps + 1e-6)


def _decoded_frame_hashes(
    video: Path,
    *,
    toolchain: VideoMeasurementToolchain,
    filter_graph: str,
    source_fps: float,
    sample_fps: float | None,
) -> tuple[list[int], dict[int, set[str]]]:
    """Trace decoded RGB digests without writing frame data or a sidecar file."""
    command = [
        str(toolchain.ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        filter_graph,
        "-c:v",
        "rawvideo",
        "-enc_time_base",
        "1:1000000",
        "-hash",
        "sha256",
        "-fps_mode",
        "passthrough",
        "-f",
        "framemd5",
        "-",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    time_base: Fraction | None = None
    origin: Fraction | None = None
    decoded_indices: list[int] = []
    hashes: dict[int, set[str]] = {}
    reached_end_of_stream = False
    try:
        if process.stdout is None:
            raise RawFrameError(f"ffmpeg did not provide a frame hash stream for {video}")
        for line in process.stdout:
            if line.startswith("#tb 0:"):
                time_base = Fraction(line.partition(":")[2].strip())
            elif not line.startswith("#") and line.strip():
                fields = line.split(",")
                if time_base is None or len(fields) < 6 or fields[0].strip() != "0":
                    raise ValueError("invalid frame hash record")
                timestamp = int(fields[2].strip()) * time_base
                if origin is None:
                    origin = timestamp
                source_index = round((timestamp - origin) * Fraction(str(source_fps)))
                digest = fields[5].strip()
                if sample_fps is None:
                    decoded_indices.append(source_index)
                elif sample_fps >= source_fps:
                    hashes.setdefault(source_index, set()).add(digest)
                else:
                    center = int(source_index * sample_fps / source_fps)
                    for sample_index in range(max(0, center - 1), center + 2):
                        if (
                            _sample_source_frame_index(sample_index, source_fps, sample_fps)
                            == source_index
                        ):
                            hashes.setdefault(source_index, set()).add(digest)
        reached_end_of_stream = True
    except (OSError, ValueError, ZeroDivisionError) as error:
        if process.poll() is None:
            process.terminate()
        raise RawFrameError(f"Invalid decoded frame hash trace for {video}: {error}") from error
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if not reached_end_of_stream and process.poll() is None:
            process.terminate()
        return_code = process.wait()

    if return_code != 0:
        raise RawFrameError(f"ffmpeg failed to trace decoded frames for {video}")
    if origin is None:
        raise RawFrameError(f"No decoded frame timestamps for {video}")
    return decoded_indices, hashes


def _scaled_frame_shape(
    source_shape: tuple[int, int], long_edge_pixels: int | None
) -> tuple[int, int]:
    """Resize ``(height, width)`` proportionally with half-up rounding."""
    if long_edge_pixels is None:
        return source_shape
    if long_edge_pixels < 2:
        raise ValueError(f"long_edge_pixels must be at least 2, got {long_edge_pixels}")
    source_height, source_width = source_shape
    resize_scale = long_edge_pixels / max(source_height, source_width)
    return (
        max(1, int(source_height * resize_scale + 0.5)),
        max(1, int(source_width * resize_scale + 0.5)),
    )


@contextmanager
def _raw_frame_stream(
    video: Path,
    *,
    toolchain: VideoMeasurementToolchain,
    filter_graph: str,
    frame_shape: tuple[int, ...],
) -> Generator[Iterator[np.ndarray], None, None]:
    """Yield complete ``uint8`` frames from one owned FFmpeg process."""
    frame_byte_count = int(np.prod(frame_shape))
    command = [
        str(toolchain.ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        filter_graph,
        "-f",
        "rawvideo",
        "-",
    ]
    # Undrained stderr can fill its pipe and deadlock a slow/abandoned frame
    # consumer. Decoder diagnostics are not part of the measurement contract.
    decoding_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    reached_end_of_stream = False

    def read_frames() -> Iterator[np.ndarray]:
        nonlocal reached_end_of_stream
        assert decoding_process.stdout is not None
        while True:
            frame_payload = decoding_process.stdout.read(frame_byte_count)
            if not frame_payload:
                reached_end_of_stream = True
                return
            if len(frame_payload) != frame_byte_count:
                raise RawFrameError(
                    f"{video} produced a truncated final frame: {len(frame_payload)} of "
                    f"{frame_byte_count} bytes for {frame_shape}"
                )
            yield np.frombuffer(frame_payload, dtype=np.uint8).reshape(frame_shape)

    try:
        yield read_frames()
    finally:
        if not reached_end_of_stream:
            decoding_process.terminate()
        if decoding_process.stdout is not None:
            decoding_process.stdout.close()
        try:
            return_code = decoding_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            decoding_process.kill()
            decoding_process.wait()
            if reached_end_of_stream:
                raise RawFrameError("decoder did not exit after its output ended") from None
            return_code = decoding_process.returncode
        if reached_end_of_stream and return_code != 0:
            raise RawFrameError(f"decode failed for {video}")


@contextmanager
def luma_frames(
    video: Path, *, toolchain: VideoMeasurementToolchain
) -> Generator[Iterator[np.ndarray], None, None]:
    """Yield coded-size ``(height, width)`` 8-bit luma frames in order."""
    frame_shape = _probe_frame_shape(video, toolchain)
    with _raw_frame_stream(
        video,
        toolchain=toolchain,
        filter_graph=LUMA_FRAME_FILTER_GRAPH,
        frame_shape=frame_shape,
    ) as frames:
        yield frames


@contextmanager
def rgb_frames(
    video: Path,
    *,
    toolchain: VideoMeasurementToolchain,
    frames_per_second: float | None = None,
    long_edge_pixels: int | None = None,
    source_fps: float | None = None,
) -> Generator[Iterator[SampledRGBFrame], None, None]:
    """Yield sampled RGB frames with their original sample and source positions.

    A first decode records picture digests for the source frames requested by
    the sampling ticks. A second, streaming decode emits candidate samples;
    comparing their digests filters out frames the ``fps`` filter repeated to
    fill decoder gaps. No full-video pixel buffer or temporary file is needed.
    ``sample_index`` preserves the original tick so callers can keep labels
    aligned when a tick is omitted.
    """
    if frames_per_second is not None and frames_per_second <= 0:
        raise ValueError(f"frames_per_second must be positive, got {frames_per_second}")
    source_height, source_width = _probe_frame_shape(video, toolchain)
    frame_height, frame_width = _scaled_frame_shape((source_height, source_width), long_edge_pixels)
    if source_fps is None:
        source_fps = _probe_frame_rate(video, toolchain)
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")

    decoded_filters: list[str] = []
    sampled_filters: list[str] = []
    if frames_per_second is not None:
        sampled_filters.append(f"fps={frames_per_second:g}:round=up")
    if long_edge_pixels is not None:
        resize_filter = f"scale={frame_width}:{frame_height}:flags=bilinear"
        decoded_filters.append(resize_filter)
        sampled_filters.append(resize_filter)
    decoded_filters.append("format=pix_fmts=rgb24")
    sampled_filters.append("format=pix_fmts=rgb24")
    decoded_filter_graph = ",".join(decoded_filters)
    sampled_filter_graph = ",".join(sampled_filters)
    frame_shape = (frame_height, frame_width, 3)
    decoded_indices, decoded_hashes = _decoded_frame_hashes(
        video,
        toolchain=toolchain,
        filter_graph=decoded_filter_graph,
        source_fps=source_fps,
        sample_fps=frames_per_second,
    )

    with _raw_frame_stream(
        video,
        toolchain=toolchain,
        filter_graph=sampled_filter_graph,
        frame_shape=frame_shape,
    ) as candidate_frames:

        def read_frames() -> Iterator[SampledRGBFrame]:
            if frames_per_second is None:
                sample_count = 0
                for sample_index, pixels in enumerate(candidate_frames):
                    if sample_index >= len(decoded_indices):
                        raise RawFrameError(f"ffmpeg produced extra RGB frames for {video}")
                    sample_count += 1
                    yield SampledRGBFrame(
                        sample_index=sample_index,
                        source_frame_index=decoded_indices[sample_index],
                        pixels=pixels,
                    )
                if len(decoded_indices) != sample_count:
                    raise RawFrameError(
                        f"ffmpeg produced {sample_count} RGB frames but recorded "
                        f"{len(decoded_indices)} decoded frame timestamps for {video}"
                    )
                return

            for sample_index, pixels in enumerate(candidate_frames):
                source_frame_index = _sample_source_frame_index(
                    sample_index, source_fps, frames_per_second
                )
                if hashlib.sha256(memoryview(pixels)).hexdigest() not in decoded_hashes.get(
                    source_frame_index, set()
                ):
                    continue
                yield SampledRGBFrame(
                    sample_index=sample_index,
                    source_frame_index=source_frame_index,
                    pixels=pixels,
                )

        yield read_frames()
