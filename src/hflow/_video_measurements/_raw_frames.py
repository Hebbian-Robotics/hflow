"""Stream decoded frames without resampling or image re-encoding."""

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ._toolchain import VideoMeasurementToolchain


class RawFrameError(RuntimeError):
    """A video could not be probed or decoded into complete raw frames."""


LUMA_FRAME_FILTER_GRAPH = "format=pix_fmts=gray"


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
            "csv=p=0:s=x",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode != 0:
        raise RawFrameError(f"ffprobe failed for {video}: {completed_process.stderr.strip()}")
    reported_dimensions = (
        completed_process.stdout.strip().splitlines()[0].split("x")
        if completed_process.stdout
        else []
    )
    if len(reported_dimensions) != 2:
        raise RawFrameError(f"ffprobe reported no video dimensions for {video}")
    frame_width, frame_height = (int(value) for value in reported_dimensions)
    if frame_width <= 0 or frame_height <= 0:
        raise RawFrameError(f"{video} reports a {frame_width}x{frame_height} frame")
    return frame_height, frame_width


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
) -> Iterator[Iterator[np.ndarray]]:
    """Yield complete ``uint8`` frames from one owned FFmpeg process."""
    frame_byte_count = int(np.prod(frame_shape))
    command = [
        str(toolchain.ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        filter_graph,
        "-f",
        "rawvideo",
        "-",
    ]
    decoding_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        if decoding_process.stdout is not None:
            decoding_process.stdout.close()
        standard_error = (
            decoding_process.stderr.read().decode() if decoding_process.stderr is not None else ""
        )
        if decoding_process.stderr is not None:
            decoding_process.stderr.close()
        return_code = decoding_process.wait()
        if reached_end_of_stream and return_code != 0:
            raise RawFrameError(f"decode failed for {video}: {standard_error.strip()}")


@contextmanager
def luma_frames(
    video: Path, *, toolchain: VideoMeasurementToolchain
) -> Iterator[Iterator[np.ndarray]]:
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
) -> Iterator[Iterator[np.ndarray]]:
    """Yield contiguous ``(height, width, 3)`` 8-bit RGB frames."""
    if frames_per_second is not None and frames_per_second <= 0:
        raise ValueError(f"frames_per_second must be positive, got {frames_per_second}")
    frame_height, frame_width = _scaled_frame_shape(
        _probe_frame_shape(video, toolchain), long_edge_pixels
    )
    filters: list[str] = []
    if frames_per_second is not None:
        filters.append(f"fps={frames_per_second:g}")
    if long_edge_pixels is not None:
        filters.append(f"scale={frame_width}:{frame_height}:flags=bilinear")
    filters.append("format=pix_fmts=rgb24")
    with _raw_frame_stream(
        video,
        toolchain=toolchain,
        filter_graph=",".join(filters),
        frame_shape=(frame_height, frame_width, 3),
    ) as frames:
        yield frames
