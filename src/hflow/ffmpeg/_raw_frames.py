"""Streaming raw frames, for checks that need pixels rather than pictures.

``Episode.frames()`` resamples to a declared rate and re-encodes to JPEG, which
is right for handing pictures to a model over HTTP and wrong for measuring:
JPEG quantization moves the very pixels a measurement is about, and for
frame-to-frame work it also skips the adjacent pairs motion lives in. These
decode straight to raw planes and yield them without buffering the clip, so a
long episode costs one frame of memory rather than all of them.

Two shapes, because two kinds of check want different things:

- :func:`luma_frames` -- every frame, 8-bit luma, coded size. Frame-to-frame
  comparison (``hflow.motion``).
- :func:`rgb_frames` -- optionally resampled and resized, 8-bit RGB. Frames
  fed to a local model that wants colour (``hflow.mediapipe_hands``).
"""

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from hflow.ffmpeg._binary import ffmpeg_path


class RawFrameError(RuntimeError):
    pass


def _probe_frame_shape(video: Path) -> tuple[int, int]:
    """(height, width) of the video's coded frames, from ffprobe."""
    from hflow.ffmpeg._binary import ffprobe_path

    completed = subprocess.run(
        [
            str(ffprobe_path()),
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
    if completed.returncode != 0:
        raise RawFrameError(f"ffprobe failed for {video}: {completed.stderr.strip()}")
    dimensions = completed.stdout.strip().splitlines()[0].split("x") if completed.stdout else []
    if len(dimensions) != 2:
        raise RawFrameError(f"ffprobe reported no video dimensions for {video}")
    width, height = (int(value) for value in dimensions)
    if width <= 0 or height <= 0:
        raise RawFrameError(f"{video} reports a {width}x{height} frame")
    return height, width


def scaled_frame_shape(
    source_shape: tuple[int, int], long_edge_pixels: int | None
) -> tuple[int, int]:
    """Proportionally resize (height, width) so its long edge matches.

    Half-up rounding, and never below one pixel, so the arithmetic is
    reproducible rather than platform-dependent. ``None`` leaves the frame at
    its coded size. Upscaling is deliberately allowed: on small footage a
    larger inference frame changes what a detector finds at all, which makes
    the long edge part of the question rather than an optimization.
    """
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
    video: Path, *, filter_graph: str, frame_shape: tuple[int, ...]
) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of ``frame_shape`` uint8 frames from one ffmpeg pipe.

    A context manager because it owns a subprocess: leaving the block closes
    the pipe and reaps ffmpeg even when the caller stops reading early, which a
    bare generator cannot promise. A truncated final frame is an error rather
    than a silently short clip -- a measurement over a frame that was half read
    is worse than no measurement.
    """
    frame_bytes = int(np.prod(frame_shape))
    command = [
        str(ffmpeg_path()),
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
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Whether the caller read to end of stream. A caller that stops early makes
    # ffmpeg fail writing to a closed pipe, which is not a decode failure --
    # and the exit code cannot tell the two apart, since ffmpeg handles the
    # broken pipe itself and exits with an ordinary error status either way.
    reached_end_of_stream = False

    def read_frames() -> Iterator[np.ndarray]:
        nonlocal reached_end_of_stream
        assert process.stdout is not None
        while True:
            payload = process.stdout.read(frame_bytes)
            if not payload:
                reached_end_of_stream = True
                return
            if len(payload) != frame_bytes:
                raise RawFrameError(
                    f"{video} produced a truncated final frame: {len(payload)} of "
                    f"{frame_bytes} bytes for {frame_shape}"
                )
            yield np.frombuffer(payload, dtype=np.uint8).reshape(frame_shape)

    try:
        yield read_frames()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr_output = process.stderr.read().decode() if process.stderr is not None else ""
        if process.stderr is not None:
            process.stderr.close()
        return_code = process.wait()
        if reached_end_of_stream and return_code != 0:
            raise RawFrameError(f"decode failed for {video}: {stderr_output.strip()}")


@contextmanager
def luma_frames(video: Path) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of ``(height, width)`` uint8 luma frames, in order."""
    frame_shape = _probe_frame_shape(video)
    # gray is the luma plane as-is for the yuv420p canonical episodes carry, so
    # no scaling or colour conversion stands between the file and the
    # measurement.
    with _raw_frame_stream(
        video, filter_graph="format=pix_fmts=gray", frame_shape=frame_shape
    ) as frames:
        yield frames


@contextmanager
def rgb_frames(
    video: Path,
    *,
    fps: float | None = None,
    long_edge_pixels: int | None = None,
) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of ``(height, width, 3)`` contiguous uint8 RGB frames.

    ``fps`` resamples before any colour conversion or scaling, so sampling one
    frame a second from 30 fps footage converts one frame's pixels rather than
    thirty. ``long_edge_pixels`` resizes proportionally (see
    :func:`scaled_frame_shape`) with bilinear interpolation; both are omitted
    from the filter graph entirely when unset, so an unresampled, unresized
    call puts nothing between the file and the frame.

    The array layout -- contiguous, uint8, height by width by RGB -- is what
    local vision models take directly.
    """
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    frame_height, frame_width = scaled_frame_shape(_probe_frame_shape(video), long_edge_pixels)
    filters = []
    if fps is not None:
        filters.append(f"fps={fps:g}")
    if long_edge_pixels is not None:
        filters.append(f"scale={frame_width}:{frame_height}:flags=bilinear")
    filters.append("format=pix_fmts=rgb24")
    with _raw_frame_stream(
        video, filter_graph=",".join(filters), frame_shape=(frame_height, frame_width, 3)
    ) as frames:
        yield frames
