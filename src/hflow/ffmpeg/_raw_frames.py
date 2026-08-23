"""Streaming full-rate luma frames, for checks that compare frame to frame.

``Episode.frames()`` resamples to a declared rate and re-encodes to JPEG, which
is right for handing pictures to a model and wrong for measuring motion: it
skips the adjacent pairs motion lives in, and its quantization moves pixels the
measurement is about. This decodes every frame to raw 8-bit luma and yields it
without buffering the clip, so a long episode costs one frame of memory rather
than all of them.
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


@contextmanager
def luma_frames(video: Path) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of ``(height, width)`` uint8 luma frames, in order.

    A context manager because it owns a subprocess: leaving the block closes the
    pipe and reaps ffmpeg even when the caller stops reading early, which a bare
    generator cannot promise. A truncated final frame is an error rather than a
    silently short clip -- a motion measurement over a frame that was half read
    is worse than no measurement.
    """
    frame_height, frame_width = _probe_frame_shape(video)
    frame_bytes = frame_height * frame_width
    command = [
        str(ffmpeg_path()),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        # gray is the luma plane as-is for the yuv420p canonical episodes carry,
        # so no scaling or colour conversion stands between the file and the
        # measurement.
        "-vf",
        "format=pix_fmts=gray",
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
                    f"{frame_bytes} bytes for {frame_width}x{frame_height}"
                )
            yield np.frombuffer(payload, dtype=np.uint8).reshape(frame_height, frame_width)

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
