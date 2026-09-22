"""Shared ffmpeg and ffprobe helpers for tests that render or inspect media."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hflow.episode import ExtractedFrame
from hflow.ffmpeg import ffmpeg_path, ffprobe_path

EXTRACTED_FRAME_STREAM_START_NS = 1_755_000_000_000_000_000


def _run_tool(
    command: list[str], *, stdin: bytes | None, timeout_seconds: float | None
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        command, input=stdin, capture_output=True, check=False, timeout=timeout_seconds
    )
    if completed.returncode != 0:
        error = subprocess.CalledProcessError(
            completed.returncode, command, completed.stdout, completed.stderr
        )
        # CalledProcessError's message omits stderr, which is the only place
        # ffmpeg says why a fixture failed to render.
        error.add_note(completed.stderr.decode(errors="replace"))
        raise error
    return completed


def run_ffmpeg(
    *arguments: str, stdin: bytes | None = None, timeout_seconds: float | None = None
) -> bytes:
    """Run the resolved ffmpeg with ``arguments`` and return its stdout."""
    command = [str(ffmpeg_path()), *arguments]
    return _run_tool(command, stdin=stdin, timeout_seconds=timeout_seconds).stdout


def run_ffprobe(*arguments: str, timeout_seconds: float | None = None) -> str:
    """Run the resolved ffprobe with ``arguments`` and return its stdout as text."""
    command = [str(ffprobe_path()), *arguments]
    return _run_tool(command, stdin=None, timeout_seconds=timeout_seconds).stdout.decode()


def render_lavfi(output: Path, *lavfi_sources: str, output_arguments: tuple[str, ...]) -> Path:
    """Render one or more lavfi sources to ``output``.

    Each source becomes its own ``-f lavfi -i`` input, in order, so
    ``output_arguments`` can refer to them as ``[0:v]``, ``[1:v]`` and so on.
    Every flag that shapes the media (codec, preset, pixel format, GOP,
    filters) is the caller's, spelled out in ``output_arguments``.
    """
    lavfi_inputs = [
        argument for source in lavfi_sources for argument in ("-f", "lavfi", "-i", source)
    ]
    run_ffmpeg(
        "-hide_banner", "-loglevel", "error", "-y", *lavfi_inputs, *output_arguments, str(output)
    )
    return output


def probe_video_stream(
    path: Path, *stream_entries: str, count_frames: bool = False
) -> dict[str, str]:
    """Read ``stream_entries`` (for example ``width``) from the first video stream."""
    count_frames_arguments = ("-count_frames",) if count_frames else ()
    output = run_ffprobe(
        "-v",
        "error",
        "-select_streams",
        "v:0",
        *count_frames_arguments,
        "-show_entries",
        "stream=" + ",".join(stream_entries),
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    )
    stream_fields: dict[str, str] = {}
    for output_line in output.splitlines():
        field_name, _, field_value = output_line.partition("=")
        stream_fields[field_name.strip()] = field_value.strip()
    return stream_fields


def decoded_frame_count(path: Path) -> int:
    """How many frames ffprobe decodes from the first video stream."""
    return int(probe_video_stream(path, "nb_read_frames", count_frames=True)["nb_read_frames"])


def video_stream_dimensions(path: Path) -> tuple[int, int]:
    """``(width, height)`` of the first video stream, or of a still image."""
    stream_fields = probe_video_stream(path, "width", "height")
    return int(stream_fields["width"]), int(stream_fields["height"])


def make_extracted_frames(directory: Path, *, count: int, size: str) -> list[ExtractedFrame]:
    """``count`` testsrc2 JPEG frames of ``size``, stamped one second apart."""
    directory.mkdir(parents=True, exist_ok=True)
    render_lavfi(
        directory / "frame_%02d.jpg",
        f"testsrc2=size={size}:rate=1:duration={count}",
        output_arguments=("-q:v", "2"),
    )
    frame_paths = sorted(directory.glob("frame_*.jpg"))
    assert len(frame_paths) == count
    return [
        ExtractedFrame(
            path=frame_path,
            log_time_ns=EXTRACTED_FRAME_STREAM_START_NS + index * 1_000_000_000,
        )
        for index, frame_path in enumerate(frame_paths)
    ]
