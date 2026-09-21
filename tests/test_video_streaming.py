"""Bounded remux consumption and cleanup at the subprocess boundary."""

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import pytest

from hflow import video

# A parseable P slice larger than the stdin buffer. The process double consumes
# bytes, while test_video exercises the same remux with real H.264 and ffmpeg.
UNIT = b"\x00\x00\x00\x01\x41\xf0" + b"x" * (64 * 1024)


@pytest.fixture
def remux_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[subprocess.Popen[bytes]]]:
    progress = tmp_path / "progress"
    executable = tmp_path / "ffmpeg"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        f"progress = pathlib.Path({str(progress)!r})\n"
        "with open(sys.argv[-1], 'wb') as output:\n"
        "    count = 0\n"
        "    while True:\n"
        f"        unit = sys.stdin.buffer.read({len(UNIT)})\n"
        "        if not unit:\n"
        "            break\n"
        "        output.write(unit)\n"
        "        output.flush()\n"
        "        count += len(unit)\n"
        "        progress.write_text(str(count))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setattr(video, "ffmpeg_path", lambda: executable)
    processes: list[subprocess.Popen[bytes]] = []
    original_popen = subprocess.Popen

    def start_process(
        command: list[str], *, stdin: int, stdout: int, stderr: IO[bytes]
    ) -> subprocess.Popen[bytes]:
        process = original_popen(command, stdin=stdin, stdout=stdout, stderr=stderr)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start_process)
    return progress, processes


def _wait_for_consumption(progress: Path, byte_count: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if progress.exists() and progress.read_text() == str(byte_count):
            return
        time.sleep(0.01)
    pytest.fail("the subprocess did not consume the unit before the next was requested")


def test_remux_consumes_each_unit_before_requesting_the_next(
    remux_process: tuple[Path, list[subprocess.Popen[bytes]]], tmp_path: Path
) -> None:
    progress, processes = remux_process
    output = tmp_path / "video.mp4"

    def units() -> Iterator[bytes]:
        for index in range(4):
            yield UNIT
            _wait_for_consumption(progress, (index + 1) * len(UNIT))
            assert not output.exists()

    assert video.write_access_units_to_mp4(units(), fps=30, output=output) == output
    assert output.read_bytes() == UNIT * 4
    assert processes[0].returncode == 0
    assert not list(tmp_path.glob(".video.mp4.*.tmp"))


@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize("failure", ["iterator", "malformed", "orphan_slice", "b_frame"])
def test_streaming_failure_reaps_process_and_preserves_atomic_output(
    remux_process: tuple[Path, list[subprocess.Popen[bytes]]],
    tmp_path: Path,
    existing_output: bool,
    failure: str,
) -> None:
    progress, processes = remux_process
    output = tmp_path / "video.mp4"
    if existing_output:
        output.write_bytes(b"previous completed MP4")

    def units() -> Iterator[bytes]:
        # The orphan slice must be the first picture; start streaming a non-VCL
        # NAL first so the validation failure still happens after a write.
        yield UNIT if failure != "orphan_slice" else b"\x00\x00\x00\x01\x09" + UNIT[5:]
        _wait_for_consumption(progress, len(UNIT))
        if failure == "iterator":
            raise RuntimeError("source decode failed")
        if failure == "malformed":
            yield b"\x00\x00\x00\x01\x41\x00"
        elif failure == "orphan_slice":
            yield b"\x00\x00\x00\x01\x41\x58"  # first_mb=1, slice_type=P
        else:
            yield b"\x00\x00\x00\x01\x41\xa0"  # first_mb=0, slice_type=B

    expected_error = RuntimeError if failure == "iterator" else ValueError
    message = {
        "iterator": "source decode failed",
        "malformed": "incomplete or truncated",
        "orphan_slice": "before any picture starts",
        "b_frame": r"1 B picture\(s\) across 2 \(reorder depth 1\)",
    }[failure]
    with pytest.raises(expected_error, match=message):
        video.write_access_units_to_mp4(units(), fps=30, output=output)
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdin is not None and processes[0].stdin.closed
    assert not list(tmp_path.glob(".video.mp4.*.tmp"))
    if existing_output:
        assert output.read_bytes() == b"previous completed MP4"
    else:
        assert not output.exists()


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("early_exit", [False, True])
def test_ffmpeg_failure_removes_partial_output_and_reports_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int, early_exit: bool
) -> None:
    executable = tmp_path / "ffmpeg"
    # More than a pipe buffer of stderr before reading stdin catches deadlocks.
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.write('x' * 200_000 + 'remux diagnostic')\n"
        "sys.stderr.flush()\n"
        + ("" if early_exit else "sys.stdin.buffer.read()\n")
        + f"sys.exit({exit_code})\n"
    )
    executable.chmod(0o755)
    monkeypatch.setattr(video, "ffmpeg_path", lambda: executable)
    output = tmp_path / "video.mp4"
    message = (
        "exited 0 but produced no output"
        if exit_code == 0 and not early_exit
        else f"exit {exit_code}.*remux diagnostic"
    )
    with pytest.raises(video.VideoEncodeError, match=message):
        video.write_access_units_to_mp4(iter((UNIT,) * 4), fps=30, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".video.mp4.*.tmp"))


def test_process_start_failure_removes_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video, "ffmpeg_path", lambda: tmp_path / "missing-ffmpeg")
    output = tmp_path / "video.mp4"
    with pytest.raises(FileNotFoundError):
        video.write_access_units_to_mp4(iter((UNIT,)), fps=30, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".video.mp4.*.tmp"))


@pytest.mark.parametrize(
    "timestamps", [[], [1], [1, 1], [3, 2, 1], [0, 10], [0, 10, 30], [0, 10, 10, 90, 95]]
)
def test_streaming_fps_preserves_exact_median_and_errors(timestamps: list[int]) -> None:
    try:
        expected = video.estimate_fps_from_log_times(timestamps, topic="/camera")
    except ValueError as error:
        with pytest.raises(ValueError) as streaming_error:
            video.estimate_fps_from_streaming_log_times(iter(timestamps), topic="/camera")
        assert str(streaming_error.value) == str(error)
    else:
        assert (
            video.estimate_fps_from_streaming_log_times(iter(timestamps), topic="/camera")
            == expected
        )
