"""Atomicity contracts for the contact-sheet writer (issue #556).

``contact_sheet`` must stage through a per-call sibling temp file and promote
it with an atomic rename only after ffmpeg succeeds: a SIGKILL mid-write must
never leave a partial JPEG where the publish step would catalog it, and
concurrent writers to one path must not share a temp file.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from hflow.episode import ExtractedFrame
from hflow.ffmpeg import _contact_sheet
from hflow.ffmpeg._contact_sheet import contact_sheet

_STREAM_START_NS = 1_755_000_000_000_000_000


def _system_ffmpeg() -> str:
    ffmpeg_binary = shutil.which("ffmpeg")
    assert ffmpeg_binary is not None, "ffmpeg required on PATH for these tests"
    return ffmpeg_binary


def _system_ffprobe() -> str:
    ffprobe_binary = shutil.which("ffprobe")
    assert ffprobe_binary is not None, "ffprobe required on PATH for these tests"
    return ffprobe_binary


def _make_frames(directory: Path, *, count: int, size: str) -> list[ExtractedFrame]:
    """``count`` JPEG frames rendered from lavfi, 1s apart."""
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _system_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}:rate=1:duration={count}",
            "-q:v",
            "2",
            str(directory / "frame_%02d.jpg"),
        ],
        capture_output=True,
        check=True,
    )
    frame_paths = sorted(directory.glob("frame_*.jpg"))
    assert len(frame_paths) == count
    return [
        ExtractedFrame(path=frame_path, log_time_ns=_STREAM_START_NS + index * 1_000_000_000)
        for index, frame_path in enumerate(frame_paths)
    ]


def _assert_fully_valid_jpeg(path: Path) -> None:
    """A complete JPEG: SOI/EOI markers present and fully decodable by ffprobe."""
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8", f"{path} is missing its JPEG SOI marker"
    assert data[-2:] == b"\xff\xd9", f"{path} is missing its JPEG EOI marker (truncated?)"
    completed = subprocess.run(
        [
            _system_ffprobe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"ffprobe cannot decode {path}: {completed.stderr}"


def _run_sheet_with_ffmpeg_sigkilled(
    frames: list[ExtractedFrame], output: Path, monkeypatch: pytest.MonkeyPatch
) -> bool:
    """Run ``contact_sheet`` while SIGKILLing its real ffmpeg child mid-write.

    The fake ``subprocess.run`` launches the exact command the writer built,
    freezes the child with SIGSTOP the instant its (temp) output holds bytes,
    then SIGKILLs it -- so genuine partial bytes exist on disk when the writer
    takes its failure path. Returns True only when the child was stopped while
    alive and then killed (returncode < 0).
    """
    signalled = False
    real_run = subprocess.run

    def kill_mid_write(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        nonlocal signalled
        if "-frames:v" not in command:
            # Font probing (fc-match) and the drawtext capability check share
            # the module's subprocess handle; only the sheet render is killed.
            return real_run(command, capture_output=capture_output, text=text, check=check)
        temporary_output = Path(command[-1])
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                if temporary_output.is_file() and temporary_output.stat().st_size > 0:
                    # Partial bytes are on disk and the child is still
                    # running: freeze it here so the kill lands mid-write.
                    process.send_signal(signal.SIGSTOP)
                    signalled = True
                    break
                time.sleep(0.001)
            if process.poll() is None:
                process.kill()
            _stdout, stderr_bytes = process.communicate()
        returncode = process.returncode
        assert returncode is not None
        if returncode == 0:
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")
        signalled = signalled and returncode < 0
        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout="",
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(_contact_sheet.subprocess, "run", kill_mid_write)
        with pytest.raises(RuntimeError, match="ffmpeg contact sheet failed"):
            contact_sheet(frames, output, columns=4, tile_width=640)
    return signalled


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGSTOP/SIGKILL required")
def test_sigkill_mid_write_never_leaves_partial_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _make_frames(tmp_path / "big-frames", count=8, size="1280x720")
    output = tmp_path / "sheet.jpg"

    assert _run_sheet_with_ffmpeg_sigkilled(frames, output, monkeypatch), (
        "ffmpeg finished before the kill landed; the test was vacuous"
    )
    assert not output.exists(), "a SIGKILL mid-write left bytes at the final path"
    assert not list(tmp_path.glob(".sheet.jpg.*.tmp")), "orphaned temp file after SIGKILL"

    # A pre-existing valid sheet must survive a later kill byte-identically:
    # ffmpeg only ever truncates the per-call temp, never the final path.
    contact_sheet(frames, output, columns=4, tile_width=640)
    _assert_fully_valid_jpeg(output)
    before = output.read_bytes()
    assert _run_sheet_with_ffmpeg_sigkilled(frames, output, monkeypatch), (
        "ffmpeg finished before the kill landed; the test was vacuous"
    )
    assert output.read_bytes() == before
    _assert_fully_valid_jpeg(output)
    assert not list(tmp_path.glob(".sheet.jpg.*.tmp")), "orphaned temp file after SIGKILL"


def test_concurrent_sheets_to_same_path_stay_valid(tmp_path: Path) -> None:
    """Two concurrent sheets to one path: per-call temps, last complete replace wins."""
    frames = _make_frames(tmp_path / "frames", count=8, size="640x480")
    output = tmp_path / "shared.jpg"
    barrier = Barrier(2)

    def render(_round_index: int) -> Path:
        barrier.wait(timeout=60)
        return contact_sheet(frames, output, columns=4, tile_width=320).path

    for round_index in range(5):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(render, round_index) for _ in range(2)]
            assert [future.result(timeout=120) for future in futures] == [output, output]
        _assert_fully_valid_jpeg(output)
    assert not list(tmp_path.glob(".shared.jpg.*.tmp")), "orphaned temp file after concurrency"
