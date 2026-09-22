from __future__ import annotations

import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from media_test_helpers import decoded_frame_count, run_ffmpeg

import hflow.video as video

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_requires_system_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="system ffmpeg/ffprobe required for remux concurrency regression",
)


@_requires_system_ffmpeg
def test_concurrent_remuxes_match_single_process_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _FFMPEG is not None
    system_ffmpeg = Path(_FFMPEG)
    monkeypatch.setattr(video, "ffmpeg_path", lambda: system_ffmpeg)

    encoded = run_ffmpeg(
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x120:rate=30:duration=2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-bf",
        "0",
        "-g",
        "30",
        "-f",
        "h264",
        "-",
    )

    reference = video.write_access_units_to_mp4(
        [encoded], fps=30.0, output=tmp_path / "reference.mp4"
    )
    output = tmp_path / "camera.mp4"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(video.write_access_units_to_mp4, [encoded], fps=30.0, output=output)
            for _ in range(2)
        ]
        for future in futures:
            assert future.result() == output

    assert (
        hashlib.sha256(output.read_bytes()).digest()
        == hashlib.sha256(reference.read_bytes()).digest()
    )
    assert decoded_frame_count(output) == 60
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
