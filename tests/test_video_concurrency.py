from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import hflow.video as video


def test_concurrent_remuxes_use_distinct_temp_files(tmp_path, monkeypatch) -> None:
    output = tmp_path / "camera.mp4"
    barrier = threading.Barrier(2)
    seen: list[Path] = []
    seen_lock = threading.Lock()

    monkeypatch.setattr(video, "scan_picture_coding_types", lambda _stream: SimpleNamespace(b_picture_count=0))
    monkeypatch.setattr(video, "ffmpeg_path", lambda: Path("/usr/bin/ffmpeg"))

    def fake_run(command, *, input, capture_output):
        temp_path = Path(command[-1])
        with seen_lock:
            seen.append(temp_path)
        barrier.wait(timeout=5)
        temp_path.write_bytes(b"valid-mp4")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(video.subprocess, "run", fake_run)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(video.write_access_units_to_mp4, [b"frame"], fps=30.0, output=output) for _ in range(2)]
        for future in futures:
            assert future.result() == output

    assert len(seen) == 2
    assert len(set(seen)) == 2
    assert output.read_bytes() == b"valid-mp4"
    assert all(not path.exists() for path in seen)
