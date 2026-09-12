"""Shared test setup.

Pins the whole suite to the system ffmpeg via the explicit override so tests
never trigger the pinned-build download (a per-machine, network-bound step).
Set at import time, before any test imports resolve the (cached) binary path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hflow.storage import BucketStorageRoot

_system_ffmpeg = shutil.which("ffmpeg")
if _system_ffmpeg is not None:
    os.environ.setdefault("HFLOW_FFMPEG", _system_ffmpeg)

# ``-fps_mode`` replaced ``-vsync`` in FFmpeg 5.1 (n5.1, 2022-07). Ubuntu
# 22.04 ships 4.4.2, which rejects the flag and turns encode tests into
# three assertion failures that look like a broken branch.
_MIN_FFMPEG_VERSION = (5, 1)


def _ffmpeg_version_tuple(version_line: str) -> tuple[int, int] | None:
    """Parse the major.minor from ``ffmpeg -version``'s first line."""
    match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", version_line)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


@pytest.fixture(scope="session", autouse=True)
def _require_ffmpeg_fps_mode() -> None:
    """Fail once, with versions, if the suite ffmpeg is older than 5.1."""
    if _system_ffmpeg is None:
        return
    completed = subprocess.run(
        [_system_ffmpeg, "-version"],
        check=False,
        capture_output=True,
        text=True,
    )
    version_line = (completed.stdout or completed.stderr).splitlines()[0] if (
        completed.stdout or completed.stderr
    ) else ""
    parsed = _ffmpeg_version_tuple(version_line)
    if parsed is None:
        return
    if parsed < _MIN_FFMPEG_VERSION:
        required = ".".join(str(part) for part in _MIN_FFMPEG_VERSION)
        found = ".".join(str(part) for part in parsed)
        pytest.exit(
            f"ffmpeg {found} does not support -fps_mode "
            f"(need {required} or newer; found {version_line!r}). "
            "Upgrade ffmpeg or unset HFLOW_FFMPEG so the pinned build can be used.",
            returncode=1,
        )


@pytest.fixture
def bucket_over_tmp(tmp_path: Path) -> tuple[BucketStorageRoot, Path]:
    """A real bucket root over obstore's local backend and its remote dir."""
    pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")
    from hflow.storage import BucketStorageRoot

    remote_dir = tmp_path / "bucket"
    remote_dir.mkdir(parents=True, exist_ok=True)
    root = BucketStorageRoot(f"file://{remote_dir}", mirror=tmp_path / "bucket-mirror")
    return root, remote_dir
