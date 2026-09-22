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


def _suite_ffmpeg() -> str | None:
    """ffmpeg the suite actually uses: HFLOW_FFMPEG if set, else PATH."""
    override = os.environ.get("HFLOW_FFMPEG")
    if override:
        return override
    return _system_ffmpeg


def _ffmpeg_below_fps_mode_floor() -> tuple[bool, str]:
    """Return (too_old, reason) for the suite ffmpeg, if it can be parsed."""
    ffmpeg = _suite_ffmpeg()
    if ffmpeg is None:
        return False, ""
    completed = subprocess.run(
        [ffmpeg, "-version"],
        check=False,
        capture_output=True,
        text=True,
    )
    raw = completed.stdout or completed.stderr
    version_line = raw.splitlines()[0] if raw else ""
    parsed = _ffmpeg_version_tuple(version_line)
    if parsed is None or parsed >= _MIN_FFMPEG_VERSION:
        return False, ""
    required = ".".join(str(part) for part in _MIN_FFMPEG_VERSION)
    found = ".".join(str(part) for part in parsed)
    reason = (
        f"ffmpeg {found} does not support -fps_mode "
        f"(need {required} or newer; found {version_line!r}). "
        "Upgrade ffmpeg or unset HFLOW_FFMPEG so the pinned build can be used."
    )
    return True, reason


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked as needing system ffmpeg when the binary is older than 5.1."""
    too_old, reason = _ffmpeg_below_fps_mode_floor()
    if not too_old:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("requires_system_ffmpeg"):
            item.add_marker(skip)


@pytest.fixture
def bucket_over_tmp(tmp_path: Path) -> tuple[BucketStorageRoot, Path]:
    """A real bucket root over obstore's local backend and its remote dir."""
    pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")
    from hflow.storage import BucketStorageRoot

    remote_dir = tmp_path / "bucket"
    remote_dir.mkdir(parents=True, exist_ok=True)
    root = BucketStorageRoot(f"file://{remote_dir}", mirror=tmp_path / "bucket-mirror")
    return root, remote_dir
