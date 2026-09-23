"""Shared test setup.

Pins the whole suite to the system ffmpeg via the explicit override so tests
never trigger the pinned-build download (a per-machine, network-bound step).
Set at import time, before any test imports resolve the (cached) binary path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from hflow.storage import BucketStorageRoot

_system_ffmpeg = shutil.which("ffmpeg")
if _system_ffmpeg is not None:
    os.environ.setdefault("HFLOW_FFMPEG", _system_ffmpeg)


@pytest.fixture
def bucket_over_tmp(tmp_path: Path) -> tuple[BucketStorageRoot, Path]:
    """A real bucket root over obstore's local backend and its remote dir."""
    pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")
    from hflow.storage import BucketStorageRoot

    remote_dir = tmp_path / "bucket"
    remote_dir.mkdir(parents=True, exist_ok=True)
    root = BucketStorageRoot(f"file://{remote_dir}", mirror=tmp_path / "bucket-mirror")
    return root, remote_dir


@pytest.fixture(scope="session")
def one_second_camera_less_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1 s joints-only source episode, synthesized once for the whole run.

    Shared across modules, so treat it as read-only: a test that needs to
    change a source writes its own copy.
    """
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    return synthesize_episode(
        tmp_path_factory.mktemp("camera-less-source") / "episode_0001.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
    )
