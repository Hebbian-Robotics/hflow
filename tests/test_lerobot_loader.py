"""Opt-in compatibility with the official LeRobot v3 loader.

Run pytest in a LeRobot environment, with HFLOW_EXPORT_PYTHON pointing to the
project environment's Python. This keeps the two packages' incompatible Hub
SDK requirements separate and leaves the project lockfile unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("HFLOW_LEROBOT_TESTS") != "1",
    reason="set HFLOW_LEROBOT_TESTS=1 to run the official LeRobot loader",
)
@pytest.mark.requires_system_ffmpeg
@pytest.mark.parametrize("task_ids", [(0, 1), (0, 10)])
def test_export_official_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_ids: tuple[int, int]
) -> None:
    """Metadata, camera tensors and a whole episode load locally without Hub downloads."""
    loader = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    torch = pytest.importorskip("torch")
    assert loader.CODEBASE_VERSION == "v3.0"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def refuse_download(*args: object, **kwargs: object) -> None:
        pytest.fail("the local export must load without a Hub download")

    monkeypatch.setattr(loader.LeRobotDatasetMetadata, "pull_from_repo", refuse_download)
    monkeypatch.setattr(loader.LeRobotDataset, "download", refuse_download)
    subprocess.run(
        [
            os.environ.get("HFLOW_EXPORT_PYTHON", sys.executable),
            "-c",
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from test_lerobot_export import _export_real_video_fixture; "
            "_export_real_video_fixture(Path(sys.argv[2]), (0, int(sys.argv[3])))",
            str(Path(__file__).parent),
            str(tmp_path),
            str(task_ids[1]),
        ],
        check=True,
    )
    destination = tmp_path / "exported"
    source_root = tmp_path / "source"
    CAMS = ("observation.images.up", "observation.images.side")
    meta = loader.LeRobotDatasetMetadata("local/export", root=destination)
    assert meta.total_episodes == 2
    assert meta.total_frames == 7
    assert meta.total_tasks == 2
    ds = loader.LeRobotDataset("local/export", root=destination, video_backend="pyav")
    assert len(ds) == 7
    frame = ds[0]
    assert frame["task"] == "task-1"
    assert frame["episode_index"].item() == 0
    for camera in CAMS:
        assert isinstance(frame[camera], torch.Tensor)
        assert frame[camera].shape == (3, 48, 64)
        assert torch.isfinite(frame[camera]).all()
    episode = loader.LeRobotDataset(
        "local/export", root=destination, episodes=[1], video_backend="pyav"
    )
    assert len(episode) == 3
    for index in range(len(episode)):
        frame = episode[index]
        assert frame["task"] == "task-0"
        assert frame["episode_index"].item() == 1
        assert frame["frame_index"].item() == index
        assert frame["observation.state"].tolist() == [float(index)] * 6
        for camera in CAMS:
            assert frame[camera].shape == (3, 48, 64)
            assert torch.isfinite(frame[camera]).all()
    for camera in CAMS:
        assert (destination / f"videos/{camera}/chunk-000/file-000.mp4").read_bytes() == (
            source_root / f"videos/{camera}/chunk-000/file-000.mp4"
        ).read_bytes()
