from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from episode_test_helpers import synthesize_canonical_episode

from hflow import Episode
from hflow.testing import SyntheticEpisodeSpec


@pytest.mark.requires_system_ffmpeg
def test_concurrent_frames_extraction_leaves_no_staging_dirs(tmp_path: Path) -> None:
    """#634: Concurrent callers sharing one workdir must both receive complete
    frames and leave zero staging directories behind."""
    canonical = synthesize_canonical_episode(
        tmp_path / "source", SyntheticEpisodeSpec(duration_s=1.0)
    )
    shared_workdir = tmp_path / "shared"
    shared_workdir.mkdir()

    with Episode(canonical, workdir=shared_workdir) as episode:
        camera = episode.cameras[0]
        episode.video(camera)

    def extract() -> list[bytes]:
        with Episode(canonical, workdir=shared_workdir) as episode:
            frames = episode.frames(camera, fps=2.0)
            return [frame.path.read_bytes() for frame in frames]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract) for _ in range(2)]
        results = [future.result() for future in futures]

    assert len(results[0]) > 0
    assert results[0] == results[1]
    assert not list(shared_workdir.glob("*_tmp_*"))


@pytest.mark.requires_system_ffmpeg
def test_concurrent_frames_at_indices_extraction_leaves_no_staging_dirs(
    tmp_path: Path,
) -> None:
    """#634: Concurrent callers sharing one workdir must both receive complete
    indexed frames and leave zero staging directories behind."""
    canonical = synthesize_canonical_episode(
        tmp_path / "source", SyntheticEpisodeSpec(duration_s=1.0)
    )
    shared_workdir = tmp_path / "shared"
    shared_workdir.mkdir()

    with Episode(canonical, workdir=shared_workdir) as episode:
        camera = episode.cameras[0]
        episode.video(camera)

    selected_indices = [0, 1, 2]

    def extract() -> list[bytes]:
        with Episode(canonical, workdir=shared_workdir) as episode:
            frames = episode.frames_at_indices(camera, frame_indices=selected_indices)
            return [frame.path.read_bytes() for frame in frames]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract) for _ in range(2)]
        results = [future.result() for future in futures]

    assert len(results[0]) == len(selected_indices)
    assert results[0] == results[1]
    assert not list(shared_workdir.glob("*_tmp_*"))
