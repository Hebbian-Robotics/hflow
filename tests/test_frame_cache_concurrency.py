from __future__ import annotations

import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from hflow.episode import Episode
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode


def _worker_frames(canonical_path: str, workdir: str, fps: float) -> list[str]:
    with Episode(canonical_path, workdir=workdir) as ep:
        frames = ep.frames(fps=fps)
        return [str(frame.path) for frame in frames]


def _worker_frames_at_indices(canonical_path: str, workdir: str, indices: list[int]) -> list[str]:
    with Episode(canonical_path, workdir=workdir) as ep:
        frames = ep.frames_at_indices(frame_indices=indices)
        return [str(frame.path) for frame in frames]


@pytest.fixture(scope="module")
def canonical_episode_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temp_dir = tmp_path_factory.mktemp("test-frame-cache-concurrency")
    source_mcap = temp_dir / "source.mcap"
    canonical_mcap = temp_dir / "canonical.mcap"
    synthesize_episode(
        source_mcap,
        SyntheticEpisodeSpec(
            duration_s=1.0,
            cameras=("cam",),
            image_hz=5.0,
        ),
    )
    write_canonical_episode(source_mcap, canonical_mcap)
    return canonical_mcap


def test_concurrent_identical_requests_frames(canonical_episode_path: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_worker_frames, str(canonical_episode_path), str(workdir), 2.0)
            for _ in range(4)
        ]
        results = [f.result() for f in futures]

    first_result = results[0]
    assert len(first_result) > 0
    for r in results[1:]:
        assert r == first_result

    for frame_str in first_result:
        assert Path(frame_str).is_file()

    # Verify no temp dirs left and lock file is preserved
    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))


def test_concurrent_identical_requests_frames_at_indices(
    canonical_episode_path: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                _worker_frames_at_indices,
                str(canonical_episode_path),
                str(workdir),
                [0, 2],
            )
            for _ in range(4)
        ]
        results = [f.result() for f in futures]

    first_result = results[0]
    assert len(first_result) == 2
    for r in results[1:]:
        assert r == first_result

    for frame_str in first_result:
        assert Path(frame_str).is_file()

    # Verify no temp dirs left and lock file is preserved
    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))


def test_separate_processes_coordination_frames(
    canonical_episode_path: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "workdir_proc_frames"
    workdir.mkdir()

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
        futures = [
            pool.submit(_worker_frames, str(canonical_episode_path), str(workdir), 2.0)
            for _ in range(3)
        ]
        results = [f.result() for f in futures]

    first_result = results[0]
    assert len(first_result) > 0
    for r in results[1:]:
        assert r == first_result

    for frame_str in first_result:
        assert Path(frame_str).is_file()

    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))


def test_separate_processes_coordination_frames_at_indices(
    canonical_episode_path: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "workdir_proc_indices"
    workdir.mkdir()

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _worker_frames_at_indices,
                str(canonical_episode_path),
                str(workdir),
                [0, 2],
            )
            for _ in range(3)
        ]
        results = [f.result() for f in futures]

    first_result = results[0]
    assert len(first_result) == 2
    for r in results[1:]:
        assert r == first_result

    for frame_str in first_result:
        assert Path(frame_str).is_file()

    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))


def test_failed_extraction_leaves_no_artifacts_and_retry_succeeds_frames(
    canonical_episode_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir_fail_frames"
    workdir.mkdir()

    original_run = subprocess.run

    def failing_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Let ffmpeg --version or non-frame calls succeed if any
        cmd = args[0] if args else kwargs.get("args", [])
        if any("frame_%06d.jpg" in str(arg) for arg in cmd):
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="Simulated extraction failure"
            )
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_run)

    with (
        Episode(canonical_episode_path, workdir=workdir) as ep,
        pytest.raises(RuntimeError, match="ffmpeg frame extraction produced no frames"),
    ):
        ep.frames(fps=2.0)

    # Cache should not be published, and no temporary directories should remain
    assert not list(workdir.glob("frames_*[!.tmp][!.lock]"))
    assert not list(workdir.glob("*.tmp"))
    # Lock file is preserved
    assert list(workdir.glob("*.lock"))

    # Restore normal subprocess.run and retry
    monkeypatch.undo()

    with Episode(canonical_episode_path, workdir=workdir) as ep:
        frames = ep.frames(fps=2.0)
        assert len(frames) > 0
        for frame in frames:
            assert frame.path.is_file()

    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))


def test_failed_extraction_leaves_no_artifacts_and_retry_succeeds_frames_at_indices(
    canonical_episode_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = tmp_path / "workdir_fail_indices"
    workdir.mkdir()

    original_run = subprocess.run

    def failing_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = args[0] if args else kwargs.get("args", [])
        if any("frame_%06d.jpg" in str(arg) for arg in cmd):
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="Simulated filter extraction failure"
            )
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_run)

    with (
        Episode(canonical_episode_path, workdir=workdir) as ep,
        pytest.raises(RuntimeError, match="ffmpeg extracted"),
    ):
        ep.frames_at_indices(frame_indices=[0, 2])

    assert not list(workdir.glob("frames_*[!.tmp][!.lock]"))
    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))

    monkeypatch.undo()

    with Episode(canonical_episode_path, workdir=workdir) as ep:
        frames = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(frames) == 2
        for frame in frames:
            assert frame.path.is_file()

    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))


def test_incomplete_indexed_cache_is_rebuilt_safely(
    canonical_episode_path: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "workdir_incomplete"
    workdir.mkdir()

    with Episode(canonical_episode_path, workdir=workdir) as ep:
        frames = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(frames) == 2
        for frame in frames:
            assert frame.path.is_file()

        cache_dirs = [d for d in workdir.iterdir() if d.is_dir() and not d.name.endswith(".tmp")]
        assert len(cache_dirs) == 1
        published_dir = cache_dirs[0]

        # Simulate incomplete cache: delete one of the expected frame files
        second_frame = published_dir / "frame_000001.jpg"
        assert second_frame.is_file()
        second_frame.unlink()

        # Re-requesting must detect incomplete cache, wipe it, and rebuild safely
        rebuilt_frames = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(rebuilt_frames) == 2
        for frame in rebuilt_frames:
            assert frame.path.is_file()

    assert not list(workdir.glob("*.tmp"))
    assert list(workdir.glob("*.lock"))


def test_cache_hits_succeed_in_read_only_workdir(
    canonical_episode_path: Path, tmp_path: Path
) -> None:
    import os
    import stat

    workdir = tmp_path / "workdir_readonly"
    workdir.mkdir()

    # First pass: populate the cache
    with Episode(canonical_episode_path, workdir=workdir) as ep:
        frames = ep.frames(fps=2.0)
        assert len(frames) > 0
        frames_idx = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(frames_idx) == 2

    # Make the workdir and all its contents strictly read-only
    for root, dirs, files in os.walk(workdir):
        for d in dirs:
            Path(root, d).chmod(stat.S_IREAD | stat.S_IEXEC)
        for f in files:
            Path(root, f).chmod(stat.S_IREAD)
    workdir.chmod(stat.S_IREAD | stat.S_IEXEC)

    try:
        # Second pass: cache hits should return without attempting to create/acquire locks
        with Episode(canonical_episode_path, workdir=workdir) as ep:
            frames2 = ep.frames(fps=2.0)
            assert len(frames2) == len(frames)

            frames_idx2 = ep.frames_at_indices(frame_indices=[0, 2])
            assert len(frames_idx2) == 2
    finally:
        # Restore permissions so pytest can clean up
        for root, dirs, files in os.walk(workdir):
            for d in dirs:
                Path(root, d).chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
            for f in files:
                Path(root, f).chmod(stat.S_IREAD | stat.S_IWRITE)
        workdir.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)


def test_stranded_staging_directories_cleaned_up_on_retry(
    canonical_episode_path: Path, tmp_path: Path
) -> None:
    from hflow.episode import _FRAME_STAGING_MARKER

    probe_workdir = tmp_path / "probe_workdir"
    probe_workdir.mkdir()
    workdir = tmp_path / "workdir_stranded"
    workdir.mkdir()

    # Determine target cache directory names by running once in probe directory
    with Episode(canonical_episode_path, workdir=probe_workdir) as ep:
        target_name = ep.frames(fps=2.0)[0].path.parent.name
        target_idx_name = ep.frames_at_indices(frame_indices=[0, 2])[0].path.parent.name

    # Simulate stranded temporary directories left behind by killed workers
    stranded_1 = workdir / f"{target_name}.killed1.tmp"
    stranded_1.mkdir()
    (stranded_1 / _FRAME_STAGING_MARKER).write_text(target_name)
    (stranded_1 / "partial_frame.jpg").write_text("corrupt partial frame")

    stranded_2 = workdir / f"{target_name}.killed2.tmp"
    stranded_2.mkdir()
    (stranded_2 / _FRAME_STAGING_MARKER).write_text(target_name)
    (stranded_2 / "corrupt.bin").write_bytes(b"1234")

    stranded_idx = workdir / f"{target_idx_name}.killed_idx.tmp"
    stranded_idx.mkdir()
    (stranded_idx / _FRAME_STAGING_MARKER).write_text(target_idx_name)
    (stranded_idx / "partial.jpg").write_text("partial")

    assert len(list(workdir.glob("*.tmp"))) == 3

    # On retry/next call, stranded staging directories must be cleaned up under lock
    with Episode(canonical_episode_path, workdir=workdir) as ep:
        frames = ep.frames(fps=2.0)
        assert len(frames) > 0
        assert not list(workdir.glob(f"{target_name}.*.tmp"))

        frames_idx = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(frames_idx) == 2
        assert not list(workdir.glob(f"{target_idx_name}.*.tmp"))

    assert not list(workdir.glob("*.tmp"))


def test_unrelated_backup_files_and_directories_are_preserved_on_cache_miss(
    canonical_episode_path: Path, tmp_path: Path
) -> None:
    probe_workdir = tmp_path / "probe_workdir"
    probe_workdir.mkdir()
    workdir = tmp_path / "workdir_backups"
    workdir.mkdir()

    with Episode(canonical_episode_path, workdir=probe_workdir) as ep:
        target_name = ep.frames(fps=2.0)[0].path.parent.name
        target_idx_name = ep.frames_at_indices(frame_indices=[0, 2])[0].path.parent.name

    # Create caller-supplied backup files and directories matching <cache-key>.*.tmp
    backup_file_1 = workdir / f"{target_name}.backup.tmp"
    backup_file_1.write_text("user backup file content")

    backup_dir_1 = workdir / f"{target_name}.backup_dir.tmp"
    backup_dir_1.mkdir()
    (backup_dir_1 / "saved_data.txt").write_text("precious user backup directory")

    backup_file_idx = workdir / f"{target_idx_name}.snapshot.tmp"
    backup_file_idx.write_text("user snapshot index backup")

    backup_dir_idx = workdir / f"{target_idx_name}.manual_copy.tmp"
    backup_dir_idx.mkdir()
    (backup_dir_idx / "notes.md").write_text("do not delete")

    # Run extraction on a cache miss
    with Episode(canonical_episode_path, workdir=workdir) as ep:
        frames = ep.frames(fps=2.0)
        assert len(frames) > 0

        frames_idx = ep.frames_at_indices(frame_indices=[0, 2])
        assert len(frames_idx) == 2

    # Verify all unrelated backup files and directories are completely preserved
    assert backup_file_1.is_file()
    assert backup_file_1.read_text() == "user backup file content"

    assert backup_dir_1.is_dir()
    assert (backup_dir_1 / "saved_data.txt").read_text() == "precious user backup directory"

    assert backup_file_idx.is_file()
    assert backup_file_idx.read_text() == "user snapshot index backup"

    assert backup_dir_idx.is_dir()
    assert (backup_dir_idx / "notes.md").read_text() == "do not delete"
