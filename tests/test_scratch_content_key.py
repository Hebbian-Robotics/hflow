"""Scratch remuxes must follow canonical content, not the source-keyed run dir.

Issue #536: a sync-omitted META/MEDIA run can etag-fetch a replaced canonical
while reusing a flat ``run_dir/scratch`` remux from the previous bytes. The
catalog then stores old measurements under the new content ``episode_id``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import hflow
from hflow.catalog import content_episode_id
from hflow.checks import camera_frame_stats
from hflow.storage import BucketStorageRoot
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

_CAMERA = "wrist_cam"
# One second at 15 Hz with black in (0.2, 0.5) yields ~33% black (same pin as
# tests/test_default_checks.py). Clean footage with no black segment is ~0%.
_CLEAN_SPEC = SyntheticEpisodeSpec(
    duration_s=1.0,
    cameras=(_CAMERA,),
    image_hz=15.0,
    seed=11,
    black_segment=None,
    timestamp_offset_segment=None,
    joint_jump_at_s=None,
)
_BLACK_SPEC = SyntheticEpisodeSpec(
    duration_s=1.0,
    cameras=(_CAMERA,),
    image_hz=15.0,
    seed=11,
    black_segment=(0.2, 0.5),
    timestamp_offset_segment=None,
    joint_jump_at_s=None,
)


def _black_frame_pct(report: hflow.ProcessReport) -> float:
    result = report.check("camera_frame_stats").result
    assert result is not None
    matching = [
        (key, value)
        for key, value in result.measurements.items()
        if key.endswith("/black_frame_pct") and _CAMERA in key
    ]
    assert len(matching) == 1, matching
    value = matching[0][1]
    assert isinstance(value, float)
    return value


def _black_frame_key(report: hflow.ProcessReport) -> str:
    result = report.check("camera_frame_stats").result
    assert result is not None
    matching = [
        key for key in result.measurements if key.endswith("/black_frame_pct") and _CAMERA in key
    ]
    assert len(matching) == 1
    return matching[0]


def _app(name: str, data_root: Path | BucketStorageRoot) -> hflow.App:
    return hflow.App(name, data_root=data_root, default_checks=(camera_frame_stats,))


@pytest.mark.parametrize("with_media", [False, True], ids=["meta", "meta-media"])
def test_sync_omitted_meta_remeasures_after_canonical_mutation(
    tmp_path: Path, with_media: bool
) -> None:
    """Mutation proof: replace canonical bytes, leave scratch alone, META must change.

    Local data root: ``fetch`` is a path return, so overwriting the canonical
    file models the sync-omitted worker that already has the new bytes on disk
    while its scratch still holds the old remux.
    """
    data_root = tmp_path / "data"
    source = synthesize_episode(tmp_path / "clean.mcap", _CLEAN_SPEC)
    app = _app("scratch-mutation", data_root)
    stages = {hflow.Stage.META}
    if with_media:
        stages.add(hflow.Stage.MEDIA)

    first = asyncio.run(app.process(source, stages=stages | {hflow.Stage.SYNC}, verbose=False))
    assert not first.has_errors
    first_black = _black_frame_pct(first)
    assert first_black == pytest.approx(0.0, abs=0.6)
    first_episode_id = content_episode_id(first.canonical_path)

    scratch_root = first.canonical_path.parent / "scratch"
    remux_paths = list(scratch_root.rglob("*.mp4"))
    assert remux_paths, "expected a remux after camera_frame_stats"
    remux_before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in remux_paths}
    frames_before = {path: path.read_bytes() for path in scratch_root.rglob("*.jpg")}
    if with_media:
        assert frames_before, "MEDIA must populate the extracted-frame cache"

    black_source = synthesize_episode(tmp_path / "black.mcap", _BLACK_SPEC)
    # Replacement canonical keeps the same pipeline stamps (marker still
    # validates) but different camera bytes -- content hash changes.
    replacement = asyncio.run(
        app.process(
            black_source,
            record=False,
            stages={hflow.Stage.SYNC},
            output_dir=tmp_path / "replacement-run",
            verbose=False,
        )
    )
    assert content_episode_id(replacement.canonical_path) != first_episode_id
    first.canonical_path.write_bytes(replacement.canonical_path.read_bytes())
    for path, (mtime_ns, payload) in remux_before.items():
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime_ns

    second = asyncio.run(app.process(source, stages=stages, verbose=False))
    assert not second.has_errors
    second_black = _black_frame_pct(second)
    assert content_episode_id(second.canonical_path) == content_episode_id(
        replacement.canonical_path
    )
    assert second_black == pytest.approx(100.0 * 5 / 15, abs=5.0)
    assert second_black != pytest.approx(first_black, abs=0.6)
    second_scratch_ids = {
        path.parent.name for path in (second.canonical_path.parent / "scratch").rglob("*.mp4")
    }
    assert content_episode_id(second.canonical_path) in second_scratch_ids
    for path, (mtime_ns, payload) in remux_before.items():
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime_ns
    if with_media:
        new_frames = list((scratch_root / content_episode_id(second.canonical_path)).rglob("*.jpg"))
        assert new_frames
        assert any(path.read_bytes() not in frames_before.values() for path in new_frames)
        assert all(path.read_bytes() == payload for path, payload in frames_before.items())
    assert second.catalog_entry is not None
    connection = hflow.open_catalog_connection(app.workspace.catalog_root)
    try:
        stored = connection.execute(
            "SELECT value_double FROM measurements_latest WHERE episode_id = ? AND key = ?",
            [second.catalog_entry.episode_id, _black_frame_key(second)],
        ).fetchone()
    finally:
        connection.close()
    assert stored is not None
    assert stored[0] == pytest.approx(second_black)


def test_file_bucket_sync_omitted_worker_does_not_reuse_peer_stale_scratch(
    tmp_path: Path,
) -> None:
    """Dual-mirror file:// store: worker with stale scratch measures fetched v2."""
    pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    remote_url = f"file://{remote_dir}"

    worker1_root = BucketStorageRoot(remote_url, mirror=tmp_path / "worker1-mirror")
    worker2_root = BucketStorageRoot(remote_url, mirror=tmp_path / "worker2-mirror")

    clean_local = synthesize_episode(tmp_path / "clean.mcap", _CLEAN_SPEC)
    worker1_root.publish(clean_local, "landing/episode.mcap")

    worker1 = _app("worker1", worker1_root)
    worker2 = _app("worker2", worker2_root)

    v1 = asyncio.run(
        worker1.process(
            "landing/episode.mcap",
            stages={hflow.Stage.SYNC, hflow.Stage.META},
            verbose=False,
        )
    )
    v1_black = _black_frame_pct(v1)
    assert v1_black == pytest.approx(0.0, abs=0.6)
    v1_id = content_episode_id(v1.canonical_path)
    assert list((v1.canonical_path.parent / "scratch").rglob("*.mp4"))

    black_local = synthesize_episode(tmp_path / "black.mcap", _BLACK_SPEC)
    worker2_root.publish(black_local, "landing/episode.mcap")
    v2 = asyncio.run(
        worker2.process(
            "landing/episode.mcap",
            stages={hflow.Stage.SYNC, hflow.Stage.META},
            verbose=False,
        )
    )
    v2_black = _black_frame_pct(v2)
    assert v2_black == pytest.approx(100.0 * 5 / 15, abs=5.0)
    v2_id = content_episode_id(v2.canonical_path)
    assert v2_id != v1_id

    # Worker 1 still has the v1 remux under its mirror; sync-omitted META must
    # fetch the v2 canonical and measure v2 pixels, not the cached remux.
    measured = asyncio.run(
        worker1.process(
            "landing/episode.mcap",
            stages={hflow.Stage.META},
            verbose=False,
        )
    )
    measured_black = _black_frame_pct(measured)
    assert content_episode_id(measured.canonical_path) == v2_id
    assert measured_black == pytest.approx(v2_black, abs=0.6)
    assert measured_black != pytest.approx(v1_black, abs=0.6)
    assert measured.catalog_entry is not None

    black_key = _black_frame_key(measured)
    connection = hflow.open_catalog_connection(worker1.workspace.catalog_root)
    try:
        rows = connection.execute(
            "SELECT episode_id, value_double FROM measurements WHERE key = ? ORDER BY recorded_at",
            [black_key],
        ).fetchall()
    finally:
        connection.close()
    assert any(episode_id == v2_id and abs(value - v2_black) < 1.0 for episode_id, value in rows)


def test_unchanged_content_reuses_scratch_and_sync_rewrite_clears_it(tmp_path: Path) -> None:
    """Reuse keeps completed remuxes/frames; a forced rewrite removes old scratch."""
    source = synthesize_episode(tmp_path / "clean.mcap", _CLEAN_SPEC)
    app = _app("scratch-lifecycle", tmp_path / "data")
    stages = {hflow.Stage.SYNC, hflow.Stage.META, hflow.Stage.MEDIA}
    first = asyncio.run(app.process(source, stages=stages))
    assert not first.has_errors
    scratch_root = first.canonical_path.parent / "scratch"
    artifacts = list(scratch_root.rglob("*.mp4")) + list(scratch_root.rglob("*.jpg"))
    assert any(path.suffix == ".mp4" for path in artifacts)
    assert any(path.suffix == ".jpg" for path in artifacts)
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in artifacts}
    for replay_stages in (stages, stages - {hflow.Stage.SYNC}):
        replay = asyncio.run(app.process(source, stages=replay_stages))
        assert not replay.has_errors
        assert replay.sync_reused == (hflow.Stage.SYNC in replay_stages)
        assert _black_frame_pct(replay) == pytest.approx(_black_frame_pct(first))
        for path, (mtime_ns, payload) in before.items():
            assert path.stat().st_mtime_ns == mtime_ns
            assert path.read_bytes() == payload

    # Include both pre-fix flat scratch and an obsolete content directory.
    legacy_file = scratch_root / "legacy.mp4"
    legacy_file.write_bytes(b"obsolete flat cache")
    old_content = scratch_root / "old-content"
    old_content.mkdir()
    (old_content / "stale.mp4").write_bytes(b"obsolete keyed cache")
    (first.canonical_path.parent / ".sync-complete.json").unlink()
    rewritten = asyncio.run(app.process(source, stages=stages))
    assert not rewritten.has_errors
    assert not rewritten.sync_reused
    assert not legacy_file.exists()
    assert not old_content.exists()
    assert _black_frame_pct(rewritten) == pytest.approx(_black_frame_pct(first))
