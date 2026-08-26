"""Sync keeps the canonical episode it already produced, when it can prove it.

Transcoding is the most expensive thing HFlow does, and re-ingesting an
unchanged recording did all of it again for output it already had. Every test
here is about the PROOF: the gate must fail closed on anything it cannot rule
out, because a wrong reuse silently publishes stale bytes under a current
version stamp.
"""

import os
from pathlib import Path

import pytest

import hflow
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import EpisodeStamps, TransformConfig, write_canonical_episode

SYNC_ONLY = {hflow.Stage.SYNC}


def _episode(path: Path, *, duration_s: float = 1.0, seed: int = 0) -> Path:
    return synthesize_episode(
        path, SyntheticEpisodeSpec(duration_s=duration_s, cameras=(), seed=seed)
    )


def test_an_unchanged_source_is_not_transcoded_twice(tmp_path: Path) -> None:
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("reuse", data_root=tmp_path / "data", default_checks=())

    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    # Read BEFORE the second run: reading after would compare the same file
    # against itself and pass whatever happened.
    mtime_after_first = first.canonical_path.stat().st_mtime_ns
    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    assert first.sync_reused is False
    assert second.sync_reused is True
    # The canonical file was not rewritten, which is the whole point.
    assert second.canonical_path.stat().st_mtime_ns == mtime_after_first
    assert first.stamps == second.stamps


def test_the_reused_run_still_records_the_same_episode(tmp_path: Path) -> None:
    """Reuse changes what work happens, never what is recorded: the same
    canonical bytes give the same content-addressed episode_id, so the append
    is the ordinary idempotent no-op."""
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("reuse-record", data_root=tmp_path / "data", default_checks=())

    first = app.process(source, record=True, stages=SYNC_ONLY, verbose=False)
    second = app.process(source, record=True, stages=SYNC_ONLY, verbose=False)

    assert first.catalog_entry is not None
    assert second.catalog_entry is not None
    assert first.catalog_entry.episode_id == second.catalog_entry.episode_id
    assert first.catalog_entry.written is True
    assert second.catalog_entry.written is False


def test_a_source_replaced_in_place_is_transcoded_again(tmp_path: Path) -> None:
    """The witness is content. A source swapped under the same path, with its
    size and mtime preserved, must still be noticed -- which is exactly what
    size-and-mtime bookkeeping would miss."""
    source_path = tmp_path / "episode_0001.mcap"
    _episode(source_path, seed=1)
    app = hflow.App("replaced", data_root=tmp_path / "data", default_checks=())
    first = app.process(source_path, record=False, stages=SYNC_ONLY, verbose=False)
    original_stat = source_path.stat()
    canonical_after_first = first.canonical_path.read_bytes()

    replacement = _episode(tmp_path / "other.mcap", seed=2)
    source_path.write_bytes(replacement.read_bytes())
    os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = app.process(source_path, record=False, stages=SYNC_ONLY, verbose=False)

    assert second.sync_reused is False
    # The canonical really was rebuilt from the new source, not just re-stamped.
    assert second.canonical_path.read_bytes() != canonical_after_first


def test_a_changed_transform_config_is_transcoded_again(tmp_path: Path) -> None:
    source = _episode(tmp_path / "episode_0001.mcap")
    data_root = tmp_path / "data"
    hflow.App("retuned", data_root=data_root, default_checks=()).process(
        source, record=False, stages=SYNC_ONLY, verbose=False
    )

    retuned = hflow.App(
        "retuned",
        data_root=data_root,
        transform=TransformConfig(chunk_size_bytes=400_000),
        default_checks=(),
    )
    second = retuned.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    assert second.sync_reused is False


def test_a_marker_without_a_witness_transcodes_once_and_gains_one(tmp_path: Path) -> None:
    """The whole migration for corpora written before the witness existed."""
    import json

    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("legacy", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    marker_path = first.canonical_path.parent / ".sync-complete.json"
    legacy_payload = {
        key: value
        for key, value in json.loads(marker_path.read_text()).items()
        if key in {"source_path", "schema_version", "pipeline_version"}
    }
    marker_path.write_text(json.dumps(legacy_payload))

    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert second.sync_reused is False

    third = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert third.sync_reused is True


def test_deleting_the_marker_is_the_way_to_force_a_retranscode(tmp_path: Path) -> None:
    """No flag: a new parameter would work in the dev loop and silently not
    reach a generated DAG, whose conf carries no room for one. The marker is
    the switch, at every vantage."""
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("forced", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    (first.canonical_path.parent / ".sync-complete.json").unlink()

    assert app.process(source, record=False, stages=SYNC_ONLY, verbose=False).sync_reused is False


def test_a_missing_canonical_transcodes_rather_than_raising(tmp_path: Path) -> None:
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("missing", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    first.canonical_path.unlink()

    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert second.sync_reused is False
    assert second.canonical_path.is_file()


def test_a_registered_transform_override_never_reuses(tmp_path: Path) -> None:
    """An override has no explicit version, so reuse cannot prove it is unchanged."""
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("override", data_root=tmp_path / "data", default_checks=())

    @app.transform
    def passthrough(source_path: Path, output_path: Path, config: TransformConfig) -> EpisodeStamps:
        return write_canonical_episode(source_path, output_path, config)

    app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    assert second.sync_reused is False


def test_removing_an_override_does_not_reuse_its_canonical(tmp_path: Path) -> None:
    """The subtle one. An override is required to end in
    write_canonical_episode, so it stamps the SAME pipeline_version the default
    transform would -- a digest-and-versions gate alone would happily reuse a
    canonical the current pipeline can no longer produce."""
    source = _episode(tmp_path / "episode_0001.mcap")
    data_root = tmp_path / "data"

    with_override = hflow.App("swap", data_root=data_root, default_checks=())

    @with_override.transform
    def passthrough(source_path: Path, output_path: Path, config: TransformConfig) -> EpisodeStamps:
        return write_canonical_episode(source_path, output_path, config)

    with_override.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    without_override = hflow.App("swap", data_root=data_root, default_checks=())
    second = without_override.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    assert second.sync_reused is False


def test_later_stages_still_read_what_a_reused_sync_left(tmp_path: Path) -> None:
    """A full profile over a reused canonical has to behave identically."""
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("full", data_root=tmp_path / "data")

    first = app.process(source, record=True, stages=None, verbose=False)
    second = app.process(source, record=True, stages=None, verbose=False)

    assert second.sync_reused is True
    assert not second.has_errors
    first_measured = {run.check.name for run in first.checks if run.result is not None}
    second_measured = {run.check.name for run in second.checks if run.result is not None}
    assert first_measured == second_measured


def test_the_summary_says_when_it_reused(tmp_path: Path) -> None:
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("summary", data_root=tmp_path / "data", default_checks=())
    app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    summary = app.process(source, record=False, stages=SYNC_ONLY, verbose=False).summary()

    assert "reused the existing canonical episode" in summary


@pytest.mark.parametrize("stage_set", [SYNC_ONLY, None])
def test_reuse_never_leaves_the_marker_missing(
    tmp_path: Path, stage_set: set[hflow.Stage] | None
) -> None:
    """The marker is cleared before a rewrite so a failed sync cannot leave a
    stale canonical looking valid. Reuse must not trip that: it proved the
    marker good rather than assuming a file on disk was."""
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("marker", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=stage_set, verbose=False)
    marker_path = first.canonical_path.parent / ".sync-complete.json"
    marker_before = marker_path.read_bytes()

    app.process(source, record=False, stages=stage_set, verbose=False)

    assert marker_path.read_bytes() == marker_before
