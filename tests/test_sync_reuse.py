"""Sync keeps the canonical episode it already produced, when it can prove it.

Transcoding is the most expensive thing HFlow does, and re-ingesting an
unchanged recording did all of it again for output it already had. Every test
here is about the PROOF: the gate must fail closed on anything it cannot rule
out, because a wrong reuse silently publishes stale bytes under a current
version stamp.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

import hflow
from hflow.app import _read_sync_completion_marker
from hflow.format import FFMPEG_VERSION_NOT_USED
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import EpisodeStamps, TransformConfig, write_canonical_episode

SYNC_ONLY = {hflow.Stage.SYNC}


def _episode(path: Path, *, duration_s: float = 1.0, seed: int = 0) -> Path:
    return synthesize_episode(
        path, SyntheticEpisodeSpec(duration_s=duration_s, cameras=(), seed=seed)
    )


def _camera_episode(path: Path, *, duration_s: float = 0.5, seed: int = 0) -> Path:
    """One short camera-bearing source: enough for ffmpeg to stamp a real version."""
    return synthesize_episode(
        path,
        SyntheticEpisodeSpec(
            duration_s=duration_s,
            cameras=("wrist_cam",),
            image_hz=10.0,
            seed=seed,
            black_segment=None,
            timestamp_offset_segment=None,
            joint_jump_at_s=None,
        ),
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


def test_an_unchanged_source_with_a_camera_is_not_transcoded_twice(tmp_path: Path) -> None:
    """Control: the ffmpeg reuse comparisons are reachable, and still pass."""
    source = _camera_episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("reuse-camera", data_root=tmp_path / "data", default_checks=())

    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    mtime_after_first = first.canonical_path.stat().st_mtime_ns
    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    assert first.stamps.ffmpeg_version != FFMPEG_VERSION_NOT_USED
    assert first.sync_reused is False
    assert second.sync_reused is True
    assert second.canonical_path.stat().st_mtime_ns == mtime_after_first
    assert first.stamps == second.stamps


def test_a_marker_recording_a_different_ffmpeg_version_is_not_reused(tmp_path: Path) -> None:
    """The witness carries the build that encoded the canonical; a different
    recorded version cannot prove the next transcode would match those bytes."""
    source = _camera_episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("marker-ffmpeg", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    marker_path = first.canonical_path.parent / ".sync-complete.json"
    marker_payload = json.loads(marker_path.read_text())
    marker_payload["ffmpeg_version"] = "ffmpeg version other-than-canonical-stamps"
    marker_path.write_text(json.dumps(marker_payload, sort_keys=True) + "\n")

    # The witness is still complete: refusal must come from the stamps-vs-marker
    # comparison, not from dropping a partial group.
    assert first.stamps.ffmpeg_version != FFMPEG_VERSION_NOT_USED
    witness = _read_sync_completion_marker(marker_path).reuse_witness
    assert witness is not None
    assert witness.ffmpeg_version != first.stamps.ffmpeg_version

    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert second.sync_reused is False


def test_a_canonical_ffmpeg_version_that_no_longer_matches_the_live_build_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transform stamps ffmpeg outside pipeline_version, so a newer build
    must refuse reuse even when the marker still agrees with the canonical."""
    source = _camera_episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("live-ffmpeg", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert first.stamps.ffmpeg_version != FFMPEG_VERSION_NOT_USED

    # Local import inside the reuse gate: patch the source module, not a name
    # bound on hflow.app.
    monkeypatch.setattr("hflow.ffmpeg.ffmpeg_version", lambda: "ffmpeg version not-the-live-build")

    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert second.sync_reused is False


def test_sync_completion_marker_keeps_its_existing_json_bytes(tmp_path: Path) -> None:
    """Typing the witness must not change its persisted wire representation."""
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("marker-bytes", data_root=tmp_path / "data", default_checks=())

    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    marker_path = first.canonical_path.parent / ".sync-complete.json"
    expected_payload = {
        "ffmpeg_version": first.stamps.ffmpeg_version,
        "pipeline_version": first.stamps.pipeline_version,
        "schema_version": first.stamps.schema_version,
        "source_digest": f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}",
        "source_path": str(source.resolve()),
        "transform_kind": "default",
    }

    assert (
        marker_path.read_bytes() == (json.dumps(expected_payload, sort_keys=True) + "\n").encode()
    )


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


@pytest.mark.parametrize(
    "present_fields",
    [
        ("source_digest",),
        ("ffmpeg_version",),
        ("transform_kind",),
        ("source_digest", "ffmpeg_version"),
        ("source_digest", "transform_kind"),
        ("ffmpeg_version", "transform_kind"),
    ],
)
def test_every_partial_reuse_witness_stays_readable_but_is_not_reused(
    tmp_path: Path, present_fields: tuple[str, ...]
) -> None:
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("partial-witness", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)

    marker_path = first.canonical_path.parent / ".sync-complete.json"
    marker_payload = json.loads(marker_path.read_text())
    witness_fields = {"source_digest", "ffmpeg_version", "transform_kind"}
    marker_payload = {
        key: value
        for key, value in marker_payload.items()
        if key not in witness_fields or key in present_fields
    }
    marker_path.write_text(json.dumps(marker_payload, sort_keys=True) + "\n")

    # Pin the mechanism, not only the outcome: the reader must drop the whole
    # group. Read before the runs below, which rewrite the marker with a
    # complete witness. Without this, the parse could go back to accepting a
    # partial witness and the run below would still refuse reuse, on a later
    # gate check rather than on this one.
    assert _read_sync_completion_marker(marker_path).reuse_witness is None

    metadata_only = app.process(source, record=False, stages={hflow.Stage.META}, verbose=False)
    assert metadata_only.canonical_path == first.canonical_path

    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert second.sync_reused is False


@pytest.mark.parametrize(
    "raw_transform_kind", [None, "future-transform"], ids=["missing", "unknown"]
)
def test_a_marker_without_a_known_transform_kind_stays_readable_but_is_not_reused(
    tmp_path: Path, raw_transform_kind: str | None
) -> None:
    source = _episode(tmp_path / "episode_0001.mcap")
    app = hflow.App("unknown-kind", data_root=tmp_path / "data", default_checks=())
    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    marker_path = first.canonical_path.parent / ".sync-complete.json"
    marker_payload = json.loads(marker_path.read_text())
    if raw_transform_kind is None:
        marker_payload.pop("transform_kind")
    else:
        marker_payload["transform_kind"] = raw_transform_kind
    marker_path.write_text(json.dumps(marker_payload, sort_keys=True) + "\n")

    # Non-sync stages still accept old or future markers: the transform kind
    # is a reuse witness, not part of the marker's basic readability contract.
    metadata_only = app.process(source, record=False, stages={hflow.Stage.META}, verbose=False)
    assert metadata_only.canonical_path == first.canonical_path

    second = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    assert second.sync_reused is False


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

    first = app.process(source, record=False, stages=SYNC_ONLY, verbose=False)
    marker_path = first.canonical_path.parent / ".sync-complete.json"
    assert json.loads(marker_path.read_text())["transform_kind"] == "override"
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
