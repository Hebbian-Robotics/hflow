"""#506: undecompressable canonical bytes use #502's integrity refusal path."""

from pathlib import Path

import pytest
from reuse_test_helpers import corrupt_zstd_chunk_payload
from test_canonical_integrity import SPEC, _app_with_probe_check, _corrupt_first_chunk_crc

import hflow
from hflow.app import CANONICAL_INTEGRITY_STEP_NAME
from hflow.curation import open_catalog_connection
from hflow.reader import (
    CANONICAL_CRC_MISMATCH_REASON,
    CANONICAL_DECOMPRESSION_FAILED_REASON,
    verify_canonical_integrity,
)
from hflow.stage_execution import process_stage_batch
from hflow.testing import synthesize_episode


def test_undecompressable_canonical_is_refused_before_user_steps(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app, probe_runs, caption_runs = _app_with_probe_check(data_root)
    source_uri = "episodes-in/episode.mcap"
    source = synthesize_episode(data_root / source_uri, SPEC)
    synced = app.process(source, stages={hflow.Stage.SYNC}, record=False)
    assert verify_canonical_integrity(synced.canonical_path) == (True, None)
    corrupt_zstd_chunk_payload(synced.canonical_path)

    reason = CANONICAL_DECOMPRESSION_FAILED_REASON
    assert reason == "canonical-decompression-failed"
    assert verify_canonical_integrity(synced.canonical_path) == (False, reason)
    refused = app.process(source, stages="metadata_backfill")
    assert refused.refusal_reason == reason
    assert refused.has_errors
    assert refused.checks == []
    assert f"REFUSED: {reason}" in refused.summary()

    relabel_refused = app.process(source, stages="relabel")
    assert relabel_refused.refusal_reason == reason
    assert relabel_refused.enrichments == []
    assert process_stage_batch(app, [source_uri], "meta") == {
        "processed": 0,
        "quarantined": 0,
        "errors": 1,
    }
    assert probe_runs == []
    assert caption_runs == []

    connection = open_catalog_connection(data_root / "catalog")
    try:
        rows = connection.execute(
            "SELECT check_name, status, critical, error FROM check_runs"
        ).fetchall()
        episode_status = connection.execute("SELECT status FROM episodes").fetchall()
        failures = connection.execute("SELECT failure_kind FROM ingest_failures").fetchall()
    finally:
        connection.close()
    assert rows == [(CANONICAL_INTEGRITY_STEP_NAME, "error", True, reason)]
    assert episode_status == [("unverified",)]
    assert failures == []


def test_one_error_filter_finds_both_canonical_corruption_species(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app, probe_runs, _caption_runs = _app_with_probe_check(data_root)
    for name, corrupt in (
        ("crc", _corrupt_first_chunk_crc),
        ("zstd", corrupt_zstd_chunk_payload),
    ):
        source = synthesize_episode(data_root / "episodes-in" / f"{name}.mcap", SPEC)
        synced = app.process(source, stages={hflow.Stage.SYNC}, record=False)
        corrupt(synced.canonical_path)
        app.process(source, stages="metadata_backfill")

    connection = open_catalog_connection(data_root / "catalog")
    try:
        rows = connection.execute(
            "SELECT check_name, status, error FROM check_runs WHERE error IN (?, ?) ORDER BY error",
            [CANONICAL_CRC_MISMATCH_REASON, CANONICAL_DECOMPRESSION_FAILED_REASON],
        ).fetchall()
    finally:
        connection.close()
    assert rows == [
        (CANONICAL_INTEGRITY_STEP_NAME, "error", CANONICAL_CRC_MISMATCH_REASON),
        (CANONICAL_INTEGRITY_STEP_NAME, "error", CANONICAL_DECOMPRESSION_FAILED_REASON),
    ]
    assert probe_runs == []


def test_missing_canonical_remains_an_infrastructure_failure(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app, probe_runs, _caption_runs = _app_with_probe_check(data_root)
    source_uri = "episodes-in/episode.mcap"
    source = synthesize_episode(data_root / source_uri, SPEC)
    synced = app.process(source, stages={hflow.Stage.SYNC}, record=False)
    synced.canonical_path.unlink()

    with pytest.raises(FileNotFoundError):
        verify_canonical_integrity(synced.canonical_path)
    with pytest.raises(FileNotFoundError, match="no canonical episode exists"):
        app.process(source, stages="metadata_backfill")
    assert process_stage_batch(app, [source_uri], "meta") == {
        "processed": 0,
        "quarantined": 0,
        "errors": 1,
    }
    assert probe_runs == []

    connection = open_catalog_connection(data_root / "catalog")
    try:
        failures = connection.execute(
            "SELECT source_uri, stage, failure_kind, error_type FROM ingest_failures"
        ).fetchall()
        refusal_rows = connection.execute("SELECT error FROM check_runs").fetchall()
    finally:
        connection.close()
    assert failures == [(source_uri, "meta", "infrastructure", "FileNotFoundError")]
    assert refusal_rows == []
