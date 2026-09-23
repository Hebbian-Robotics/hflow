"""The check lane's front door: canonical episodes are integrity-checked once.

#474: every post-sync read ran with chunk CRC validation off, so a canonical
that decayed on disk after sync was re-certified by the very lanes that exist
to judge it -- fresh ``measured`` findings over bytes the file's own stamp
refuses. The guard refuses the episode once, with a named reason, before any
check runs, and memoizes the verdict so one run pays for one strict read.

#506: undecompressable canonical bytes use the same refusal path.
"""

import asyncio
from pathlib import Path

import pytest
from reuse_test_helpers import corrupt_first_chunk_crc, corrupt_zstd_chunk_payload

import hflow
import hflow.app
from hflow.app import CANONICAL_INTEGRITY_STEP_NAME
from hflow.curation import open_catalog_connection
from hflow.reader import (
    CANONICAL_CRC_MISMATCH_REASON,
    CANONICAL_DECOMPRESSION_FAILED_REASON,
    verify_canonical_integrity,
)
from hflow.stage_execution import process_stage_batch
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

SPEC = SyntheticEpisodeSpec(duration_s=2.0, cameras=())


@pytest.fixture
def strict_canonical_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every strict canonical read the app performs, in order.

    The spy wraps the real validator, so the pass still happens.
    """
    strict_reads: list[Path] = []
    real_verify = hflow.app.verify_canonical_integrity

    def counting_verify(path: Path | str) -> tuple[bool, str | None]:
        strict_reads.append(Path(path))
        return real_verify(path)

    monkeypatch.setattr(hflow.app, "verify_canonical_integrity", counting_verify)
    return strict_reads


def _app_with_probe_check(
    data_root: Path,
) -> tuple[hflow.App, list[int], list[int]]:
    """An app whose one check and one enrichment count their invocations."""
    app = hflow.App("integrity", data_root=data_root, default_checks=())
    probe_runs: list[int] = []
    caption_runs: list[int] = []

    @app.check(version="1")
    async def probe(ep: hflow.Episode) -> hflow.CheckResult:
        probe_runs.append(1)
        return hflow.CheckResult(measurements={"probe": 1})

    @app.enrich(version="1")
    def caption(ep: hflow.Episode) -> hflow.EnrichmentResult:
        caption_runs.append(1)
        return hflow.EnrichmentResult(labels={"caption": "a robot arm moves"})

    return app, probe_runs, caption_runs


def test_a_decayed_canonical_is_refused_once_with_a_named_reason(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app, probe_runs, caption_runs = _app_with_probe_check(data_root)
    source = synthesize_episode(tmp_path / "episode.mcap", SPEC)

    healthy = asyncio.run(app.process(source, stages="full"))
    assert healthy.refusal_reason is None
    probe_runs_after_healthy = len(probe_runs)
    caption_runs_after_healthy = len(caption_runs)

    corrupt_first_chunk_crc(healthy.canonical_path)

    refused = asyncio.run(app.process(source, stages="metadata_backfill"))
    assert refused.refusal_reason == CANONICAL_CRC_MISMATCH_REASON
    # ONE diagnosis: the refusal lives on the report field and on one
    # framework-owned catalog row; report.checks is empty because no check
    # ran, so zero check tracebacks is structural.
    assert refused.checks == []
    assert "REFUSED: canonical-crc-mismatch" in refused.summary()
    assert len(probe_runs) == probe_runs_after_healthy
    assert refused.has_errors

    # The refusal covers every consuming lane. The relabel lane declines to
    # spend enrichments on the damaged bytes ...
    relabel_refused = asyncio.run(app.process(source, stages="relabel"))
    assert relabel_refused.refusal_reason == CANONICAL_CRC_MISMATCH_REASON
    assert relabel_refused.enrichments == []
    assert len(caption_runs) == caption_runs_after_healthy

    # ... and the production meta task (the generated DAG's own batch entry,
    # the lane an online re-check flows through) refuses with it.
    recheck_counts = asyncio.run(process_stage_batch(app, [str(source)], "meta"))
    assert recheck_counts == {"processed": 0, "quarantined": 0, "errors": 1}

    # The reason is a queryable value, not just a log string: one refusal row
    # filtered by exact error equality, and an episode the curation views no
    # longer read as ok.
    connection = open_catalog_connection(data_root / "catalog")
    try:
        refusal_rows = connection.execute(
            "SELECT check_name, status, error FROM check_runs WHERE error = ?",
            [CANONICAL_CRC_MISMATCH_REASON],
        ).fetchall()
        episode_status = connection.execute("SELECT status FROM episodes").fetchall()
    finally:
        connection.close()
    assert refusal_rows == [(CANONICAL_INTEGRITY_STEP_NAME, "error", CANONICAL_CRC_MISMATCH_REASON)]
    assert episode_status == [("unverified",)]


def test_a_healthy_canonical_runs_clean(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app, probe_runs, _caption_runs = _app_with_probe_check(data_root)
    source = synthesize_episode(tmp_path / "episode.mcap", SPEC)

    report = asyncio.run(app.process(source, stages="full"))
    assert report.refusal_reason is None
    assert [run.status for run in report.checks] == [hflow.CheckStatus.MEASURED]
    assert not report.has_errors
    assert len(probe_runs) == 1
    assert verify_canonical_integrity(report.canonical_path) == (True, None)

    connection = open_catalog_connection(data_root / "catalog")
    try:
        refusal_row_count = connection.execute(
            "SELECT count(*) FROM check_runs WHERE status = 'error'"
        ).fetchone()
    finally:
        connection.close()
    assert refusal_row_count is not None
    assert int(refusal_row_count[0]) == 0


def test_one_corrupt_episode_is_strict_read_once_per_run(
    tmp_path: Path, strict_canonical_reads: list[Path]
) -> None:
    """Two check-lane runs over the same damaged bytes pay for one strict read.

    The spy wraps the real validator -- the pass still happens -- because the
    thing being pinned is the cost the memo exists to avoid: a second full
    CRC read of bytes already judged in this run. Sync performs no strict
    read (see the no-step-work test), so the count starts at the check lane.
    """
    data_root = tmp_path / "data"
    app, _probe_runs, _caption_runs = _app_with_probe_check(data_root)
    source = synthesize_episode(tmp_path / "episode.mcap", SPEC)

    synced = asyncio.run(app.process(source, stages={hflow.Stage.SYNC}, record=False))
    corrupt_first_chunk_crc(synced.canonical_path)

    first = asyncio.run(app.process(source, stages="metadata_backfill"))
    second = asyncio.run(app.process(source, stages="metadata_backfill"))

    assert len(strict_canonical_reads) == 1
    assert first.refusal_reason == second.refusal_reason == CANONICAL_CRC_MISMATCH_REASON

    # Both runs observed the same outcome, so the catalog keeps ONE refusal
    # record: the exact replay deduped through the run fingerprint.
    connection = open_catalog_connection(data_root / "catalog")
    try:
        refusal_row_count = connection.execute(
            "SELECT count(*) FROM check_runs WHERE error = ?", [CANONICAL_CRC_MISMATCH_REASON]
        ).fetchone()
    finally:
        connection.close()
    assert refusal_row_count is not None
    assert int(refusal_row_count[0]) == 1


def test_a_run_with_no_step_work_pays_no_strict_read(
    tmp_path: Path, strict_canonical_reads: list[Path]
) -> None:
    """The guard protects step work over the canonical bytes; a stage
    selection with none to do pays no strict read."""
    data_root = tmp_path / "data"
    app = hflow.App("no-steps", data_root=data_root, default_checks=())
    source = synthesize_episode(tmp_path / "episode.mcap", SPEC)

    # Sync owns the canonical and just wrote it; a relabel with no
    # enrichments registered and a meta with no checks registered have no
    # step work to protect.
    asyncio.run(app.process(source, stages={hflow.Stage.SYNC}, record=False))
    asyncio.run(app.process(source, stages="relabel", record=False))
    asyncio.run(app.process(source, stages="metadata_backfill", record=False))

    assert strict_canonical_reads == []


def test_undecompressable_canonical_is_refused_before_user_steps(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app, probe_runs, caption_runs = _app_with_probe_check(data_root)
    source_uri = "episodes-in/episode.mcap"
    source = synthesize_episode(data_root / source_uri, SPEC)
    synced = asyncio.run(app.process(source, stages={hflow.Stage.SYNC}, record=False))
    assert verify_canonical_integrity(synced.canonical_path) == (True, None)
    corrupt_zstd_chunk_payload(synced.canonical_path)

    reason = CANONICAL_DECOMPRESSION_FAILED_REASON
    assert reason == "canonical-decompression-failed"
    assert verify_canonical_integrity(synced.canonical_path) == (False, reason)
    refused = asyncio.run(app.process(source, stages="metadata_backfill"))
    assert refused.refusal_reason == reason
    assert refused.has_errors
    assert refused.checks == []
    assert f"REFUSED: {reason}" in refused.summary()

    relabel_refused = asyncio.run(app.process(source, stages="relabel"))
    assert relabel_refused.refusal_reason == reason
    assert relabel_refused.enrichments == []
    assert asyncio.run(process_stage_batch(app, [source_uri], "meta")) == {
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
        ("crc", corrupt_first_chunk_crc),
        ("zstd", corrupt_zstd_chunk_payload),
    ):
        source = synthesize_episode(data_root / "episodes-in" / f"{name}.mcap", SPEC)
        synced = asyncio.run(app.process(source, stages={hflow.Stage.SYNC}, record=False))
        corrupt(synced.canonical_path)
        asyncio.run(app.process(source, stages="metadata_backfill"))

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
    synced = asyncio.run(app.process(source, stages={hflow.Stage.SYNC}, record=False))
    synced.canonical_path.unlink()

    with pytest.raises(FileNotFoundError):
        verify_canonical_integrity(synced.canonical_path)
    with pytest.raises(FileNotFoundError, match="no canonical episode exists"):
        asyncio.run(app.process(source, stages="metadata_backfill"))
    assert asyncio.run(process_stage_batch(app, [source_uri], "meta")) == {
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
