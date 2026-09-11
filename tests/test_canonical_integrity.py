"""The check lane's front door: canonical episodes are integrity-checked once.

#474: every post-sync read ran with chunk CRC validation off, so a canonical
that decayed on disk after sync was re-certified by the very lanes that exist
to judge it -- fresh ``measured`` findings over bytes the file's own stamp
refuses. The guard refuses the episode once, with a named reason, before any
check runs, and memoizes the verdict so one run pays for one strict read.
"""

import io
from pathlib import Path

import pytest
from mcap.reader import make_reader

import hflow
import hflow.app
from hflow.app import CANONICAL_INTEGRITY_STEP_NAME
from hflow.curation import open_catalog_connection
from hflow.reader import CANONICAL_CRC_MISMATCH_REASON, verify_canonical_integrity
from hflow.stage_execution import process_stage_batch
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

SPEC = SyntheticEpisodeSpec(duration_s=2.0, cameras=())


def _app_with_probe_check(
    data_root: Path,
) -> tuple[hflow.App, list[int], list[int]]:
    """An app whose one check and one enrichment count their invocations."""
    app = hflow.App("integrity", data_root=data_root, default_checks=())
    probe_runs: list[int] = []
    caption_runs: list[int] = []

    @app.check(version="1")
    def probe(ep: hflow.Episode) -> hflow.CheckResult:
        probe_runs.append(1)
        return hflow.CheckResult(measurements={"probe": 1})

    @app.enrich(version="1")
    def caption(ep: hflow.Episode) -> hflow.EnrichmentResult:
        caption_runs.append(1)
        return hflow.EnrichmentResult(labels={"caption": "a robot arm moves"})

    return app, probe_runs, caption_runs


def _corrupt_first_chunk_crc(canonical_path: Path) -> None:
    """Flip one bit of the first chunk's stored CRC: header rot, not payload
    damage -- the payload still decompresses, so only the CRC knows."""
    data = bytearray(canonical_path.read_bytes())
    summary = make_reader(io.BytesIO(bytes(data))).get_summary()
    assert summary is not None and summary.chunk_indexes
    crc_offset = summary.chunk_indexes[0].chunk_start_offset + 33
    data[crc_offset] ^= 0x01
    canonical_path.write_bytes(bytes(data))


def test_a_decayed_canonical_is_refused_once_with_a_named_reason(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    app, probe_runs, caption_runs = _app_with_probe_check(data_root)
    source = synthesize_episode(tmp_path / "episode.mcap", SPEC)

    healthy = app.process(source, stages="full")
    assert healthy.refusal_reason is None
    probe_runs_after_healthy = len(probe_runs)
    caption_runs_after_healthy = len(caption_runs)

    _corrupt_first_chunk_crc(healthy.canonical_path)

    refused = app.process(source, stages="metadata_backfill")
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
    relabel_refused = app.process(source, stages="relabel")
    assert relabel_refused.refusal_reason == CANONICAL_CRC_MISMATCH_REASON
    assert relabel_refused.enrichments == []
    assert len(caption_runs) == caption_runs_after_healthy

    # ... and the production meta task (the generated DAG's own batch entry,
    # the lane an online re-check flows through) refuses with it.
    recheck_counts = process_stage_batch(app, [str(source)], "meta")
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

    report = app.process(source, stages="full")
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two check-lane runs over the same damaged bytes pay for one strict read.

    The spy wraps the real validator -- the pass still happens -- because the
    thing being pinned is the cost the memo exists to avoid: a second full
    CRC read of bytes already judged in this run.
    """
    data_root = tmp_path / "data"
    app, _probe_runs, _caption_runs = _app_with_probe_check(data_root)
    source = synthesize_episode(tmp_path / "episode.mcap", SPEC)

    synced = app.process(source, stages={hflow.Stage.SYNC}, record=False)
    _corrupt_first_chunk_crc(synced.canonical_path)

    strict_reads: list[Path] = []
    real_verify = hflow.app.verify_canonical_integrity

    def counting_verify(path: Path | str) -> tuple[bool, str | None]:
        strict_reads.append(Path(path))
        return real_verify(path)

    monkeypatch.setattr(hflow.app, "verify_canonical_integrity", counting_verify)

    first = app.process(source, stages="metadata_backfill")
    second = app.process(source, stages="metadata_backfill")

    assert len(strict_reads) == 1
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard protects step work over the canonical bytes; a stage
    selection with none to do pays no strict read."""
    data_root = tmp_path / "data"
    app = hflow.App("no-steps", data_root=data_root, default_checks=())
    source = synthesize_episode(tmp_path / "episode.mcap", SPEC)

    strict_reads: list[Path] = []
    real_verify = hflow.app.verify_canonical_integrity

    def counting_verify(path: Path | str) -> tuple[bool, str | None]:
        strict_reads.append(Path(path))
        return real_verify(path)

    monkeypatch.setattr(hflow.app, "verify_canonical_integrity", counting_verify)

    # Sync owns the canonical and just wrote it; a relabel with no
    # enrichments registered and a meta with no checks registered have no
    # step work to protect.
    app.process(source, stages={hflow.Stage.SYNC}, record=False)
    app.process(source, stages="relabel", record=False)
    app.process(source, stages="metadata_backfill", record=False)

    assert strict_reads == []
