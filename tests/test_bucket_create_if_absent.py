"""Deterministic races for bucket-backed create-if-absent publishers."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import duckdb
import pytest
from catalog_test_helpers import FAKE_STAMPS, recorded_at_values

import hflow
from hflow.catalog import AppendResult, Catalog, CheckRunRow
from hflow.curation import open_catalog_connection
from hflow.dataset import ManifestAlreadyExistsError, write_dataset_manifest
from hflow.ingest_ledger import IngestFailure, record_ingest_failure
from hflow.storage import BucketStorageRoot
from hflow.workspace import Workspace

pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")


class _CreateIfAbsentRace:
    """Hold two matching puts at the real storage operation and count losses."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        target: Callable[[BucketStorageRoot, str], bool],
    ) -> None:
        self.attempts = 0
        self.refusals = 0
        self._barrier = Barrier(2)
        self._lock = Lock()
        store_file_if_absent = BucketStorageRoot.store_file_if_absent

        def gated_store_file_if_absent(
            root: BucketStorageRoot, local_file: Path, relative: str
        ) -> bool:
            if not target(root, relative):
                return store_file_if_absent(root, local_file, relative)
            with self._lock:
                self.attempts += 1
            self._barrier.wait(timeout=5)
            created = store_file_if_absent(root, local_file, relative)
            if not created:
                with self._lock:
                    self.refusals += 1
            return created

        monkeypatch.setattr(BucketStorageRoot, "store_file_if_absent", gated_store_file_if_absent)


def _append_outcome(catalog: Catalog, canonical_path: Path, value: float = 1.0) -> AppendResult:
    row = CheckRunRow(
        check_name="example_check",
        check_version="v1",
        critical=False,
        status=hflow.CheckStatus.MEASURED,
        duration_s=0.01,
        measurements={"score": value},
        tags=["seen"],
        intervals=[hflow.Interval(start_ns=0, end_ns=10, label="span")],
    )
    return catalog.append_episode(
        canonical_path=canonical_path,
        stamps=FAKE_STAMPS,
        episode_metadata={"task": "bucket race"},
        check_rows=[row],
    )


@pytest.mark.parametrize("recurring", [False, True])
def test_concurrent_bucket_catalog_appends_publish_one_complete_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bucket_over_tmp: tuple[BucketStorageRoot, Path],
    recurring: bool,
) -> None:
    bucket_root, remote_dir = bucket_over_tmp
    catalog = Catalog(bucket_root.child("catalog"))
    canonical = tmp_path / "episode.canonical.mcap"
    canonical.write_bytes(b"canonical episode")
    if recurring:
        assert _append_outcome(catalog, canonical).written
        assert _append_outcome(catalog, canonical, value=2.0).written
    race = _CreateIfAbsentRace(
        monkeypatch,
        lambda _root, relative: relative.startswith("episodes/"),
    )

    def append(label: str) -> AppendResult:
        # Separate workers must discover the same predecessor from the store,
        # without relying on a shared, already-warm mirror.
        worker_root = BucketStorageRoot(
            bucket_root.child("catalog").url, mirror=tmp_path / f"worker-{label}"
        )
        return _append_outcome(Catalog(worker_root), canonical)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, ("A", "B")))

    assert {result.written for result in results} == {True, False}
    assert race.attempts == 2
    assert race.refusals == 1
    assert {result.episode_id for result in results} == {results[0].episode_id}
    assert {result.run_fingerprint for result in results} == {results[0].run_fingerprint}

    stem = f"{results[0].episode_id}-{results[0].run_fingerprint}"
    assert len(recorded_at_values(remote_dir / "catalog", stem)) == 1


def test_concurrent_bucket_manifest_writes_publish_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bucket_over_tmp: tuple[BucketStorageRoot, Path],
) -> None:
    bucket_root, remote_dir = bucket_over_tmp
    workspace = Workspace(bucket_root)
    catalog = Catalog(workspace.catalog_root)
    canonical = tmp_path / "episode.canonical.mcap"
    canonical.write_bytes(b"canonical episode")
    assert _append_outcome(catalog, canonical).written is True
    race = _CreateIfAbsentRace(
        monkeypatch,
        lambda root, relative: root.url.endswith("/manifests") and relative == "pinned.parquet",
    )

    def write(_label: str) -> bool:
        try:
            written = write_dataset_manifest(
                workspace,
                name="clean",
                sql="SELECT episode_id FROM episodes",
                file_stem="pinned",
            )
        except ManifestAlreadyExistsError:
            return False
        assert written.report.row_count == 1
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, ("A", "B")))

    assert sorted(results) == [False, True]
    assert race.attempts == 2
    assert race.refusals == 1
    manifest = remote_dir / "manifests" / "pinned.parquet"
    assert manifest.is_file()
    connection = duckdb.connect()
    try:
        assert connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(manifest)]
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_concurrent_bucket_ingest_failures_publish_one_ledger_row(
    monkeypatch: pytest.MonkeyPatch,
    bucket_over_tmp: tuple[BucketStorageRoot, Path],
) -> None:
    bucket_root, _ = bucket_over_tmp
    catalog_root = bucket_root.child("catalog")
    race = _CreateIfAbsentRace(
        monkeypatch,
        lambda _root, relative: relative.startswith("ingest_failures/"),
    )

    def record(_label: str) -> IngestFailure:
        return record_ingest_failure(
            catalog_root,
            source_uri="episodes-in/corrupt.mcap",
            stage="sync",
            pipeline_version="v1",
            error=ValueError("boom"),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(record, ("A", "B")))

    assert {result.attempt_fingerprint for result in results} == {results[0].attempt_fingerprint}
    assert race.attempts == 2
    assert race.refusals == 1
    assert len(catalog_root.list_names("ingest_failures")) == 1

    with open_catalog_connection(catalog_root) as connection:
        rows = connection.execute(
            "SELECT source_uri, stage, pipeline_version, attempt_fingerprint FROM ingest_failures"
        ).fetchall()
    assert rows == [("episodes-in/corrupt.mcap", "sync", "v1", results[0].attempt_fingerprint)]
