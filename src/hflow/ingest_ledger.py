"""``ingest_failures``: the sources that produced no catalog row, and why.

The catalog can only describe episodes that exist. Its primary identity is a
content hash of the canonical file (``catalog.content_episode_id``), so a
recording that never canonicalized -- a truncated upload, a file that is not
MCAP, a key that is not there -- has nothing to hash and no row to be. It
simply vanished from the record, and the only trace was a traceback in
whichever log happened to be watching.

That was survivable while every run went through Airflow, whose task logs are
the observability story. It is not survivable now that ``hflow ingest`` runs
in this process when no runtime is addressed, with nothing behind it.

Named for failures rather than for attempts on purpose. A table of "attempts"
that only ever receives the failures invites ``SELECT count(*)`` and the
answer reads as the size of the corpus. Successful attempts already have an
episodes row; this is the complement.

Deliberately NOT part of ``catalog.TABLE_COLUMN_DDL``. Everything in that dict
is keyed by ``(episode_id, run_fingerprint)`` and is written, published and
reconciled on every ordinary episode append. This table has neither column,
and that absence is precisely its reason to exist.
"""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import duckdb

from hflow.catalog import Catalog
from hflow.storage import StorageRoot

INGEST_FAILURES_TABLE_NAME = "ingest_failures"

INGEST_FAILURES_COLUMN_DDL = (
    "source_uri VARCHAR, stage VARCHAR, failure_kind VARCHAR, error_type VARCHAR, "
    "message VARCHAR, pipeline_version VARCHAR, orchestrator_run_id VARCHAR, "
    "attempt_fingerprint VARCHAR, recorded_at TIMESTAMPTZ"
)


class IngestFailureKind(StrEnum):
    """Whose problem this was, as far as the engine can honestly tell.

    Three members, because three is what the evidence supports. The point of
    the distinction is triage: a source that is bad stays bad and wants a
    human, while infrastructure trouble wants a retry -- and reporting one as
    the other sends people to the wrong place.

    ``INFRASTRUCTURE`` is the default for anything unrecognized, never the
    other way round. Guessing "your data is bad" from a crash the engine did
    not recognize is an accusation, and a wrong one is expensive; guessing
    "we broke" is merely humble.
    """

    SOURCE_MISSING = "source-missing"
    SOURCE_UNREADABLE = "source-unreadable"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class IngestFailure:
    """One attempt that produced no catalog row."""

    source_uri: str
    stage: str
    failure_kind: IngestFailureKind
    error_type: str
    message: str
    pipeline_version: str
    orchestrator_run_id: str | None
    attempt_fingerprint: str


def classify_ingest_failure(error: BaseException) -> IngestFailureKind:
    """Which kind of failure this exception represents.

    A heuristic, and stored as one: ``error_type`` and ``message`` are kept
    verbatim beside the classification, so a wrong guess is always visible
    next to the evidence that would correct it rather than replacing it.
    """
    from hflow.app import SourceNotFound

    if isinstance(error, SourceNotFound):
        return IngestFailureKind.SOURCE_MISSING
    try:
        from mcap.exceptions import McapError
    except ImportError:  # pragma: no cover - mcap is a hard dependency
        return IngestFailureKind.INFRASTRUCTURE
    if isinstance(error, McapError):
        # The single base of InvalidMagic, EndOfFile, RecordLengthLimitExceeded
        # and friends: the file is not a readable MCAP, which is a fact about
        # the recording rather than about this machine.
        return IngestFailureKind.SOURCE_UNREADABLE
    return IngestFailureKind.INFRASTRUCTURE


def _attempt_fingerprint(
    *, source_uri: str, stage: str, failure_kind: str, error_type: str, pipeline_version: str
) -> str:
    """Identity of one failure, so replaying it records nothing new.

    The message, any traceback, and the timestamp are deliberately excluded:
    a path or a byte offset inside an error string would make every retry of
    one unchanging problem a new row, and the ledger would grow without
    telling anyone anything it had not already said.
    """
    payload = json.dumps(
        {
            "source_uri": source_uri,
            "stage": stage,
            "failure_kind": failure_kind,
            "error_type": error_type,
            "pipeline_version": pipeline_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def record_ingest_failure(
    catalog_root: Path | str | StorageRoot,
    *,
    source_uri: str,
    stage: str,
    pipeline_version: str,
    error: BaseException,
    orchestrator_run_id: str | None = None,
) -> IngestFailure:
    """Append one failed attempt to the ledger, idempotently.

    Constructs a :class:`~hflow.catalog.Catalog` first, because that is what
    writes the catalog's format marker: a workspace where every episode failed
    to canonicalize would otherwise hold a ledger no reader would open.
    """
    catalog = Catalog(catalog_root)
    failure_kind = classify_ingest_failure(error)
    error_type = type(error).__name__
    failure = IngestFailure(
        source_uri=source_uri,
        stage=stage,
        failure_kind=failure_kind,
        error_type=error_type,
        message=str(error),
        pipeline_version=pipeline_version,
        orchestrator_run_id=orchestrator_run_id or None,
        attempt_fingerprint=_attempt_fingerprint(
            source_uri=source_uri,
            stage=stage,
            failure_kind=failure_kind.value,
            error_type=error_type,
            pipeline_version=pipeline_version,
        ),
    )
    source_digest = hashlib.sha256(source_uri.encode()).hexdigest()[:16]
    relative_key = (
        f"{INGEST_FAILURES_TABLE_NAME}/{source_digest}-{failure.attempt_fingerprint}.parquet"
    )
    connection = duckdb.connect()
    try:
        connection.execute(f"CREATE TABLE failures ({INGEST_FAILURES_COLUMN_DDL})")
        connection.execute(
            "INSERT INTO failures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                failure.source_uri,
                failure.stage,
                failure.failure_kind.value,
                failure.error_type,
                failure.message,
                failure.pipeline_version,
                failure.orchestrator_run_id,
                failure.attempt_fingerprint,
                datetime.now(UTC),
            ],
        )
        with tempfile.TemporaryDirectory(prefix="hflow-ingest-failure-") as staging_directory:
            staged_file = Path(staging_directory) / "failure.parquet"
            connection.execute(
                f"COPY failures TO '{staged_file}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            # Create-if-absent: the same attempt recorded twice is one row,
            # exactly as replaying an episode append is one row.
            catalog.location.store_file_if_absent(staged_file, relative_key)
    finally:
        connection.close()
    return failure
