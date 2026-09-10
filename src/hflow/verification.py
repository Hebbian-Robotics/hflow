"""Shared verification types for every ``hflow verify ...`` command.

``VerificationReport`` is the common surface for delivery verifiers: a
verifier reads the receipt a delivery carries, compares it to the bytes
under the root being verified, and returns one report. Files nobody
listed are ignored. The LeRobot import verifier
(:func:`hflow.importers.lerobot_verify.verify_lerobot_import`, #454) and the
dataset snapshot verifier (:func:`hflow.snapshot.verify_dataset_snapshot`,
#428) both return this shape, so one CLI and one exit-code mapping cover
every delivered artifact.

Reasons use a closed string enum so callers can branch on a documented set of
values.
``exit_code_for`` maps a report to the verify-family exit codes: 0 clean,
1 damaged, 3 unverifiable. Exit 2 (unreadable input) is raised as an
exception before a report exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class VerificationReason(StrEnum):
    """Machine-readable reason for a verification finding."""

    MISSING = "missing"
    SIZE_MISMATCH = "size-mismatch"
    CONTENT_ID_MISMATCH = "content-id-mismatch"
    # #428: a readable snapshot format.json that carries no integrity receipt.
    NO_RECEIPT = "no-receipt"


# Keep the original names as public aliases for callers that already import
# the reason constants.
REASON_MISSING = VerificationReason.MISSING
REASON_SIZE_MISMATCH = VerificationReason.SIZE_MISMATCH
REASON_CONTENT_ID_MISMATCH = VerificationReason.CONTENT_ID_MISMATCH
REASON_NO_RECEIPT = VerificationReason.NO_RECEIPT


class VerificationStatus(StrEnum):
    """Outcome of a verify that successfully read its receipt format."""

    OK = "ok"
    DAMAGED = "damaged"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class VerificationFinding:
    """One claimed object that does not match its receipt."""

    uri: str
    reason: VerificationReason
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    """Result of verifying a delivered artifact against its receipts.

    ``.ok`` is True when every claimed object still matches, including the
    empty-claim case (a readable receipt that lists nothing). ``status``
    distinguishes clean, damaged, and unverifiable (no receipt) so CLI exit
    codes stay distinct.
    """

    status: VerificationStatus
    findings: list[VerificationFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is VerificationStatus.OK


def exit_code_for(report: VerificationReport) -> int:
    """Map a readable verification outcome to the verify-family exit code.

    ``0`` clean, ``1`` damaged, ``3`` unverifiable. Exit ``2`` (unreadable
    input) is raised as an exception before a report exists.
    """
    if report.status is VerificationStatus.OK:
        return 0
    if report.status is VerificationStatus.DAMAGED:
        return 1
    if report.status is VerificationStatus.UNVERIFIABLE:
        return 3
    raise AssertionError(f"unhandled verification status {report.status!r}")
