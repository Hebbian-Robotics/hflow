"""Verify a LeRobot prepared-manifest delivery against its episode receipts.

Shared report types live in :mod:`hflow.verification`. This module owns only
the import-delivery half: resolve each claimed episode under the verified
data root as ``landing/<basename>``, compare ``size_bytes`` and
``content_id``, and ignore unlisted files.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from hflow.catalog import content_episode_id
from hflow.storage import StorageRoot, parse_storage_root
from hflow.verification import (
    REASON_CONTENT_ID_MISMATCH,
    REASON_MISSING,
    REASON_SIZE_MISMATCH,
    VerificationFinding,
    VerificationReport,
    VerificationStatus,
)

PREPARED_MANIFEST_RELATIVE_KEY = "prepared-manifest.json"
SUPPORTED_PREPARED_MANIFEST_SCHEMA_VERSION = 3
LANDING_DIRECTORY = "landing"


def verify_lerobot_import(data_root: str | Path | StorageRoot) -> VerificationReport:
    """Check a LeRobot import delivery against its schema-3 prepared manifest.

    Reads ``prepared-manifest.json`` under ``data_root`` and, for each episode
    receipt, compares ``size_bytes`` and ``content_id`` to
    ``landing/<basename>`` under that same root (fetched into the local
    mirror when the root is a bucket). Does not re-convert, does not touch
    the Hugging Face cache, and ignores unlisted files under ``landing/``.

    Raises:
        ValueError: the root or manifest cannot be read as a prepared import
            (CLI exit ``2``).
    """
    try:
        storage = parse_storage_root(data_root)
    except ValueError as error:
        raise ValueError(f"cannot read data root: {error}") from error

    if not storage.exists(PREPARED_MANIFEST_RELATIVE_KEY):
        return VerificationReport(status=VerificationStatus.UNVERIFIABLE)

    try:
        manifest_path = storage.fetch(PREPARED_MANIFEST_RELATIVE_KEY)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{PREPARED_MANIFEST_RELATIVE_KEY} is not valid JSON: {error.msg}"
        ) from error
    except OSError as error:
        raise ValueError(f"cannot read {PREPARED_MANIFEST_RELATIVE_KEY}: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"{PREPARED_MANIFEST_RELATIVE_KEY} must be a JSON object")

    schema_version = payload.get("schema_version")
    if schema_version != SUPPORTED_PREPARED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{PREPARED_MANIFEST_RELATIVE_KEY} schema_version must be "
            f"{SUPPORTED_PREPARED_MANIFEST_SCHEMA_VERSION}, got {schema_version!r}"
        )

    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{PREPARED_MANIFEST_RELATIVE_KEY} is missing an episodes list")
    if not episodes:
        # A readable schema-3 receipt that claims nothing is clean, not
        # unverifiable: there is no missing receipt for a human to hunt down.
        return VerificationReport(status=VerificationStatus.OK)

    findings: list[VerificationFinding] = []
    for index, entry in enumerate(episodes):
        findings.extend(_findings_for_episode_receipt(storage, entry, index=index))

    if findings:
        return VerificationReport(status=VerificationStatus.DAMAGED, findings=findings)
    return VerificationReport(status=VerificationStatus.OK)


def _landing_relative_key_from_receipt_uri(uri: str, *, index: int) -> str:
    """Map a publish-time uri to ``landing/<basename>`` under the verified root.

    Schema 3 stores an absolute publish uri as provenance. The importer always
    places episodes at ``landing/lerobot_episode_XXXX.mcap``, so the basename
    is enough to re-resolve under a copied data root without a schema bump.
    """
    parsed = urlparse(uri)
    path_text = unquote(parsed.path) if parsed.scheme else uri
    basename = PurePosixPath(path_text.replace("\\", "/")).name
    if not basename or basename in {".", ".."} or "/" in basename or "\\" in basename:
        raise ValueError(f"episodes[{index}].uri has no usable landing basename: {uri!r}")
    return f"{LANDING_DIRECTORY}/{basename}"


def _findings_for_episode_receipt(
    storage: StorageRoot, entry: Any, *, index: int
) -> list[VerificationFinding]:
    if not isinstance(entry, dict):
        raise ValueError(f"episodes[{index}] must be a JSON object")

    try:
        uri = entry["uri"]
        expected_content_id = entry["content_id"]
        expected_size_bytes = entry["size_bytes"]
    except KeyError as error:
        raise ValueError(
            f"episodes[{index}] is missing required field {error.args[0]!r}"
        ) from error

    if not isinstance(uri, str) or not uri:
        raise ValueError(f"episodes[{index}].uri must be a non-empty string")
    if not isinstance(expected_content_id, str) or not expected_content_id:
        raise ValueError(f"episodes[{index}].content_id must be a non-empty string")
    if not isinstance(expected_size_bytes, int) or isinstance(expected_size_bytes, bool):
        raise ValueError(f"episodes[{index}].size_bytes must be an integer")

    relative_key = _landing_relative_key_from_receipt_uri(uri, index=index)
    if not storage.exists(relative_key):
        return [
            VerificationFinding(
                uri=uri,
                reason=REASON_MISSING,
                detail=f"landing object missing at {relative_key!r} under the verified root",
            )
        ]

    try:
        local_path = storage.fetch(relative_key)
    except FileNotFoundError:
        return [
            VerificationFinding(
                uri=uri,
                reason=REASON_MISSING,
                detail=f"landing object missing at {relative_key!r} under the verified root",
            )
        ]
    except OSError as error:
        raise ValueError(f"cannot read episode {relative_key!r}: {error}") from error

    findings: list[VerificationFinding] = []
    actual_size = local_path.stat().st_size
    if actual_size != expected_size_bytes:
        findings.append(
            VerificationFinding(
                uri=uri,
                reason=REASON_SIZE_MISMATCH,
                detail=f"size_bytes {actual_size} != receipt {expected_size_bytes}",
            )
        )

    actual_content_id = content_episode_id(local_path)
    if actual_content_id != expected_content_id:
        findings.append(
            VerificationFinding(
                uri=uri,
                reason=REASON_CONTENT_ID_MISMATCH,
                detail=f"content_id {actual_content_id!r} != receipt {expected_content_id!r}",
            )
        )
    return findings


__all__ = ["verify_lerobot_import"]
