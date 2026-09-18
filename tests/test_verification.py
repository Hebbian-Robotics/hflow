"""Outcome-focused tests for prepared-manifest delivery verification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hflow.catalog import content_episode_id
from hflow.cli import main
from hflow.importers.lerobot_verify import verify_lerobot_import
from hflow.storage import BucketStorageRoot
from hflow.verification import (
    REASON_CONTENT_ID_MISMATCH,
    REASON_MISSING,
    REASON_NO_RECEIPT,
    REASON_SIZE_MISMATCH,
    VerificationFinding,
    VerificationReason,
    VerificationStatus,
    exit_code_for,
)


def _write_prepared_delivery(root: Path, *, payload: bytes = b"episode-0") -> Path:
    landing = root / "landing"
    landing.mkdir(parents=True)
    episode_path = landing / "lerobot_episode_0001.mcap"
    episode_path.write_bytes(payload)
    manifest = {
        "schema_version": 3,
        "dataset": {"repo_id": "fake/repo", "revision": "abc", "license": "apache-2.0"},
        "camera_keys": ["observation.image"],
        "episodes_converted": 1,
        "episodes": [
            {
                "uri": str(episode_path.resolve()),
                "content_id": content_episode_id(episode_path),
                "size_bytes": episode_path.stat().st_size,
            }
        ],
        "converter_version": "test",
    }
    (root / "prepared-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return episode_path


def test_verification_reason_constants_are_enum_members() -> None:
    assert REASON_MISSING is VerificationReason.MISSING
    assert REASON_SIZE_MISMATCH is VerificationReason.SIZE_MISMATCH
    assert REASON_CONTENT_ID_MISMATCH is VerificationReason.CONTENT_ID_MISMATCH
    assert REASON_NO_RECEIPT is VerificationReason.NO_RECEIPT
    assert VerificationReason.NO_RECEIPT == "no-receipt"

    finding = VerificationFinding(
        uri="landing/episode.mcap",
        reason=VerificationReason.NO_RECEIPT,
        detail="receipt is missing",
    )
    assert finding.reason is VerificationReason.NO_RECEIPT


def test_verify_lerobot_import_accepts_an_unchanged_delivery(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    _write_prepared_delivery(root)
    (root / "landing" / "extra-unlisted.bin").write_bytes(b"noise")

    report = verify_lerobot_import(root)

    assert report.ok
    assert report.status is VerificationStatus.OK
    assert report.findings == []
    assert exit_code_for(report) == 0


def test_verify_lerobot_import_detects_truncation(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    episode_path = _write_prepared_delivery(root)
    episode_path.write_bytes(b"")

    report = verify_lerobot_import(root)

    assert not report.ok
    assert report.status is VerificationStatus.DAMAGED
    reasons = {finding.reason for finding in report.findings}
    assert reasons == {REASON_SIZE_MISMATCH, REASON_CONTENT_ID_MISMATCH}
    assert exit_code_for(report) == 1


def test_verify_lerobot_import_detects_same_length_byte_replacement(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    episode_path = _write_prepared_delivery(root, payload=b"episode-0")
    original_size = episode_path.stat().st_size
    episode_path.write_bytes(b"Episode-0")
    assert episode_path.stat().st_size == original_size

    report = verify_lerobot_import(root)

    assert report.status is VerificationStatus.DAMAGED
    assert [finding.reason for finding in report.findings] == [REASON_CONTENT_ID_MISMATCH]
    assert all(finding.uri.endswith("lerobot_episode_0001.mcap") for finding in report.findings)


def test_verify_lerobot_import_detects_a_missing_landing_object(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    episode_path = _write_prepared_delivery(root)
    episode_path.unlink()

    report = verify_lerobot_import(root)

    assert report.status is VerificationStatus.DAMAGED
    assert [finding.reason for finding in report.findings] == [REASON_MISSING]
    assert exit_code_for(report) == 1


def test_verify_lerobot_import_is_unverifiable_without_a_manifest(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    report = verify_lerobot_import(root)

    assert report.status is VerificationStatus.UNVERIFIABLE
    assert report.findings == []
    assert exit_code_for(report) == 3


def test_verify_lerobot_import_accepts_an_empty_episodes_list(tmp_path: Path) -> None:
    """A readable receipt that claims nothing is clean, not unverifiable."""
    root = tmp_path / "delivery"
    root.mkdir()
    (root / "prepared-manifest.json").write_text(
        json.dumps({"schema_version": 3, "episodes": []}),
        encoding="utf-8",
    )

    report = verify_lerobot_import(root)

    assert report.status is VerificationStatus.OK
    assert report.ok
    assert exit_code_for(report) == 0


def test_verify_lerobot_import_refuses_corrupt_manifest_json(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    root.mkdir()
    (root / "prepared-manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        verify_lerobot_import(root)


def test_verify_lerobot_import_refuses_unsupported_schema_version(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    root.mkdir()
    (root / "prepared-manifest.json").write_text(
        json.dumps({"schema_version": 2, "episodes": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 3"):
        verify_lerobot_import(root)


def test_verify_lerobot_import_reads_a_copied_delivery_not_the_original(
    tmp_path: Path,
) -> None:
    """Kevin's #432 DoD: verify the root handed in, never the publish-time path."""
    original = tmp_path / "original"
    original_episode = _write_prepared_delivery(original, payload=b"episode-0")
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    copied_episode = copied / "landing" / "lerobot_episode_0001.mcap"

    # Manifest still names the absolute original uri; that is the recipient case.
    manifest = json.loads((copied / "prepared-manifest.json").read_text(encoding="utf-8"))
    assert manifest["episodes"][0]["uri"] == str(original_episode.resolve())

    report = verify_lerobot_import(copied)
    assert report.status is VerificationStatus.OK

    # Prove the copy was hashed: mutating only the original must not matter.
    original_episode.write_bytes(b"Episode-0")
    assert verify_lerobot_import(copied).status is VerificationStatus.OK

    # Damaging only the copy must go red (the false-pass Kevin reproduced).
    copied_episode.write_bytes(b"Episode-0")
    damaged = verify_lerobot_import(copied)
    assert damaged.status is VerificationStatus.DAMAGED
    assert [finding.reason for finding in damaged.findings] == [REASON_CONTENT_ID_MISMATCH]


def test_verify_lerobot_import_accepts_a_copy_when_the_original_is_gone(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    _write_prepared_delivery(original, payload=b"episode-0")
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    shutil.rmtree(original)

    report = verify_lerobot_import(copied)

    assert report.status is VerificationStatus.OK
    assert report.findings == []


def test_verify_lerobot_import_resolves_bucket_deliveries_under_the_verified_root(
    tmp_path: Path,
    bucket_over_tmp: tuple[BucketStorageRoot, Path],
) -> None:
    """Bucket roots use the same landing/<basename> fetch; no cloud credentials."""
    original_root, original_remote = bucket_over_tmp
    payload = b"episode-0"
    staged = tmp_path / "staged.mcap"
    staged.write_bytes(payload)
    published_uri = original_root.publish(staged, "landing/lerobot_episode_0001.mcap")
    content_id = content_episode_id(staged)
    size_bytes = staged.stat().st_size
    manifest = {
        "schema_version": 3,
        "episodes": [
            {
                "uri": published_uri,
                "content_id": content_id,
                "size_bytes": size_bytes,
            }
        ],
    }
    manifest_path = tmp_path / "prepared-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_root.publish(manifest_path, "prepared-manifest.json")

    assert verify_lerobot_import(original_root).status is VerificationStatus.OK

    copied_remote = tmp_path / "bucket-copy"
    shutil.copytree(original_remote, copied_remote)
    copied_root = BucketStorageRoot(
        f"file://{copied_remote}",
        mirror=tmp_path / "bucket-copy-mirror",
    )

    assert verify_lerobot_import(copied_root).status is VerificationStatus.OK

    # Damage only the copy; original bucket prefix stays intact.
    (copied_remote / "landing" / "lerobot_episode_0001.mcap").write_bytes(b"Episode-0")
    damaged = verify_lerobot_import(copied_root)
    assert damaged.status is VerificationStatus.DAMAGED
    assert [finding.reason for finding in damaged.findings] == [REASON_CONTENT_ID_MISMATCH]
    assert verify_lerobot_import(original_root).status is VerificationStatus.OK


def test_cli_verify_lerobot_import_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    clean_root = tmp_path / "clean"
    _write_prepared_delivery(clean_root)
    assert main(["verify", "lerobot-import", str(clean_root)]) == 0
    assert "ok" in capsys.readouterr().out

    damaged_root = tmp_path / "damaged"
    episode_path = _write_prepared_delivery(damaged_root)
    episode_path.write_bytes(b"")
    assert main(["verify", "lerobot-import", str(damaged_root)]) == 1
    damaged_err = capsys.readouterr().err
    assert REASON_SIZE_MISMATCH in damaged_err
    assert REASON_CONTENT_ID_MISMATCH in damaged_err

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert main(["verify", "lerobot-import", str(empty_root)]) == 3
    assert "unverifiable" in capsys.readouterr().err

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    (bad_root / "prepared-manifest.json").write_text("{broken", encoding="utf-8")
    assert main(["verify", "lerobot-import", str(bad_root)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
