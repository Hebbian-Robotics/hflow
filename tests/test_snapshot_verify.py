"""Snapshot verification: the delivery checked against its receipt (#428).

Every test drives the REAL pipeline: a catalog is built, a snapshot is
exported through the public API, the delivery is damaged surgically, and
verify_dataset_snapshot must report exactly the damage -- nothing more,
nothing less.
"""

import json
import shutil
from pathlib import Path

import pytest
from test_dataset_snapshot import _append_snapshot_episode

import hflow
from hflow.catalog import Catalog
from hflow.cli import main as cli_main
from hflow.snapshot import verify_dataset_snapshot


def _export_two_episode_snapshot(tmp_path: Path, media_mode: str) -> tuple[Path, dict]:
    catalog = Catalog(tmp_path / "catalog")
    selected_episode_id, _ = _append_snapshot_episode(
        catalog, tmp_path, name="fold-shirt", score=0.75, with_media=(media_mode == "copy")
    )
    _append_snapshot_episode(catalog, tmp_path, name="pour-water", score=0.25, with_media=False)
    manifest = tmp_path / "manifest.parquet"
    hflow.curate(
        catalog.location,
        f"SELECT episode_id FROM episodes WHERE episode_id = '{selected_episode_id}'",
        output=manifest,
    )
    output_directory = tmp_path / "dataset-snapshot"
    hflow.export_dataset_snapshot(
        catalog.location, output_directory, manifest=manifest, media_mode=media_mode
    )
    marker = json.loads((output_directory / "format.json").read_text())
    return output_directory, marker


def _rewrite_format_without_integrity(output_directory: Path) -> None:
    marker_path = output_directory / "format.json"
    marker = json.loads(marker_path.read_text())
    marker.pop("integrity", None)
    marker_path.write_text(json.dumps(marker, indent=2))


def test_clean_snapshot_verifies_clean_in_references_mode(tmp_path: Path) -> None:
    output_directory, _ = _export_two_episode_snapshot(tmp_path, "references")
    report = verify_dataset_snapshot(output_directory)
    assert report.ok
    assert report.findings == []


def test_clean_snapshot_verifies_clean_in_copy_mode(tmp_path: Path) -> None:
    output_directory, _ = _export_two_episode_snapshot(tmp_path, "copy")
    report = verify_dataset_snapshot(output_directory)
    assert report.ok
    assert report.findings == []


def test_bytes_changed_reports_content_mismatch_alone(tmp_path: Path) -> None:
    """Same size, different bytes: the receipt must call this a content
    mismatch, not a size mismatch, and must not raise."""
    output_directory, marker = _export_two_episode_snapshot(tmp_path, "references")
    table_path = output_directory / marker["integrity"]["tables"]["samples"]["path"]
    data = bytearray(table_path.read_bytes())
    data[len(data) // 2] ^= 0xFF  # same length, different content
    table_path.write_bytes(bytes(data))

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    assert [f.reason for f in report.findings] == ["content-id-mismatch"]
    finding = report.findings[0]
    assert finding.uri == marker["integrity"]["tables"]["samples"]["path"]
    assert finding.detail


def test_missing_file_reports_missing_alone(tmp_path: Path) -> None:
    output_directory, marker = _export_two_episode_snapshot(tmp_path, "references")
    (output_directory / marker["integrity"]["tables"]["samples"]["path"]).unlink()

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    assert [f.reason for f in report.findings] == ["missing"]
    assert marker["integrity"]["tables"]["samples"]["path"] in report.findings[0].uri


def test_removed_receipt_entry_and_file_raise_inventory_mismatch(tmp_path: Path) -> None:
    """#473's deleted-member case: when a receipt entry and its file are both
    gone, the surviving entries agree with each other and every per-file
    check passes; only the stored inventory content_id, computed over the
    original set, can witness the loss. The marker is internally
    inconsistent, so verify raises (CLI exit 2) instead of certifying."""

    def strip_measurements(output_directory: Path) -> str:
        marker_path = output_directory / "format.json"
        marker = json.loads(marker_path.read_text())
        entry = marker["integrity"]["tables"].pop("measurements")
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        (output_directory / entry["path"]).unlink()
        return entry["path"]

    output_directory, _ = _export_two_episode_snapshot(tmp_path, "references")
    removed = strip_measurements(output_directory)

    with pytest.raises(ValueError, match="content_id"):
        verify_dataset_snapshot(output_directory)

    # A fresh export for the CLI path: the raise must map to exit 2, the
    # unreadable-input code, not to a findings-based exit.
    output_directory, _ = _export_two_episode_snapshot(tmp_path / "cli", "references")
    strip_measurements(output_directory)
    assert cli_main(["verify", "snapshot", str(output_directory)]) == 2
    assert removed


def test_deleted_file_with_intact_receipt_reports_missing(tmp_path: Path) -> None:
    """Negative control for #473: delete the file but keep its receipt entry.
    This is the ordinary ``missing`` path and must keep reporting DAMAGED
    with or without the inventory gate; it exercises the per-file loop, not
    the gate."""
    output_directory, marker = _export_two_episode_snapshot(tmp_path, "references")
    (output_directory / marker["integrity"]["tables"]["measurements"]["path"]).unlink()

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    assert [f.reason for f in report.findings] == ["missing"]


@pytest.mark.parametrize(
    ("replacement", "label"),
    [(None, "absent"), ("", "empty"), (0, "not-a-string"), ([], "wrong-type")],
    ids=["absent", "empty", "not-a-string", "wrong-type"],
)
def test_receipt_without_a_usable_content_id_is_refused(
    tmp_path: Path, replacement: object, label: str
) -> None:
    """The other half of the #473 gate, which the mismatch test cannot reach.

    A receipt whose ``content_id`` is missing or unusable cannot witness a
    deleted member at all, so certifying it would be certifying that the
    check ran. Deleting this branch left the whole suite green, so it needs
    its own case. An ``integrity`` block with no ``content_id`` is not
    something hflow writes (both arrived in #401), which is exactly why a
    marker carrying one is unreadable input rather than damaged bytes.
    """
    output_directory, _ = _export_two_episode_snapshot(tmp_path, "references")
    marker_path = output_directory / "format.json"
    marker = json.loads(marker_path.read_text())
    if replacement is None:
        marker["integrity"].pop("content_id")
    else:
        marker["integrity"]["content_id"] = replacement
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="no usable content_id"):
        verify_dataset_snapshot(output_directory)

    assert cli_main(["verify", "snapshot", str(output_directory)]) == 2, label


def test_truncated_file_reports_size_mismatch_and_skips_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size is the cheap pre-filter: a truncated file is reported by size
    alone without spending the hash read."""
    output_directory, marker = _export_two_episode_snapshot(tmp_path, "references")
    table_path = output_directory / marker["integrity"]["tables"]["measurements"]["path"]
    original = table_path.read_bytes()
    table_path.write_bytes(original[: len(original) // 2])

    hashed: list[Path] = []
    real = hflow.snapshot._sha256_hex

    def spy(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(hflow.snapshot, "_sha256_hex", spy)

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    reasons = [f.reason for f in report.findings]
    assert "size-mismatch" in reasons
    assert "content-id-mismatch" not in reasons
    truncated_uri = marker["integrity"]["tables"]["measurements"]["path"]
    assert truncated_uri not in [str(path) for path in hashed]
    assert hashed, "spy must have recorded at least one hashed file"


def test_copied_asset_damage_is_reported(tmp_path: Path) -> None:
    """Copy mode stores media inside the snapshot; the receipt covers it."""
    output_directory, marker = _export_two_episode_snapshot(tmp_path, "copy")
    assert marker["integrity"]["assets"], "fixture must include copied assets"
    asset_uri = marker["integrity"]["assets"][0]["path"]
    asset_path = output_directory / asset_uri
    data = bytearray(asset_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    asset_path.write_bytes(bytes(data))

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    damaged = [f for f in report.findings if f.uri == asset_uri]
    assert damaged and damaged[0].reason == "content-id-mismatch"


def test_pre_401_format_json_is_unverifiable_not_corrupt(tmp_path: Path) -> None:
    """A valid v1 snapshot without an integrity receipt is unverifiable, not
    corrupt: the finding says no_receipt and nothing raises."""
    output_directory, _ = _export_two_episode_snapshot(tmp_path, "references")
    _rewrite_format_without_integrity(output_directory)

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    assert [f.reason for f in report.findings] == ["no-receipt"]


def test_extra_files_under_assets_are_ignored(tmp_path: Path) -> None:
    """Files the receipt does not name produce no finding and no warning:
    unlisted extras are outside the receipt's contract."""
    output_directory, _ = _export_two_episode_snapshot(tmp_path, "copy")
    (output_directory / "assets" / "unlisted-extra.bin").write_bytes(b"extra bytes")

    report = verify_dataset_snapshot(output_directory)

    assert report.ok
    assert report.findings == []


def test_partial_transfer_reports_every_mismatch_in_one_report(tmp_path: Path) -> None:
    """A partial transfer usually damages more than one file: the report
    carries every mismatch in one list instead of raising on the first."""
    output_directory, marker = _export_two_episode_snapshot(tmp_path, "references")
    samples_path = output_directory / marker["integrity"]["tables"]["samples"]["path"]
    data = bytearray(samples_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    samples_path.write_bytes(bytes(data))
    (output_directory / marker["integrity"]["tables"]["tags"]["path"]).unlink()

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    reasons = sorted(f.reason for f in report.findings)
    assert reasons == ["content-id-mismatch", "missing"]


def test_verify_snapshot_cli_exit_codes(tmp_path: Path) -> None:
    """0 clean, 1 damaged, 3 unverifiable, 2 unreadable -- through the CLI."""
    output_directory, _ = _export_two_episode_snapshot(tmp_path, "references")
    argv = ["verify", "snapshot", str(output_directory)]
    assert cli_main(argv) == 0

    marker_path = output_directory / "format.json"
    marker = json.loads(marker_path.read_text())
    table_path = output_directory / marker["integrity"]["tables"]["samples"]["path"]
    data = bytearray(table_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    table_path.write_bytes(bytes(data))
    assert cli_main(argv) == 1

    _rewrite_format_without_integrity(output_directory)
    assert cli_main(argv) == 3

    marker_path.write_bytes(b"")
    assert cli_main(argv) == 2

    marker_path.write_bytes(b"\x00\xff\xfe\x01garbage")
    assert cli_main(argv) == 2

    missing_root = str(tmp_path / "does-not-exist")
    assert cli_main(["verify", "snapshot", missing_root]) == 2


def test_verify_is_read_only_against_the_delivery(tmp_path: Path) -> None:
    """verify must not rewrite format.json, the manifest, the table files,
    or the catalog. A re-read after a verify run must return byte-identical
    contents and unchanged mtimes."""
    output_directory, _ = _export_two_episode_snapshot(tmp_path, "references")
    catalog_path = tmp_path / "catalog"

    def snapshot_everything() -> dict[Path, tuple[bytes, float]]:
        snapshot: dict[Path, tuple[bytes, float]] = {}
        for path in [*output_directory.rglob("*"), *catalog_path.rglob("*")]:
            if path.is_file():
                stat = path.stat()
                snapshot[path] = (path.read_bytes(), stat.st_mtime)
        return snapshot

    before = snapshot_everything()
    report = verify_dataset_snapshot(output_directory)
    after = snapshot_everything()

    assert report.ok
    assert before == after, "verify rewrote at least one file under the delivery or catalog"


def test_moved_root_verifies_from_the_new_root_alone(tmp_path: Path) -> None:
    """Kingston's homework (#428): no absolute-path read may survive.

    Export to root A, copy the whole delivery to root B, delete A entirely.
    Verify B clean; damage one file under B and verify B again. With A gone,
    any read outside the handed root would fail or see nothing, so a clean
    and then a damaged report can only come from B's own bytes.
    """
    root_a, marker = _export_two_episode_snapshot(tmp_path, "references")
    root_b = tmp_path / "moved-delivery"
    shutil.copytree(root_a, root_b)
    shutil.rmtree(root_a)

    assert not root_a.exists()
    clean_report = verify_dataset_snapshot(root_b)
    assert clean_report.ok
    assert clean_report.findings == []

    table_path = root_b / marker["integrity"]["tables"]["samples"]["path"]
    data = bytearray(table_path.read_bytes())
    data[len(data) // 2] ^= 0xFF  # same length, different content
    table_path.write_bytes(bytes(data))

    damaged_report = verify_dataset_snapshot(root_b)
    assert not damaged_report.ok
    assert [f.reason for f in damaged_report.findings] == ["content-id-mismatch"]
    assert damaged_report.findings[0].uri == marker["integrity"]["tables"]["samples"]["path"]


def test_damage_is_reported_from_the_verified_root_not_the_export_root(
    tmp_path: Path,
) -> None:
    """Variant with both roots alive: a damaged copy B must be reported as
    damaged, not masked by the still-clean original A. Verification reads
    only the root it was handed."""
    root_a, marker = _export_two_episode_snapshot(tmp_path, "references")
    root_b = tmp_path / "damaged-delivery"
    shutil.copytree(root_a, root_b)
    table_path = root_b / marker["integrity"]["tables"]["samples"]["path"]
    data = bytearray(table_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    table_path.write_bytes(bytes(data))

    assert verify_dataset_snapshot(root_a).ok
    damaged_report = verify_dataset_snapshot(root_b)
    assert not damaged_report.ok
    assert [f.reason for f in damaged_report.findings] == ["content-id-mismatch"]
