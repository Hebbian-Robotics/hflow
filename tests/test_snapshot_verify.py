"""Snapshot verification: the delivery checked against its receipt (#428).

Every test drives the REAL pipeline: a catalog is built, a snapshot is
exported through the public API, the delivery is damaged surgically, and
verify_dataset_snapshot must report exactly the damage -- nothing more,
nothing less.
"""

import copy
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
from catalog_test_helpers import append_snapshot_episode

import hflow
from hflow.catalog import Catalog
from hflow.cli import main as cli_main
from hflow.snapshot import verify_dataset_snapshot


def _export_two_episode_snapshot(tmp_path: Path, media_mode: str) -> tuple[Path, dict]:
    catalog = Catalog(tmp_path / "catalog")
    selected_episode_id, _ = append_snapshot_episode(
        catalog, tmp_path, name="fold-shirt", score=0.75, with_media=(media_mode == "copy")
    )
    append_snapshot_episode(catalog, tmp_path, name="pour-water", score=0.25, with_media=False)
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


ExportedSnapshotCopier = Callable[[Path, str], tuple[Path, dict]]


@pytest.fixture(scope="module")
def exported_snapshot_templates(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[Path, dict]]:
    """One real export per media mode, built once and never handed out.

    Tests damage their delivery, so each one gets a copy (see
    ``exported_snapshot``). Tests about which root verify reads, and the
    read-only test that also watches the catalog, export fresh instead: a
    surviving template would mask a read outside the handed root.
    """
    return {
        media_mode: _export_two_episode_snapshot(
            tmp_path_factory.mktemp(f"snapshot-template-{media_mode}"), media_mode
        )
        for media_mode in ("references", "copy")
    }


@pytest.fixture
def exported_snapshot(
    exported_snapshot_templates: dict[str, tuple[Path, dict]],
) -> ExportedSnapshotCopier:
    """Copy the module's export for ``media_mode`` under ``destination_root``."""

    def copy_exported_snapshot(destination_root: Path, media_mode: str) -> tuple[Path, dict]:
        template_directory, marker = exported_snapshot_templates[media_mode]
        output_directory = destination_root / "dataset-snapshot"
        shutil.copytree(template_directory, output_directory)
        return output_directory, copy.deepcopy(marker)

    return copy_exported_snapshot


def _flip_middle_byte(path: Path) -> None:
    """Same length, different content: a damage only the hash can see."""
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


EditResult = TypeVar("EditResult")


def _edit_marker(
    output_directory: Path, edit: Callable[[dict[str, Any]], EditResult]
) -> EditResult:
    """Apply ``edit`` to format.json in place and rewrite it as hflow does."""
    marker_path = output_directory / "format.json"
    marker = json.loads(marker_path.read_text())
    edit_result = edit(marker)
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return edit_result


def _rewrite_format_without_integrity(output_directory: Path) -> None:
    marker_path = output_directory / "format.json"
    marker = json.loads(marker_path.read_text())
    marker.pop("integrity", None)
    marker_path.write_text(json.dumps(marker, indent=2))


@pytest.mark.parametrize("media_mode", ["references", "copy"])
def test_clean_snapshot_verifies_clean(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier, media_mode: str
) -> None:
    output_directory, _ = exported_snapshot(tmp_path, media_mode)
    report = verify_dataset_snapshot(output_directory)
    assert report.ok
    assert report.findings == []


def test_bytes_changed_reports_content_mismatch_alone(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """Same size, different bytes: the receipt must call this a content
    mismatch, not a size mismatch, and must not raise."""
    output_directory, marker = exported_snapshot(tmp_path, "references")
    table_path = output_directory / marker["integrity"]["tables"]["samples"]["path"]
    _flip_middle_byte(table_path)  # same length, different content

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    assert [f.reason for f in report.findings] == ["content-id-mismatch"]
    finding = report.findings[0]
    assert finding.uri == marker["integrity"]["tables"]["samples"]["path"]
    assert finding.detail


def test_missing_file_reports_missing_alone(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    output_directory, marker = exported_snapshot(tmp_path, "references")
    (output_directory / marker["integrity"]["tables"]["samples"]["path"]).unlink()

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    assert [f.reason for f in report.findings] == ["missing"]
    assert marker["integrity"]["tables"]["samples"]["path"] in report.findings[0].uri


def test_removed_receipt_entry_and_file_raise_inventory_mismatch(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """#473's deleted-member case: when a receipt entry and its file are both
    gone, the surviving entries agree with each other and every per-file
    check passes; only the stored inventory content_id, computed over the
    original set, can witness the loss. The marker is internally
    inconsistent, so verify raises (CLI exit 2) instead of certifying."""

    def strip_measurements(output_directory: Path) -> None:
        entry = _edit_marker(
            output_directory, lambda marker: marker["integrity"]["tables"].pop("measurements")
        )
        (output_directory / entry["path"]).unlink()

    output_directory, _ = exported_snapshot(tmp_path, "references")
    strip_measurements(output_directory)

    with pytest.raises(ValueError, match="content_id"):
        verify_dataset_snapshot(output_directory)

    # A second delivery for the CLI path: the raise must map to exit 2, the
    # unreadable-input code, not to a findings-based exit.
    output_directory, _ = exported_snapshot(tmp_path / "cli", "references")
    strip_measurements(output_directory)
    assert cli_main(["verify", "snapshot", str(output_directory)]) == 2


def test_deleted_file_with_intact_receipt_reports_missing(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """Negative control for #473: delete the file but keep its receipt entry.
    This is the ordinary ``missing`` path and must keep reporting DAMAGED
    with or without the inventory gate; it exercises the per-file loop, not
    the gate."""
    output_directory, marker = exported_snapshot(tmp_path, "references")
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
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier, replacement: object, label: str
) -> None:
    """The other half of the #473 gate, which the mismatch test cannot reach.

    A receipt whose ``content_id`` is missing or unusable cannot witness a
    deleted member at all, so certifying it would be certifying that the
    check ran. Deleting this branch left the whole suite green, so it needs
    its own case. An ``integrity`` block with no ``content_id`` is not
    something hflow writes (both arrived in #401), which is exactly why a
    marker carrying one is unreadable input rather than damaged bytes.
    """
    output_directory, _ = exported_snapshot(tmp_path, "references")

    def replace_content_id(marker: dict[str, Any]) -> None:
        if replacement is None:
            marker["integrity"].pop("content_id")
        else:
            marker["integrity"]["content_id"] = replacement

    _edit_marker(output_directory, replace_content_id)

    with pytest.raises(ValueError, match="no usable content_id"):
        verify_dataset_snapshot(output_directory)

    assert cli_main(["verify", "snapshot", str(output_directory)]) == 2, label


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("tables", None, r"integrity\.tables must be a JSON object"),
        ("tables", [], r"integrity\.tables must be a JSON object"),
        ("assets", None, r"integrity\.assets must be a JSON array"),
        ("assets", {}, r"integrity\.assets must be a JSON array"),
    ],
    ids=["tables-null", "tables-array", "assets-null", "assets-object"],
)
def test_null_or_wrong_type_integrity_containers_are_refused_at_the_boundary(
    tmp_path: Path,
    exported_snapshot: ExportedSnapshotCopier,
    field: str,
    replacement: object,
    match: str,
) -> None:
    """#575: present-but-null (or wrong-type) tables/assets used to crash.

    ``integrity.get("tables", {})`` does not apply when the key exists with
    JSON ``null``, so verify raised AttributeError/TypeError through the CLI
    instead of exit 2. Same family as #489's typed entry boundary.
    """
    output_directory, _ = exported_snapshot(tmp_path, "references")
    _edit_marker(
        output_directory, lambda marker: marker["integrity"].__setitem__(field, replacement)
    )

    with pytest.raises(ValueError, match=match):
        verify_dataset_snapshot(output_directory)

    assert cli_main(["verify", "snapshot", str(output_directory)]) == 2


def test_truncated_file_reports_size_mismatch_and_skips_the_hash(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size is the cheap pre-filter: a truncated file is reported by size
    alone without spending the hash read."""
    output_directory, marker = exported_snapshot(tmp_path, "references")
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


def test_copied_asset_damage_is_reported(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """Copy mode stores media inside the snapshot; the receipt covers it."""
    output_directory, marker = exported_snapshot(tmp_path, "copy")
    assert marker["integrity"]["assets"], "fixture must include copied assets"
    asset_uri = marker["integrity"]["assets"][0]["path"]
    asset_path = output_directory / asset_uri
    _flip_middle_byte(asset_path)

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    damaged = [f for f in report.findings if f.uri == asset_uri]
    assert damaged and damaged[0].reason == "content-id-mismatch"


def test_pre_401_format_json_is_unverifiable_not_corrupt(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """A valid v1 snapshot without an integrity receipt is unverifiable, not
    corrupt: the finding says no_receipt and nothing raises."""
    output_directory, _ = exported_snapshot(tmp_path, "references")
    _rewrite_format_without_integrity(output_directory)

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    assert [f.reason for f in report.findings] == ["no-receipt"]


def test_foreign_marker_is_refused_at_the_boundary(tmp_path: Path) -> None:
    """#472: a directory the exporter would refuse cannot be certified. A
    marker with an integrity-shaped key but no format identity never reaches
    the receipt logic; verify raises and the CLI maps to exit 2."""
    foreign = tmp_path / "some-other-tools-output"
    foreign.mkdir()
    payload = b"not-a-hflow-snapshot-at-all"
    (foreign / "data.parquet").write_bytes(payload)
    (foreign / "format.json").write_text(
        json.dumps(
            {
                "producer": "not-hflow",
                "integrity": {
                    "tables": {
                        "data": {
                            "path": "data.parquet",
                            "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="not a 'hflow-dataset-snapshot'"):
        verify_dataset_snapshot(foreign)

    assert cli_main(["verify", "snapshot", str(foreign)]) == 2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        # #472: version 1 is the only version there has ever been, and the
        # comparison is deliberately identical to the writer, which records
        # the version as a string. A future version raises, and so does a JSON
        # number 1: an easy honest mistake, so the error says exactly why.
        pytest.param("format_version", "2", "format_version '2'", id="future-version"),
        pytest.param("format_version", 1, "JSON number 1 is refused", id="numeric-version"),
        # The other half of the identity predicate.
        # `test_foreign_marker_is_refused_at_the_boundary` uses a marker
        # carrying neither field, so the version check alone refuses it and the
        # format-name check is never the thing that fires. Dropping the name
        # comparison from the predicate left the whole suite green. A marker
        # claiming version 1 of somebody else's format is still not ours.
        pytest.param(
            "format",
            "someone-elses-dataset-snapshot",
            "someone-elses-dataset-snapshot",
            id="foreign-format-name",
        ),
    ],
)
def test_a_marker_that_is_not_our_format_version_1_is_refused(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier, field: str, value: object, match: str
) -> None:
    output_directory, _ = exported_snapshot(tmp_path, "references")
    _edit_marker(output_directory, lambda marker: marker.__setitem__(field, value))

    with pytest.raises(ValueError, match=match):
        verify_dataset_snapshot(output_directory)
    assert cli_main(["verify", "snapshot", str(output_directory)]) == 2


def test_a_rewritten_marker_with_the_string_version_1_still_verifies(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """The control for the version refusals: the writer's own spelling passes."""
    output_directory, _ = exported_snapshot(tmp_path, "references")
    _edit_marker(output_directory, lambda marker: marker.__setitem__("format_version", "1"))
    assert cli_main(["verify", "snapshot", str(output_directory)]) == 0


def test_extra_files_under_assets_are_ignored(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """Files the receipt does not name produce no finding and no warning:
    unlisted extras are outside the receipt's contract."""
    output_directory, _ = exported_snapshot(tmp_path, "copy")
    (output_directory / "assets" / "unlisted-extra.bin").write_bytes(b"extra bytes")

    report = verify_dataset_snapshot(output_directory)

    assert report.ok
    assert report.findings == []


def test_partial_transfer_reports_every_mismatch_in_one_report(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """A partial transfer usually damages more than one file: the report
    carries every mismatch in one list instead of raising on the first."""
    output_directory, marker = exported_snapshot(tmp_path, "references")
    samples_path = output_directory / marker["integrity"]["tables"]["samples"]["path"]
    _flip_middle_byte(samples_path)
    (output_directory / marker["integrity"]["tables"]["tags"]["path"]).unlink()

    report = verify_dataset_snapshot(output_directory)

    assert not report.ok
    reasons = sorted(f.reason for f in report.findings)
    assert reasons == ["content-id-mismatch", "missing"]


def test_verify_snapshot_cli_exit_codes(
    tmp_path: Path, exported_snapshot: ExportedSnapshotCopier
) -> None:
    """0 clean, 1 damaged, 3 unverifiable, 2 unreadable -- through the CLI."""
    output_directory, _ = exported_snapshot(tmp_path, "references")
    argv = ["verify", "snapshot", str(output_directory)]
    assert cli_main(argv) == 0

    marker_path = output_directory / "format.json"
    marker = json.loads(marker_path.read_text())
    table_path = output_directory / marker["integrity"]["tables"]["samples"]["path"]
    _flip_middle_byte(table_path)
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
    _flip_middle_byte(table_path)  # same length, different content

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
    _flip_middle_byte(table_path)

    assert verify_dataset_snapshot(root_a).ok
    damaged_report = verify_dataset_snapshot(root_b)
    assert not damaged_report.ok
    assert [f.reason for f in damaged_report.findings] == ["content-id-mismatch"]


_KNOWN_RECEIPT_ENTRIES: list[dict[str, str | int]] = [
    {
        "path": "samples.parquet",
        "size_bytes": 164981,
        "sha256": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
    },
    {
        "path": "measurements.parquet",
        "size_bytes": 5223,
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    },
    {
        "path": "assets/wrist_cam/frame_0000000001.jpg",
        "size_bytes": 20481,
        "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
    },
]

# The digest over these exact entries, serialized the way the exporter has
# always done it (sorted by path, keys sorted, compact separators). If this
# value changes, every snapshot ever exported fails verification.
_GOLDEN_INVENTORY_CONTENT_ID = "b4cd6b846051175bbf1c57e5f5fcd5edee479b5cf2afd8294335c071561217bf"


def test_inventory_digest_is_byte_identical_through_the_record_bridge() -> None:
    """#489's hard constraint: typing the receipt entries must not move the
    content_id hash by one byte. The old path hashes raw dicts straight from
    the marker; the new path hashes records converted back through
    ``to_dict_for_hashing``. Both must produce the same string and the same
    digest, and the digest must equal the golden value."""
    old_payload = json.dumps(
        sorted(_KNOWN_RECEIPT_ENTRIES, key=lambda entry: str(entry["path"])),
        sort_keys=True,
        separators=(",", ":"),
    )
    records = [
        hflow.snapshot._parse_file_integrity_record(entry) for entry in _KNOWN_RECEIPT_ENTRIES
    ]
    new_payload = json.dumps(
        [record.to_dict_for_hashing() for record in sorted(records, key=lambda r: r.path)],
        sort_keys=True,
        separators=(",", ":"),
    )

    assert new_payload == old_payload
    assert hflow.snapshot._inventory_content_id(records) == _GOLDEN_INVENTORY_CONTENT_ID


def test_the_digest_covers_the_three_delivery_fields_and_nothing_else() -> None:
    """Typing the entries narrowed what the digest is computed over, and that
    is a decision rather than an accident.

    Hashing raw dicts meant the digest depended on every key an entry
    happened to carry. Hashing records means it depends on exactly ``path``,
    ``size_bytes`` and ``sha256``, which are the three facts that define a
    delivery. An entry carrying an extra key therefore hashes the same now
    and used to hash differently.

    The consequence to keep: additive metadata in a later format revision
    cannot silently invalidate the digest of every snapshot already
    exported. The consequence to know: a marker whose entries were edited to
    add a field is no longer caught here. That is not a loss, because the
    receipt travels unsigned inside the file it describes and was never a
    tamper defence, and the guarantee the docs make (a deleted member stays
    visible) is unaffected. Restoring raw-dict hashing to "tighten" this
    would trade a real compatibility property for an imaginary one.
    """
    entries_with_an_extra_field = [
        {**entry, "injected_field": "not written by hflow"} for entry in _KNOWN_RECEIPT_ENTRIES
    ]
    records = [
        hflow.snapshot._parse_file_integrity_record(entry) for entry in entries_with_an_extra_field
    ]

    assert hflow.snapshot._inventory_content_id(records) == _GOLDEN_INVENTORY_CONTENT_ID

    # And the deleted-member guarantee still holds over the narrowed digest.
    assert hflow.snapshot._inventory_content_id(records[:-1]) != _GOLDEN_INVENTORY_CONTENT_ID


def test_receipt_entry_with_numeric_sha256_is_refused_at_the_boundary() -> None:
    """#489's silent bug: a receipt whose sha256 arrived as a JSON number
    used to reach a per-file comparison that can never succeed and was
    reported as damaged bytes. The boundary refuses it instead, naming the
    field, because a malformed receipt is unreadable input."""
    from hflow.snapshot import _parse_file_integrity_record

    with pytest.raises(ValueError, match="sha256"):
        _parse_file_integrity_record({"path": "samples.parquet", "size_bytes": 10, "sha256": 123})


def _hand_built_snapshot_with_receipt_path(
    snap: Path, *, receipt_path: str, payload: bytes = b"secret-bytes"
) -> None:
    """Minimal identity+integrity marker whose single receipt uses ``receipt_path``.

    ``content_id`` matches that one-entry inventory so the deleted-member gate
    is not the thing that fires; the path containment check is.
    """
    from hflow.snapshot import (
        DATASET_SNAPSHOT_FORMAT_NAME,
        DATASET_SNAPSHOT_FORMAT_VERSION,
        FileIntegrityRecord,
        _inventory_content_id,
    )

    digest = hashlib.sha256(payload).hexdigest()
    record = FileIntegrityRecord(path=receipt_path, size_bytes=len(payload), sha256=digest)
    marker = {
        "format": DATASET_SNAPSHOT_FORMAT_NAME,
        "format_version": DATASET_SNAPSHOT_FORMAT_VERSION,
        "media_mode": "references",
        "media_uri_base": None,
        "tables": ["samples.parquet"],
        "integrity": {
            "tables": {
                "samples": {
                    "path": record.path,
                    "size_bytes": record.size_bytes,
                    "sha256": record.sha256,
                }
            },
            "assets": [],
            "content_id": _inventory_content_id([record]),
        },
    }
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "format.json").write_text(json.dumps(marker, indent=2) + "\n")


def test_receipt_path_escaping_the_handed_directory_is_refused(tmp_path: Path) -> None:
    """#469: relative ``..`` and absolute paths hash outside the root today;
    both must raise before any read (exit 2), not report ok."""
    snap = tmp_path / "snap"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"secret-bytes")

    for escape_path in (f"../outside/{secret.name}", str(secret.resolve())):
        _hand_built_snapshot_with_receipt_path(snap, receipt_path=escape_path)
        with pytest.raises(ValueError, match="must stay under the handed snapshot directory"):
            verify_dataset_snapshot(snap)
        assert cli_main(["verify", "snapshot", str(snap)]) == 2


def test_normalized_parent_escape_through_an_existing_subdir_is_refused(
    tmp_path: Path,
) -> None:
    """Kingston's third shape: ``tables/../../outside/...`` only looks like it
    failed before because ``snap/tables/`` was missing. Create it and the bare
    join escapes; containment must still refuse before the read."""
    snap = tmp_path / "snap"
    outside = tmp_path / "outside"
    outside.mkdir()
    (snap / "tables").mkdir(parents=True)
    secret = outside / "secret.bin"
    secret.write_bytes(b"secret-bytes")

    _hand_built_snapshot_with_receipt_path(snap, receipt_path=f"tables/../../outside/{secret.name}")
    with pytest.raises(ValueError, match="must stay under the handed snapshot directory"):
        verify_dataset_snapshot(snap)
    assert cli_main(["verify", "snapshot", str(snap)]) == 2
