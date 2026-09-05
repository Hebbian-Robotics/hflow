"""Regression coverage for native overlay manifest parse-stage refusals."""

import csv
import io
import json
from pathlib import Path
from typing import cast

import pytest

from hflow.packaging import (
    CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    CythonOverlayBuildConfig,
    CythonOverlayManifest,
    CythonOverlayManifestError,
    apply_cython_overlay,
    build_cython_overlay,
)


def _write_example_distribution(temporary_directory: Path) -> Path:
    site_packages_directory = temporary_directory / "site-packages"
    package_root = site_packages_directory / "sample_native_package"
    distribution_metadata_root = site_packages_directory / "sample_native_package-7.2.dist-info"
    package_root.mkdir(parents=True)
    distribution_metadata_root.mkdir()

    (package_root / "__init__.py").write_text(
        "from .worker import compute\n\n__all__ = ['compute']\n",
        encoding="utf-8",
    )
    (package_root / "worker.py").write_text(
        "def compute(value: int) -> int:\n    return value * 3\n",
        encoding="utf-8",
    )

    record_path = distribution_metadata_root / "RECORD"
    serialized_record = io.StringIO(newline="")
    writer = csv.writer(serialized_record, lineterminator="\n")
    for path in (
        "sample_native_package/__init__.py",
        "sample_native_package/worker.py",
        "sample_native_package-7.2.dist-info/RECORD",
    ):
        writer.writerow((path, "", ""))
    record_path.write_text(serialized_record.getvalue(), encoding="utf-8")
    return package_root


def _example_record_path(package_root: Path) -> Path:
    return package_root.parent / "sample_native_package-7.2.dist-info" / "RECORD"


def _build_overlay(tmp_path: Path) -> tuple[Path, Path, CythonOverlayManifest, bytes, bytes]:
    package_root = _write_example_distribution(tmp_path)
    source_path = package_root / "worker.py"
    source_bytes = source_path.read_bytes()
    original_record = _example_record_path(package_root).read_bytes()
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    return package_root, overlay_directory, manifest, source_bytes, original_record


def _manifest_payload(manifest_path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))


def _write_manifest_bytes(manifest_path: Path, contents: bytes) -> None:
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(contents)


def _assert_apply_refused_without_mutation(
    package_root: Path,
    overlay_directory: Path,
    manifest: CythonOverlayManifest,
    source_bytes: bytes,
    original_record: bytes,
    expected_message: str,
) -> None:
    with pytest.raises(CythonOverlayManifestError, match=expected_message):
        apply_cython_overlay(overlay_directory, package_root)

    assert (package_root / "worker.py").read_bytes() == source_bytes
    assert not any(
        (package_root / artifact.installed_artifact_path).exists()
        for artifact in manifest.artifacts
    )
    assert not (package_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME).exists()
    assert _example_record_path(package_root).read_bytes() == original_record


def test_apply_refuses_invalid_manifest_json_before_mutation(tmp_path: Path) -> None:
    package_root, overlay_directory, manifest, source_bytes, original_record = _build_overlay(
        tmp_path
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    _write_manifest_bytes(manifest_path, b'{"schema_version":')

    _assert_apply_refused_without_mutation(
        package_root,
        overlay_directory,
        manifest,
        source_bytes,
        original_record,
        "native overlay manifest is not valid JSON",
    )


def test_apply_refuses_non_array_artifacts_before_mutation(tmp_path: Path) -> None:
    package_root, overlay_directory, manifest, source_bytes, original_record = _build_overlay(
        tmp_path
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = _manifest_payload(manifest_path)
    payload["artifacts"] = {}
    _write_manifest_bytes(
        manifest_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    _assert_apply_refused_without_mutation(
        package_root,
        overlay_directory,
        manifest,
        source_bytes,
        original_record,
        "artifacts must be an array",
    )


def test_apply_refuses_semantically_identical_noncanonical_manifest_before_mutation(
    tmp_path: Path,
) -> None:
    package_root, overlay_directory, manifest, source_bytes, original_record = _build_overlay(
        tmp_path
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = _manifest_payload(manifest_path)
    noncanonical_bytes = (json.dumps(payload, indent=4, sort_keys=True) + "\n").encode("utf-8")
    assert json.loads(noncanonical_bytes) == payload
    assert noncanonical_bytes != manifest_path.read_bytes()
    _write_manifest_bytes(manifest_path, noncanonical_bytes)

    _assert_apply_refused_without_mutation(
        package_root,
        overlay_directory,
        manifest,
        source_bytes,
        original_record,
        "native overlay manifest is not canonical JSON",
    )


def test_apply_refuses_duplicate_manifest_fields_before_mutation(tmp_path: Path) -> None:
    package_root, overlay_directory, manifest, source_bytes, original_record = _build_overlay(
        tmp_path
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = _manifest_payload(manifest_path)
    serialized_manifest = manifest_path.read_text(encoding="utf-8").rstrip("\n")
    assert serialized_manifest.endswith("}")
    duplicated_manifest = (
        serialized_manifest[:-1] + f',\n  "format": {json.dumps(payload["format"])}\n}}\n'
    ).encode("utf-8")
    assert json.loads(duplicated_manifest)["format"] == payload["format"]
    _write_manifest_bytes(manifest_path, duplicated_manifest)

    _assert_apply_refused_without_mutation(
        package_root,
        overlay_directory,
        manifest,
        source_bytes,
        original_record,
        "native overlay manifest contains duplicate fields",
    )
