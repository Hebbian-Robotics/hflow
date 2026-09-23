"""Shared fixtures for native overlay packaging tests."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import cast

from hflow.packaging import (
    INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    CythonOverlayBuildConfig,
    CythonOverlayManifest,
    build_cython_overlay,
)

WORKER_MODULE_NAME = "sample_native_package.worker"


def write_example_distribution(
    temporary_directory: Path,
    *,
    include_record: bool = True,
) -> tuple[Path, Path]:
    site_packages_directory = temporary_directory / "site-packages"
    package_root = site_packages_directory / "sample_native_package"
    nested_package_root = package_root / "nested"
    distribution_metadata_root = site_packages_directory / "sample_native_package-7.2.dist-info"
    nested_package_root.mkdir(parents=True)
    distribution_metadata_root.mkdir()

    (package_root / "__init__.py").write_text(
        "from .worker import compute\n\n__all__ = ['compute']\n",
        encoding="utf-8",
    )
    (package_root / "__main__.py").write_text(
        "from .worker import compute\n\nprint(compute(4))\n",
        encoding="utf-8",
    )
    (package_root / "worker.py").write_text(
        (
            "from typing import NewType\n\n"
            "Token = NewType('Token', str)\n"
            "Token.__module__ = __name__\n\n"
            "def compute(value: int) -> int:\n    return value * 3\n"
        ),
        encoding="utf-8",
    )
    (nested_package_root / "__init__.py").write_text(
        "from .labels import label\n\n__all__ = ['label']\n",
        encoding="utf-8",
    )
    (nested_package_root / "labels.py").write_text(
        "def label() -> str:\n    return 'native-result'\n",
        encoding="utf-8",
    )
    (package_root / "py.typed").write_text("", encoding="utf-8")
    (package_root / "worker.pyc").write_bytes(b"legacy-bytecode")
    (distribution_metadata_root / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: sample-native-package\nVersion: 7.2\n",
        encoding="utf-8",
    )
    (distribution_metadata_root / "entry_points.txt").write_text(
        "[console_scripts]\nsample-native = sample_native_package.worker:compute\n",
        encoding="utf-8",
    )
    license_directory = distribution_metadata_root / "licenses"
    license_directory.mkdir()
    license_path = license_directory / "LICENSE"
    license_path.write_text("Example license text\n", encoding="utf-8")
    if include_record:
        record_path = distribution_metadata_root / "RECORD"
        recorded_files = (
            package_root / "__init__.py",
            package_root / "__main__.py",
            package_root / "worker.py",
            nested_package_root / "__init__.py",
            nested_package_root / "labels.py",
            package_root / "py.typed",
            package_root / "worker.pyc",
            distribution_metadata_root / "METADATA",
            distribution_metadata_root / "entry_points.txt",
            license_path,
        )
        recorded_rows = {
            relative_path: hash_and_size
            for recorded_file in recorded_files
            for relative_path, hash_and_size in (
                record_values_for_file(recorded_file, site_packages_directory),
            )
        }
        cache_tag = sys.implementation.cache_tag
        assert cache_tag is not None
        recorded_rows.update(
            {
                f"sample_native_package/__pycache__/__init__.{cache_tag}.pyc": ("", ""),
                f"sample_native_package/__pycache__/worker.{cache_tag}.pyc": ("", ""),
                f"sample_native_package/nested/__pycache__/labels.{cache_tag}.pyc": ("", ""),
            }
        )
        recorded_rows[record_path.relative_to(site_packages_directory).as_posix()] = ("", "")
        write_record(record_path, recorded_rows)
    return package_root, license_path


def record_values_for_file(file_path: Path, installation_root: Path) -> tuple[str, tuple[str, str]]:
    contents = file_path.read_bytes()
    encoded_digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=")
    return (
        file_path.relative_to(installation_root).as_posix(),
        ("sha256=" + encoded_digest.decode("ascii"), str(len(contents))),
    )


def write_record(record_path: Path, rows: dict[str, tuple[str, str]]) -> None:
    serialized_record = io.StringIO(newline="")
    writer = csv.writer(serialized_record, lineterminator="\n")
    for path, (hash_value, size_value) in sorted(rows.items()):
        writer.writerow((path, hash_value, size_value))
    record_path.write_text(serialized_record.getvalue(), encoding="utf-8")


def example_record_path(package_root: Path) -> Path:
    return package_root.parent / "sample_native_package-7.2.dist-info" / "RECORD"


def build_example_overlay(
    package_root: Path, overlay_directory: Path, *, worker_only: bool = True
) -> CythonOverlayManifest:
    """Build the example distribution's overlay.

    ``worker_only`` compiles just ``sample_native_package.worker``; otherwise
    every eligible module is compiled, which gives the manifest more than one
    artifact.
    """
    config = (
        CythonOverlayBuildConfig(package_root=package_root, module_names=(WORKER_MODULE_NAME,))
        if worker_only
        else CythonOverlayBuildConfig(package_root=package_root)
    )
    return build_cython_overlay(config, overlay_directory)


def read_manifest_payload(manifest_path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))


def write_manifest_bytes(manifest_path: Path, contents: bytes) -> None:
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(contents)


def write_manifest_payload(manifest_path: Path, payload: dict[str, object]) -> None:
    write_manifest_bytes(
        manifest_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def assert_overlay_left_uninstalled(
    package_root: Path, manifest: CythonOverlayManifest, original_record: bytes
) -> None:
    """A refused apply left every source, artifact slot, and the RECORD untouched."""
    assert all((package_root / artifact.source_path).is_file() for artifact in manifest.artifacts)
    assert not any(
        (package_root / artifact.installed_artifact_path).exists()
        for artifact in manifest.artifacts
    )
    assert not (package_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME).exists()
    assert example_record_path(package_root).read_bytes() == original_record
