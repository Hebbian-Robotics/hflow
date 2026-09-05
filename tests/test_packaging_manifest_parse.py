"""Parse-stage regression coverage for native overlay manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hflow.packaging import (
    CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    CythonOverlayBuildConfig,
    CythonOverlayManifestError,
    apply_cython_overlay,
    build_cython_overlay,
)


def _build_overlay(tmp_path: Path):
    package_root = tmp_path / "sample_native_package"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("from .worker import compute\n", encoding="utf-8")
    (package_root / "worker.py").write_text(
        "def compute(value: int) -> int:\n    return value * 3\n",
        encoding="utf-8",
    )
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    return package_root, overlay_directory, manifest


def _assert_not_applied(package_root: Path, manifest: object) -> None:
    artifacts = getattr(manifest, "artifacts")
    assert all((package_root / artifact.source_path).is_file() for artifact in artifacts)
    assert not any((package_root / artifact.installed_artifact_path).exists() for artifact in artifacts)
    assert not (package_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME).exists()


def test_apply_refuses_manifest_that_is_not_valid_json(tmp_path: Path) -> None:
    package_root, overlay_directory, manifest = _build_overlay(tmp_path)
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(b'{"schema_version":')

    with pytest.raises(
        CythonOverlayManifestError,
        match="^native overlay manifest is not valid JSON$",
    ):
        apply_cython_overlay(overlay_directory, package_root)

    _assert_not_applied(package_root, manifest)


def test_apply_refuses_artifacts_that_are_not_an_array(tmp_path: Path) -> None:
    package_root, overlay_directory, manifest = _build_overlay(tmp_path)
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"] = {}
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(CythonOverlayManifestError, match="^artifacts must be an array$"):
        apply_cython_overlay(overlay_directory, package_root)

    _assert_not_applied(package_root, manifest)


def test_apply_refuses_semantically_identical_noncanonical_json(tmp_path: Path) -> None:
    package_root, overlay_directory, manifest = _build_overlay(tmp_path)
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Valid and semantically identical, but deliberately not the canonical
    # serialization written by the overlay builder.
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(
        CythonOverlayManifestError,
        match="^native overlay manifest is not canonical JSON$",
    ):
        apply_cython_overlay(overlay_directory, package_root)

    _assert_not_applied(package_root, manifest)


def test_apply_refuses_duplicate_manifest_fields(tmp_path: Path) -> None:
    package_root, overlay_directory, manifest = _build_overlay(tmp_path)
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    canonical = manifest_path.read_text(encoding="utf-8")
    marker = '  "format": "cython-extension-overlay",\n'
    assert canonical.count(marker) == 1
    duplicate = canonical.replace(marker, marker + marker, 1)
    manifest_path.chmod(0o644)
    manifest_path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(
        CythonOverlayManifestError,
        match="^native overlay manifest contains duplicate fields$",
    ):
        apply_cython_overlay(overlay_directory, package_root)

    _assert_not_applied(package_root, manifest)
