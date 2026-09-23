"""Regression coverage for native overlay manifest parse-stage refusals."""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from packaging_test_helpers import (
    assert_overlay_left_uninstalled,
    build_example_overlay,
    example_record_path,
    read_manifest_payload,
    write_example_distribution,
    write_manifest_bytes,
    write_manifest_payload,
)

import hflow.packaging as packaging
from hflow.packaging import (
    CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    MAX_NATIVE_OVERLAY_MANIFEST_BYTES,
    CythonOverlayApplyError,
    CythonOverlayManifest,
    CythonOverlayManifestError,
    CythonOverlayVerificationCode,
    apply_cython_overlay,
    verify_cython_overlay,
)


def _build_overlay(tmp_path: Path) -> tuple[Path, Path, CythonOverlayManifest, bytes, bytes]:
    package_root, _ = write_example_distribution(tmp_path)
    source_bytes = (package_root / "worker.py").read_bytes()
    original_record = example_record_path(package_root).read_bytes()
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_example_overlay(package_root, overlay_directory)
    return package_root, overlay_directory, manifest, source_bytes, original_record


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
    assert_overlay_left_uninstalled(package_root, manifest, original_record)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("schema-version", "unsupported native overlay schema version"),
        ("format", "unsupported native overlay format"),
        ("empty-artifacts", "artifacts must not be empty"),
        ("unsorted-artifacts", "artifacts must be sorted by module_name"),
        ("duplicate-module", "artifact module names must be unique"),
    ],
)
def test_apply_refuses_invalid_manifest_before_mutation(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    # Every eligible module, so the unsorted and duplicate cases have two
    # artifacts to disorder.
    manifest = build_example_overlay(package_root, overlay_directory, worker_only=False)
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = read_manifest_payload(manifest_path)
    artifacts = cast(list[dict[str, object]], payload["artifacts"])
    if mutation == "schema-version":
        assert payload["schema_version"] == packaging.CYTHON_OVERLAY_SCHEMA_VERSION
        payload["schema_version"] = packaging.CYTHON_OVERLAY_SCHEMA_VERSION + 1
    elif mutation == "format":
        payload["format"] = "unsupported-native-overlay"
    elif mutation == "empty-artifacts":
        payload["artifacts"] = []
    elif mutation == "unsorted-artifacts":
        artifacts.reverse()
    elif mutation == "duplicate-module":
        artifacts[1]["module_name"] = artifacts[0]["module_name"]
    else:
        raise AssertionError(f"unknown manifest mutation: {mutation}")
    write_manifest_payload(manifest_path, payload)
    original_record = example_record_path(package_root).read_bytes()

    with pytest.raises(CythonOverlayManifestError, match=expected_message):
        apply_cython_overlay(overlay_directory, package_root)

    assert_overlay_left_uninstalled(package_root, manifest, original_record)


def test_schema_version_is_bound_into_the_bundle_digest(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_example_overlay(package_root, overlay_directory, worker_only=False)
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = read_manifest_payload(manifest_path)
    assert payload["schema_version"] == packaging.CYTHON_OVERLAY_SCHEMA_VERSION
    digest_payload = {
        key: payload[key] for key in ("format", "package_name", "target", "toolchain", "artifacts")
    }
    canonical_bytes = json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["bundle_digest"] = "sha256:" + hashlib.sha256(canonical_bytes).hexdigest()
    write_manifest_payload(manifest_path, payload)
    original_record = example_record_path(package_root).read_bytes()

    with pytest.raises(
        CythonOverlayManifestError,
        match="bundle_digest does not match manifest components",
    ):
        apply_cython_overlay(overlay_directory, package_root)

    assert_overlay_left_uninstalled(package_root, manifest, original_record)


@pytest.mark.parametrize("oversized", [False, True], ids=["invalid-json", "oversized"])
def test_apply_refuses_invalid_manifest_json_before_mutation(
    tmp_path: Path, oversized: bool
) -> None:
    package_root, overlay_directory, manifest, source_bytes, original_record = _build_overlay(
        tmp_path
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    write_manifest_bytes(
        manifest_path,
        b" " * (MAX_NATIVE_OVERLAY_MANIFEST_BYTES + 1) if oversized else b'{"schema_version":',
    )

    _assert_apply_refused_without_mutation(
        package_root,
        overlay_directory,
        manifest,
        source_bytes,
        original_record,
        "native overlay manifest exceeds its byte limit"
        if oversized
        else "native overlay manifest is not valid JSON",
    )


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "expected_message"),
    [
        (("artifacts",), {}, "artifacts: Input should be a valid array"),
        (("schema_version",), True, "schema_version: Input should be a valid integer"),
        (
            ("target", "python_version"),
            314,
            "target.python_version: Input should be a valid string",
        ),
        (("target", "unexpected"), "value", "target.unexpected: Unexpected keyword argument"),
        (("toolchain",), {}, "toolchain.cython_version: Field required"),
        (
            ("artifacts", 0, "artifact_size_bytes"),
            "12",
            "artifact_size_bytes: Input should be a valid integer",
        ),
    ],
)
def test_apply_refuses_invalid_manifest_structure_before_mutation(
    tmp_path: Path,
    field_path: tuple[str | int, ...],
    invalid_value: object,
    expected_message: str,
) -> None:
    package_root, overlay_directory, manifest, source_bytes, original_record = _build_overlay(
        tmp_path
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = read_manifest_payload(manifest_path)
    container: object = payload
    for component in field_path[:-1]:
        container = (
            cast(list[object], container)[component]
            if isinstance(component, int)
            else cast(dict[str, object], container)[component]
        )
    cast(dict[str, object], container)[str(field_path[-1])] = invalid_value
    write_manifest_payload(manifest_path, payload)

    _assert_apply_refused_without_mutation(
        package_root,
        overlay_directory,
        manifest,
        source_bytes,
        original_record,
        expected_message,
    )


def test_apply_refuses_semantically_identical_noncanonical_manifest_before_mutation(
    tmp_path: Path,
) -> None:
    package_root, overlay_directory, manifest, source_bytes, original_record = _build_overlay(
        tmp_path
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = read_manifest_payload(manifest_path)
    noncanonical_bytes = (json.dumps(payload, indent=4, sort_keys=True) + "\n").encode("utf-8")
    assert json.loads(noncanonical_bytes) == payload
    assert noncanonical_bytes != manifest_path.read_bytes()
    write_manifest_bytes(manifest_path, noncanonical_bytes)

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
    payload = read_manifest_payload(manifest_path)
    serialized_manifest = manifest_path.read_text(encoding="utf-8").rstrip("\n")
    assert serialized_manifest.endswith("}")
    duplicated_manifest = (
        serialized_manifest[:-1] + f',\n  "format": {json.dumps(payload["format"])}\n}}\n'
    ).encode("utf-8")
    assert json.loads(duplicated_manifest)["format"] == payload["format"]
    write_manifest_bytes(manifest_path, duplicated_manifest)

    _assert_apply_refused_without_mutation(
        package_root,
        overlay_directory,
        manifest,
        source_bytes,
        original_record,
        "native overlay manifest contains duplicate fields",
    )


@pytest.mark.parametrize("oversized", [False, True], ids=["mismatch", "oversized"])
def test_invalid_installed_manifest_reports_mismatch_without_mutating_package(
    tmp_path: Path, oversized: bool
) -> None:
    package_root, overlay_directory, manifest, _, _ = _build_overlay(tmp_path)
    apply_cython_overlay(overlay_directory, package_root)
    installed_manifest_path = package_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME
    invalid_manifest = b" " * (MAX_NATIVE_OVERLAY_MANIFEST_BYTES + 1) if oversized else b"{}"
    write_manifest_bytes(installed_manifest_path, invalid_manifest)
    preserved_files = {
        path: path.read_bytes()
        for path in (
            installed_manifest_path,
            example_record_path(package_root),
            *(package_root / artifact.installed_artifact_path for artifact in manifest.artifacts),
        )
    }

    verification = verify_cython_overlay(overlay_directory, target_package_root=package_root)
    assert CythonOverlayVerificationCode.INSTALLED_MANIFEST_MISMATCH in {
        issue.code for issue in verification.issues
    }
    with pytest.raises(CythonOverlayApplyError, match="different native overlay manifest"):
        apply_cython_overlay(overlay_directory, package_root)
    for path, expected_bytes in preserved_files.items():
        assert path.read_bytes() == expected_bytes


def test_manifest_byte_limit_includes_the_boundary_and_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, overlay_directory, manifest, _, _ = _build_overlay(tmp_path)
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    original_bytes = manifest_path.read_bytes()
    manifest_path.chmod(0o644)
    monkeypatch.setattr(packaging, "MAX_NATIVE_OVERLAY_MANIFEST_BYTES", len(original_bytes))
    packaging.write_cython_overlay_manifest(manifest, manifest_path)
    assert packaging.load_cython_overlay_manifest(manifest_path) == manifest

    monkeypatch.setattr(packaging, "MAX_NATIVE_OVERLAY_MANIFEST_BYTES", len(original_bytes) - 1)
    with pytest.raises(CythonOverlayManifestError, match="exceeds its byte limit"):
        packaging.write_cython_overlay_manifest(manifest, manifest_path)
    assert manifest_path.read_bytes() == original_bytes
