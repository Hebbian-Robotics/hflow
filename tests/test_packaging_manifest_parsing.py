"""Regression coverage for native overlay manifest parse-stage refusals."""

import json
from pathlib import Path
from typing import cast

import pytest
from packaging_test_helpers import example_record_path, write_example_distribution

from hflow.packaging import (
    CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    CythonOverlayBuildConfig,
    CythonOverlayManifest,
    CythonOverlayManifestError,
    apply_cython_overlay,
    build_cython_overlay,
)


def _build_overlay(tmp_path: Path) -> tuple[Path, Path, CythonOverlayManifest, bytes, bytes]:
    package_root, _ = write_example_distribution(tmp_path)
    source_path = package_root / "worker.py"
    source_bytes = source_path.read_bytes()
    original_record = example_record_path(package_root).read_bytes()
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
    assert example_record_path(package_root).read_bytes() == original_record


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
    payload = _manifest_payload(manifest_path)
    container: object = payload
    for component in field_path[:-1]:
        container = (
            cast(list[object], container)[component]
            if isinstance(component, int)
            else cast(dict[str, object], container)[component]
        )
    cast(dict[str, object], container)[str(field_path[-1])] = invalid_value
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
        expected_message,
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
