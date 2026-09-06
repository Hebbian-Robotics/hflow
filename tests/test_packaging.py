"""Behavioral coverage for target-bound native package overlays."""

import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from packaging_test_helpers import (
    example_record_path,
    record_values_for_file,
    write_example_distribution,
    write_record,
)

import hflow.packaging as packaging
from hflow.cli import main
from hflow.packaging import (
    CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
    CythonOverlayApplyError,
    CythonOverlayBuildConfig,
    CythonOverlayManifestError,
    CythonOverlayVerificationCode,
    CythonOverlayVerificationIssue,
    apply_cython_overlay,
    build_cython_overlay,
    current_native_build_target,
    load_cython_overlay_manifest,
    verify_cython_overlay,
)


def _read_record(record_path: Path) -> dict[str, tuple[str, str]]:
    with record_path.open(encoding="utf-8", newline="") as record_file:
        return {
            path: (hash_value, size_value)
            for path, hash_value, size_value in csv.reader(record_file, strict=True)
        }


def _read_manifest_payload(manifest_path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))


def _write_manifest_payload(manifest_path: Path, payload: dict[str, object]) -> None:
    manifest_path.chmod(0o644)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_example_package(site_packages_directory: Path) -> subprocess.CompletedProcess[str]:
    process_environment = os.environ.copy()
    process_environment["PYTHONPATH"] = str(site_packages_directory)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pickle; from importlib.metadata import version; "
                "from sample_native_package import compute; "
                "from sample_native_package.nested import label; "
                "from sample_native_package.worker import Token; "
                "print(compute(4), label(), version('sample-native-package'), "
                "pickle.loads(pickle.dumps(Token)) is Token)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=process_environment,
        cwd=site_packages_directory,
    )


def test_native_overlay_replaces_only_implementation_sources_and_preserves_distribution(
    tmp_path: Path,
) -> None:
    package_root, license_path = write_example_distribution(tmp_path)
    initial_result = _run_example_package(package_root.parent)
    assert initial_result.returncode == 0
    assert initial_result.stdout.strip() == "12 native-result 7.2 True"

    preserved_files = {
        path: path.read_bytes()
        for path in (
            package_root / "__init__.py",
            package_root / "__main__.py",
            package_root / "nested" / "__init__.py",
            package_root / "py.typed",
            package_root.parent / "sample_native_package-7.2.dist-info" / "METADATA",
            package_root.parent / "sample_native_package-7.2.dist-info" / "entry_points.txt",
            license_path,
        )
    }
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(package_root=package_root),
        overlay_directory,
    )

    assert [artifact.module_name for artifact in manifest.artifacts] == [
        "sample_native_package.nested.labels",
        "sample_native_package.worker",
    ]
    assert (
        load_cython_overlay_manifest(overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME)
        == manifest
    )
    assert verify_cython_overlay(overlay_directory).succeeded

    apply_cython_overlay(overlay_directory, package_root)

    assert verify_cython_overlay(
        overlay_directory,
        target_package_root=package_root,
    ).succeeded
    assert not (package_root / "worker.py").exists()
    assert not (package_root / "nested" / "labels.py").exists()
    assert not list(package_root.rglob("worker.*.pyc"))
    assert not list(package_root.rglob("labels.*.pyc"))
    assert not (package_root / "worker.pyc").exists()
    for preserved_path, preserved_bytes in preserved_files.items():
        assert preserved_path.read_bytes() == preserved_bytes

    record_rows = _read_record(example_record_path(package_root))
    assert "sample_native_package/worker.py" not in record_rows
    assert "sample_native_package/worker.pyc" not in record_rows
    assert "sample_native_package/nested/labels.py" not in record_rows
    assert not any(
        path.startswith("sample_native_package/__pycache__/worker.")
        or path.startswith("sample_native_package/nested/__pycache__/labels.")
        for path in record_rows
    )
    assert "sample_native_package/__init__.py" in record_rows
    assert "sample_native_package/py.typed" in record_rows
    for artifact in manifest.artifacts:
        installed_artifact_path = f"sample_native_package/{artifact.installed_artifact_path}"
        assert (
            record_rows[installed_artifact_path]
            == record_values_for_file(
                package_root / artifact.installed_artifact_path,
                package_root.parent,
            )[1]
        )
    installed_manifest_path = package_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME
    assert (
        record_rows[f"sample_native_package/{INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME}"]
        == record_values_for_file(installed_manifest_path, package_root.parent)[1]
    )

    native_result = _run_example_package(package_root.parent)
    assert native_result.returncode == 0
    assert native_result.stdout.strip() == "12 native-result 7.2 True"

    finalized_record = example_record_path(package_root).read_bytes()
    apply_cython_overlay(overlay_directory, package_root)
    assert example_record_path(package_root).read_bytes() == finalized_record


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
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(package_root=package_root),
        overlay_directory,
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = _read_manifest_payload(manifest_path)
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
    _write_manifest_payload(manifest_path, payload)
    original_record = example_record_path(package_root).read_bytes()

    with pytest.raises(CythonOverlayManifestError, match=expected_message):
        apply_cython_overlay(overlay_directory, package_root)

    assert all((package_root / artifact.source_path).is_file() for artifact in manifest.artifacts)
    assert not any(
        (package_root / artifact.installed_artifact_path).exists()
        for artifact in manifest.artifacts
    )
    assert not (package_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME).exists()
    assert example_record_path(package_root).read_bytes() == original_record


def test_schema_version_is_bound_into_the_bundle_digest(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(package_root=package_root),
        overlay_directory,
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    payload = _read_manifest_payload(manifest_path)
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
    _write_manifest_payload(manifest_path, payload)
    original_record = example_record_path(package_root).read_bytes()

    with pytest.raises(
        CythonOverlayManifestError,
        match="bundle_digest does not match manifest components",
    ):
        apply_cython_overlay(overlay_directory, package_root)

    assert all((package_root / artifact.source_path).is_file() for artifact in manifest.artifacts)
    assert not any(
        (package_root / artifact.installed_artifact_path).exists()
        for artifact in manifest.artifacts
    )
    assert not (package_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME).exists()
    assert example_record_path(package_root).read_bytes() == original_record


def test_apply_refuses_a_changed_source_before_installing_any_artifact(
    tmp_path: Path,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    source_path = package_root / "worker.py"
    source_path.write_text("def compute(value: int) -> int:\n    return value * 99\n")

    with pytest.raises(CythonOverlayApplyError, match="does not match the build receipt"):
        apply_cython_overlay(overlay_directory, package_root)

    assert source_path.is_file()
    assert not (package_root / manifest.artifacts[0].installed_artifact_path).exists()


def test_apply_never_follows_a_source_symlink(
    tmp_path: Path,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    source_path = package_root / "worker.py"
    external_source_path = tmp_path / "external.py"
    external_source_path.write_bytes(source_path.read_bytes())
    source_path.unlink()
    source_path.symlink_to(external_source_path)

    with pytest.raises(CythonOverlayApplyError, match="target source is not a regular file"):
        apply_cython_overlay(overlay_directory, package_root)

    assert external_source_path.is_file()
    assert source_path.is_symlink()
    assert not (package_root / manifest.artifacts[0].installed_artifact_path).exists()


def test_apply_resumes_from_a_prepared_wheel_record(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    record_path = example_record_path(package_root)
    original_record_rows = _read_record(record_path)
    overlay_directory = tmp_path / "native-overlay"
    build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    apply_cython_overlay(overlay_directory, package_root)
    finalized_record_bytes = record_path.read_bytes()

    prepared_record_rows = _read_record(record_path)
    for path, values in original_record_rows.items():
        if path in {
            "sample_native_package/worker.py",
            "sample_native_package/worker.pyc",
        } or path.startswith("sample_native_package/__pycache__/worker."):
            prepared_record_rows[path] = values
    write_record(record_path, prepared_record_rows)

    apply_cython_overlay(overlay_directory, package_root)

    assert record_path.read_bytes() == finalized_record_bytes
    assert verify_cython_overlay(
        overlay_directory,
        target_package_root=package_root,
    ).succeeded


def test_apply_supports_a_source_checkout_without_a_wheel_record(tmp_path: Path) -> None:
    package_root, license_path = write_example_distribution(tmp_path, include_record=False)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )

    apply_cython_overlay(overlay_directory, package_root)

    assert not (package_root / "worker.py").exists()
    assert (package_root / manifest.artifacts[0].installed_artifact_path).is_file()
    assert license_path.read_text(encoding="utf-8") == "Example license text\n"
    assert not example_record_path(package_root).exists()


def test_apply_accepts_spec_valid_blank_alternate_and_absolute_source_rows(
    tmp_path: Path,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    record_path = example_record_path(package_root)
    record_rows = _read_record(record_path)
    record_rows["sample_native_package/worker.py"] = ("", "")
    nested_source_path = package_root / "nested" / "labels.py"
    nested_contents = nested_source_path.read_bytes()
    sha512_digest = base64.urlsafe_b64encode(hashlib.sha512(nested_contents).digest()).rstrip(b"=")
    record_rows.pop("sample_native_package/nested/labels.py")
    record_rows[str(nested_source_path)] = (
        "sha512=" + sha512_digest.decode("ascii"),
        str(len(nested_contents)),
    )
    write_record(record_path, record_rows)
    overlay_directory = tmp_path / "native-overlay"
    build_cython_overlay(
        CythonOverlayBuildConfig(package_root=package_root),
        overlay_directory,
    )

    apply_cython_overlay(overlay_directory, package_root)

    finalized_record_rows = _read_record(record_path)
    assert "sample_native_package/worker.py" not in finalized_record_rows
    assert str(nested_source_path) not in finalized_record_rows
    assert verify_cython_overlay(
        overlay_directory,
        target_package_root=package_root,
    ).succeeded


def test_apply_refuses_an_unowned_package_among_installed_distributions(
    tmp_path: Path,
) -> None:
    package_root, _ = write_example_distribution(tmp_path, include_record=False)
    unrelated_metadata_root = package_root.parent / "unrelated-1.0.dist-info"
    unrelated_metadata_root.mkdir()
    unrelated_record_path = unrelated_metadata_root / "RECORD"
    write_record(
        unrelated_record_path,
        {"unrelated-1.0.dist-info/RECORD": ("", "")},
    )
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )

    with pytest.raises(CythonOverlayApplyError, match="no wheel RECORD"):
        apply_cython_overlay(overlay_directory, package_root)

    assert (package_root / "worker.py").is_file()
    assert not (package_root / manifest.artifacts[0].installed_artifact_path).exists()


def test_apply_refuses_an_unrecorded_native_artifact_collision(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    artifact = manifest.artifacts[0]
    destination_path = package_root / artifact.installed_artifact_path
    destination_path.write_bytes((overlay_directory / artifact.artifact_path).read_bytes())
    original_record_bytes = example_record_path(package_root).read_bytes()

    with pytest.raises(CythonOverlayApplyError, match="conflicts with an existing native artifact"):
        apply_cython_overlay(overlay_directory, package_root)

    assert (package_root / "worker.py").is_file()
    assert destination_path.is_file()
    assert example_record_path(package_root).read_bytes() == original_record_bytes


def test_apply_refuses_ambiguous_wheel_record_ownership(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    original_record_path = example_record_path(package_root)
    duplicate_metadata_root = package_root.parent / "duplicate-1.0.dist-info"
    duplicate_metadata_root.mkdir()
    duplicate_record_path = duplicate_metadata_root / "RECORD"
    duplicate_rows = _read_record(original_record_path)
    duplicate_rows.pop("sample_native_package-7.2.dist-info/RECORD")
    duplicate_rows["duplicate-1.0.dist-info/RECORD"] = ("", "")
    write_record(duplicate_record_path, duplicate_rows)
    original_record_bytes = original_record_path.read_bytes()

    with pytest.raises(CythonOverlayApplyError, match="ambiguous wheel RECORD ownership"):
        apply_cython_overlay(overlay_directory, package_root)

    assert (package_root / "worker.py").is_file()
    assert not (package_root / manifest.artifacts[0].installed_artifact_path).exists()
    assert original_record_path.read_bytes() == original_record_bytes


def test_apply_refuses_a_malformed_wheel_record_before_mutation(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    record_path = example_record_path(package_root)
    record_path.write_text("not,three-fields\n", encoding="utf-8")
    malformed_record_bytes = record_path.read_bytes()

    with pytest.raises(CythonOverlayApplyError, match="three fields"):
        apply_cython_overlay(overlay_directory, package_root)

    assert (package_root / "worker.py").is_file()
    assert not (package_root / manifest.artifacts[0].installed_artifact_path).exists()
    assert record_path.read_bytes() == malformed_record_bytes


@pytest.mark.parametrize("signature_name", ["RECORD.jws", "RECORD.p7s"])
def test_apply_refuses_a_signed_wheel_record_before_mutation(
    tmp_path: Path,
    signature_name: str,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    record_path = example_record_path(package_root)
    (record_path.parent / signature_name).write_text("signed", encoding="utf-8")
    original_record_bytes = record_path.read_bytes()

    with pytest.raises(CythonOverlayApplyError, match="cannot update a signed wheel RECORD"):
        apply_cython_overlay(overlay_directory, package_root)

    assert (package_root / "worker.py").is_file()
    assert not (package_root / manifest.artifacts[0].installed_artifact_path).exists()
    assert record_path.read_bytes() == original_record_bytes


def test_verification_reports_artifact_tampering_and_target_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    artifact_path = overlay_directory / manifest.artifacts[0].artifact_path
    artifact_path.chmod(0o644)
    artifact_path.write_bytes(b"changed")

    tampered_outcome = verify_cython_overlay(overlay_directory)
    assert {issue.code for issue in tampered_outcome.issues} == {
        CythonOverlayVerificationCode.ARTIFACT_HASH_MISMATCH,
        CythonOverlayVerificationCode.ARTIFACT_SIZE_MISMATCH,
    }

    artifact_path.write_bytes(b"changed again")
    mismatched_target = replace(current_native_build_target(), machine="different-machine")
    monkeypatch.setattr(
        "hflow.packaging.current_native_build_target",
        lambda: mismatched_target,
    )
    target_outcome = verify_cython_overlay(overlay_directory)
    assert CythonOverlayVerificationCode.MACHINE_MISMATCH in {
        issue.code for issue in target_outcome.issues
    }


def test_verification_reports_a_final_wheel_record_that_lost_native_ownership(
    tmp_path: Path,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    apply_cython_overlay(overlay_directory, package_root)
    record_path = example_record_path(package_root)
    record_rows = _read_record(record_path)
    record_rows.pop(f"sample_native_package/{manifest.artifacts[0].installed_artifact_path}")
    write_record(record_path, record_rows)

    verification_outcome = verify_cython_overlay(
        overlay_directory,
        target_package_root=package_root,
    )

    assert (
        CythonOverlayVerificationIssue(
            CythonOverlayVerificationCode.INSTALLED_RECORD_MISMATCH,
        )
        in verification_outcome.issues
    )


def test_build_is_reproducible_across_output_directories(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    build_config = CythonOverlayBuildConfig(
        package_root=package_root,
        module_names=("sample_native_package.worker",),
    )

    first_overlay_directory = tmp_path / "first-overlay"
    second_overlay_directory = tmp_path / "second-overlay"
    first_manifest = build_cython_overlay(build_config, first_overlay_directory)
    second_manifest = build_cython_overlay(build_config, second_overlay_directory)

    assert first_manifest == second_manifest
    assert (first_overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME).read_bytes() == (
        second_overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    ).read_bytes()
    assert (first_overlay_directory / first_manifest.artifacts[0].artifact_path).read_bytes() == (
        second_overlay_directory / second_manifest.artifacts[0].artifact_path
    ).read_bytes()


def test_verification_rejects_an_unrecorded_directory(tmp_path: Path) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    (overlay_directory / "unrecorded").mkdir()

    verification_outcome = verify_cython_overlay(overlay_directory)

    assert (
        CythonOverlayVerificationIssue(
            CythonOverlayVerificationCode.UNEXPECTED_OVERLAY_PATH,
            "unrecorded",
        )
        in verification_outcome.issues
    )


def test_manifest_is_canonical_json_and_cli_verifies_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    overlay_directory = tmp_path / "native-overlay"
    manifest = build_cython_overlay(
        CythonOverlayBuildConfig(
            package_root=package_root,
            module_names=("sample_native_package.worker",),
        ),
        overlay_directory,
    )
    manifest_path = overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME
    assert (
        manifest_path.read_bytes()
        == (json.dumps(manifest.to_json_value(), indent=2, sort_keys=True) + "\n").encode()
    )

    exit_code = main(["package", "verify", str(overlay_directory)])

    assert exit_code == 0
    assert "native overlay verified" in capsys.readouterr().out


def test_distribution_version_fixture_is_visible_to_the_current_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, _ = write_example_distribution(tmp_path)
    monkeypatch.syspath_prepend(str(package_root.parent))
    assert importlib.metadata.version("sample-native-package") == "7.2"
