"""Build and verify target-bound Cython overlays for runtime deployments.

An overlay is deliberately not a second Python distribution.  It is applied
on top of an exact, normally installed wheel, replacing selected implementation
modules with native extensions while leaving package initializers, console
entry points, version metadata, type markers, and license files owned by that
wheel.  This keeps packaging semantics ordinary and makes the native boundary
explicit: an overlay is usable only by its recorded CPython ABI and platform.
The applied tree is a runtime artifact, not a wheel, typing input, or artifact
that should be uploaded to a Python package index.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import keyword
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, TypedDict, cast

CYTHON_OVERLAY_MANIFEST_FILE_NAME = "hflow-native-overlay.json"
INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME = ".hflow-native-overlay.json"
CYTHON_OVERLAY_SCHEMA_VERSION = 1
CYTHON_OVERLAY_FORMAT = "cython-extension-overlay"
_ARTIFACT_DIRECTORY_NAME = "artifacts"
_SHA256_HEX_LENGTH = 64
_FILE_READ_CHUNK_SIZE_BYTES = 1024 * 1024


class NativeBuildTargetJson(TypedDict):
    python_implementation: str
    python_version: str
    python_abi_tag: str
    extension_suffix: str
    platform_tag: str
    operating_system: str
    machine: str


class CythonToolchainJson(TypedDict):
    cython_version: str
    setuptools_version: str


class CythonOverlayArtifactJson(TypedDict):
    module_name: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int


class CythonOverlayManifestJson(TypedDict):
    schema_version: int
    format: str
    package_name: str
    bundle_digest: str
    target: NativeBuildTargetJson
    toolchain: CythonToolchainJson
    artifacts: list[CythonOverlayArtifactJson]


@dataclass(frozen=True, slots=True)
class NativeBuildTarget:
    """Interpreter and platform facts required to load an extension overlay."""

    python_implementation: str
    python_version: str
    python_abi_tag: str
    extension_suffix: str
    platform_tag: str
    operating_system: str
    machine: str

    def to_json_value(self) -> NativeBuildTargetJson:
        return {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "python_abi_tag": self.python_abi_tag,
            "extension_suffix": self.extension_suffix,
            "platform_tag": self.platform_tag,
            "operating_system": self.operating_system,
            "machine": self.machine,
        }


@dataclass(frozen=True, slots=True)
class CythonToolchain:
    """Build-tool versions that produced the native artifacts."""

    cython_version: str
    setuptools_version: str

    def to_json_value(self) -> CythonToolchainJson:
        return {
            "cython_version": self.cython_version,
            "setuptools_version": self.setuptools_version,
        }


@dataclass(frozen=True, slots=True)
class CythonOverlayArtifact:
    """One source module and the native artifact that can replace it."""

    module_name: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int

    def to_json_value(self) -> CythonOverlayArtifactJson:
        return {
            "module_name": self.module_name,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
        }

    @property
    def installed_artifact_path(self) -> str:
        """Artifact path relative to the package root after applying the overlay."""

        return PurePosixPath(self.artifact_path).relative_to(_ARTIFACT_DIRECTORY_NAME).as_posix()


@dataclass(frozen=True, slots=True)
class CythonOverlayManifest:
    """Deterministic integrity and compatibility contract for one overlay."""

    schema_version: int
    format: str
    package_name: str
    bundle_digest: str
    target: NativeBuildTarget
    toolchain: CythonToolchain
    artifacts: tuple[CythonOverlayArtifact, ...]

    def to_json_value(self) -> CythonOverlayManifestJson:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "package_name": self.package_name,
            "bundle_digest": self.bundle_digest,
            "target": self.target.to_json_value(),
            "toolchain": self.toolchain.to_json_value(),
            "artifacts": [artifact.to_json_value() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class CythonOverlayBuildConfig:
    """Inputs for compiling modules below one installed or source package root.

    ``module_names=None`` selects every Python module except package
    ``__init__`` files and ``__main__.py`` adapters.  Explicit module names are
    fully qualified (for example, ``hflow.checks``).
    """

    package_root: Path
    package_name: str | None = None
    module_names: tuple[str, ...] | None = None


class CythonOverlayManifestError(ValueError):
    """Raised when an overlay manifest is malformed or self-inconsistent."""


class CythonOverlayBuildError(RuntimeError):
    """Raised when an extension overlay cannot be built safely."""


class CythonOverlayApplyError(RuntimeError):
    """Raised before or during an overlay application that cannot finish safely."""


class CythonOverlayVerificationCode(StrEnum):
    """Machine-readable verification findings."""

    PYTHON_IMPLEMENTATION_MISMATCH = "python_implementation_mismatch"
    PYTHON_VERSION_MISMATCH = "python_version_mismatch"
    PYTHON_ABI_MISMATCH = "python_abi_mismatch"
    EXTENSION_SUFFIX_MISMATCH = "extension_suffix_mismatch"
    PLATFORM_MISMATCH = "platform_mismatch"
    OPERATING_SYSTEM_MISMATCH = "operating_system_mismatch"
    MACHINE_MISMATCH = "machine_mismatch"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_NOT_REGULAR = "artifact_not_regular"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    ARTIFACT_SIZE_MISMATCH = "artifact_size_mismatch"
    UNEXPECTED_OVERLAY_PATH = "unexpected_overlay_path"
    INSTALLED_MANIFEST_MISSING = "installed_manifest_missing"
    INSTALLED_MANIFEST_MISMATCH = "installed_manifest_mismatch"
    INSTALLED_ARTIFACT_MISSING = "installed_artifact_missing"
    INSTALLED_ARTIFACT_HASH_MISMATCH = "installed_artifact_hash_mismatch"
    INSTALLED_RECORD_MISMATCH = "installed_record_mismatch"
    SOURCE_STILL_PRESENT = "source_still_present"


@dataclass(frozen=True, slots=True)
class CythonOverlayVerificationIssue:
    code: CythonOverlayVerificationCode
    path: str | None = None


@dataclass(frozen=True, slots=True)
class CythonOverlayVerificationOutcome:
    issues: tuple[CythonOverlayVerificationIssue, ...]

    @property
    def succeeded(self) -> bool:
        return not self.issues


class _WheelRecordState(StrEnum):
    ORIGINAL = "original"
    PREPARED = "prepared"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class _WheelRecordRow:
    path: str
    hash_value: str
    size_value: str
    resolved_path: Path


@dataclass(frozen=True, slots=True)
class _WheelRecordUpdate:
    record_path: Path
    original_file_mode: int
    current_bytes: bytes
    prepared_bytes: bytes
    final_bytes: bytes
    state: _WheelRecordState


@dataclass(frozen=True, slots=True)
class _ParsedWheelRecord:
    record_path: Path
    file_mode: int
    serialized_bytes: bytes
    rows: tuple[_WheelRecordRow, ...]


def current_native_build_target() -> NativeBuildTarget:
    """Return the current interpreter's load-bearing extension identity."""

    python_abi_tag = sysconfig.get_config_var("SOABI")
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(python_abi_tag, str) or not python_abi_tag:
        raise RuntimeError("the current interpreter does not expose SOABI")
    if not isinstance(extension_suffix, str) or not extension_suffix:
        raise RuntimeError("the current interpreter does not expose EXT_SUFFIX")
    return NativeBuildTarget(
        python_implementation=sys.implementation.name,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        python_abi_tag=python_abi_tag,
        extension_suffix=extension_suffix,
        platform_tag=sysconfig.get_platform(),
        operating_system=platform.system().lower(),
        machine=platform.machine().lower(),
    )


def build_cython_overlay(
    config: CythonOverlayBuildConfig,
    output_directory: Path | str,
) -> CythonOverlayManifest:
    """Compile a fresh, verified overlay without modifying its source package."""

    package_root, package_name, module_sources = _resolve_build_inputs(config)
    build_target = current_native_build_target()
    if build_target.python_implementation != "cpython" or build_target.operating_system != "linux":
        raise CythonOverlayBuildError(
            "native overlay builds currently require CPython on Linux; "
            f"found {build_target.python_implementation} on {build_target.operating_system}"
        )
    try:
        cython_version = importlib.metadata.version("Cython")
        setuptools_version = importlib.metadata.version("setuptools")
    except importlib.metadata.PackageNotFoundError as error:
        raise CythonOverlayBuildError(
            'native overlay builds need the optional extra: install "hflow[native-build]"'
        ) from error

    output_path = Path(output_directory).expanduser().absolute()
    package_path = package_root.absolute()
    if _paths_overlap(output_path, package_path):
        raise CythonOverlayBuildError("output_directory must be outside package_root")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"native overlay output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_unsymlinked_directory(output_path.parent, "overlay output parent")

    source_hashes_before_build = {
        module_name: _calculate_regular_file_sha256(source_path)
        for module_name, source_path in module_sources
    }
    temporary_build_root = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.build-", dir=output_path.parent)
    )
    staged_overlay_directory = temporary_build_root / "overlay"
    try:
        compiled_artifact_directory = temporary_build_root / "compiled"
        compiled_artifacts = _compile_cython_modules(
            package_root=package_root,
            module_sources=module_sources,
            compiled_artifact_directory=compiled_artifact_directory,
            compiler_work_directory=temporary_build_root / "compiler",
            build_target=build_target,
        )
        _assert_sources_unchanged(module_sources, source_hashes_before_build)

        artifact_records: list[CythonOverlayArtifact] = []
        for module_name, source_path in module_sources:
            compiled_artifact_path = compiled_artifacts[module_name]
            source_relative_path = source_path.relative_to(package_root).as_posix()
            installed_relative_path = _installed_artifact_path(
                package_name,
                module_name,
                build_target.extension_suffix,
            )
            overlay_relative_path = (
                PurePosixPath(_ARTIFACT_DIRECTORY_NAME) / installed_relative_path
            ).as_posix()
            staged_artifact_path = staged_overlay_directory / overlay_relative_path
            staged_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_copy_regular_file(compiled_artifact_path, staged_artifact_path)
            artifact_records.append(
                CythonOverlayArtifact(
                    module_name=module_name,
                    source_path=source_relative_path,
                    source_sha256=source_hashes_before_build[module_name],
                    source_size_bytes=_regular_file_size(source_path),
                    artifact_path=overlay_relative_path,
                    artifact_sha256=_calculate_regular_file_sha256(staged_artifact_path),
                    artifact_size_bytes=staged_artifact_path.stat().st_size,
                )
            )

        sorted_artifacts = tuple(
            sorted(artifact_records, key=lambda artifact: artifact.module_name)
        )
        toolchain = CythonToolchain(
            cython_version=cython_version,
            setuptools_version=setuptools_version,
        )
        manifest = CythonOverlayManifest(
            schema_version=CYTHON_OVERLAY_SCHEMA_VERSION,
            format=CYTHON_OVERLAY_FORMAT,
            package_name=package_name,
            bundle_digest=_calculate_bundle_digest(
                package_name=package_name,
                target=build_target,
                toolchain=toolchain,
                artifacts=sorted_artifacts,
            ),
            target=build_target,
            toolchain=toolchain,
            artifacts=sorted_artifacts,
        )
        _validate_manifest(manifest)
        write_cython_overlay_manifest(
            manifest,
            staged_overlay_directory / CYTHON_OVERLAY_MANIFEST_FILE_NAME,
        )
        verification_outcome = verify_cython_overlay(staged_overlay_directory)
        if not verification_outcome.succeeded:
            raise CythonOverlayBuildError(
                "new native overlay failed verification: "
                + _format_verification_issues(verification_outcome)
            )
        _synchronize_directory(staged_overlay_directory)
        staged_overlay_directory.replace(output_path)
        _synchronize_directory(output_path.parent)
        return manifest
    finally:
        shutil.rmtree(temporary_build_root, ignore_errors=True)


def verify_cython_overlay(
    overlay_directory: Path | str,
    *,
    target_package_root: Path | str | None = None,
) -> CythonOverlayVerificationOutcome:
    """Verify an overlay and, optionally, its already-applied target package."""

    overlay_path = Path(overlay_directory)
    _require_unsymlinked_directory(overlay_path, "native overlay root")
    manifest = load_cython_overlay_manifest(overlay_path / CYTHON_OVERLAY_MANIFEST_FILE_NAME)
    issues: list[CythonOverlayVerificationIssue] = []
    _append_target_issues(issues, manifest.target, current_native_build_target())

    expected_regular_files = {CYTHON_OVERLAY_MANIFEST_FILE_NAME}
    expected_directories: set[str] = set()
    overlay_entries = dict(_walk_directory_entries(overlay_path))
    for artifact in manifest.artifacts:
        expected_regular_files.add(artifact.artifact_path)
        expected_directories.update(_relative_parent_paths(artifact.artifact_path))
        artifact_path = overlay_path / artifact.artifact_path
        artifact_entry_kind = overlay_entries.get(artifact.artifact_path)
        if artifact_entry_kind is None:
            issues.append(
                CythonOverlayVerificationIssue(
                    CythonOverlayVerificationCode.ARTIFACT_MISSING,
                    artifact.artifact_path,
                )
            )
            continue
        if artifact_entry_kind != "file":
            issues.append(
                CythonOverlayVerificationIssue(
                    CythonOverlayVerificationCode.ARTIFACT_NOT_REGULAR,
                    artifact.artifact_path,
                )
            )
            continue
        _append_artifact_file_issues(
            issues,
            artifact_path,
            artifact.artifact_path,
            expected_sha256=artifact.artifact_sha256,
            expected_size_bytes=artifact.artifact_size_bytes,
            missing_code=CythonOverlayVerificationCode.ARTIFACT_MISSING,
            hash_mismatch_code=CythonOverlayVerificationCode.ARTIFACT_HASH_MISMATCH,
            size_mismatch_code=CythonOverlayVerificationCode.ARTIFACT_SIZE_MISMATCH,
        )

    for relative_path, entry_kind in overlay_entries.items():
        is_expected_path = (
            relative_path in expected_directories
            if entry_kind == "directory"
            else relative_path in expected_regular_files
        )
        if not is_expected_path:
            issues.append(
                CythonOverlayVerificationIssue(
                    CythonOverlayVerificationCode.UNEXPECTED_OVERLAY_PATH,
                    relative_path,
                )
            )

    if target_package_root is not None:
        _require_unsymlinked_directory(Path(target_package_root), "target package root")
        _append_applied_overlay_issues(
            issues,
            overlay_path,
            Path(target_package_root),
            manifest,
        )
    return CythonOverlayVerificationOutcome(tuple(issues))


def apply_cython_overlay(
    overlay_directory: Path | str,
    target_package_root: Path | str,
) -> CythonOverlayManifest:
    """Apply an overlay after a complete preflight, installing artifacts first.

    For a wheel installation, the owning ``.dist-info/RECORD`` first enters a
    prepared state that tracks both old and new paths. Native artifacts are
    then atomically renamed into place before any source disappears, and a
    final atomic RECORD replacement commits the new ownership. A retry with
    the same overlay can therefore finish a safely interrupted application.
    """

    overlay_path = Path(overlay_directory)
    target_root = Path(target_package_root)
    _require_unsymlinked_directory(overlay_path, "native overlay root")
    manifest = load_cython_overlay_manifest(overlay_path / CYTHON_OVERLAY_MANIFEST_FILE_NAME)
    overlay_verification = verify_cython_overlay(overlay_path)
    if not overlay_verification.succeeded:
        raise CythonOverlayApplyError(
            "native overlay is not usable: " + _format_verification_issues(overlay_verification)
        )
    _require_unsymlinked_directory(target_root, "target package root")
    installed_manifest_exists = _preflight_installed_manifest(overlay_path, target_root)
    wheel_record_update = _preflight_overlay_application(target_root, manifest)
    if (
        installed_manifest_exists
        and wheel_record_update is not None
        and wheel_record_update.state is _WheelRecordState.ORIGINAL
    ):
        raise CythonOverlayApplyError(
            "original wheel RECORD conflicts with an existing native overlay manifest"
        )
    if wheel_record_update is not None and wheel_record_update.state is _WheelRecordState.FINAL:
        applied_verification = verify_cython_overlay(
            overlay_path,
            target_package_root=target_root,
        )
        if not applied_verification.succeeded:
            raise CythonOverlayApplyError(
                "finalized native overlay failed verification: "
                + _format_verification_issues(applied_verification)
            )
        return manifest

    staged_artifacts: list[tuple[Path, Path]] = []
    try:
        for artifact in manifest.artifacts:
            destination_path = target_root / artifact.installed_artifact_path
            _require_unsymlinked_directory(
                destination_path.parent,
                "target artifact parent",
            )
            temporary_file_descriptor, temporary_path_text = tempfile.mkstemp(
                prefix=f".{destination_path.name}.",
                suffix=".tmp",
                dir=destination_path.parent,
            )
            temporary_path = Path(temporary_path_text)
            try:
                with os.fdopen(temporary_file_descriptor, "wb") as temporary_file:
                    _copy_regular_file_contents(
                        overlay_path / artifact.artifact_path,
                        temporary_file,
                    )
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                temporary_path.chmod(0o444)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
            staged_artifacts.append((temporary_path, destination_path))

        if wheel_record_update is not None:
            _atomic_replace_regular_file_bytes(
                wheel_record_update.record_path,
                expected_bytes=wheel_record_update.current_bytes,
                replacement_bytes=wheel_record_update.prepared_bytes,
                file_mode=wheel_record_update.original_file_mode,
            )

        for temporary_path, destination_path in staged_artifacts:
            temporary_path.replace(destination_path)
            _synchronize_directory(destination_path.parent)
        staged_artifacts.clear()

        for artifact in manifest.artifacts:
            destination_path = target_root / artifact.installed_artifact_path
            if _calculate_regular_file_sha256(destination_path) != artifact.artifact_sha256:
                raise CythonOverlayApplyError(
                    f"installed native artifact failed verification: {artifact.installed_artifact_path}"
                )

        installed_manifest_path = target_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME
        _atomic_copy_regular_file(
            overlay_path / CYTHON_OVERLAY_MANIFEST_FILE_NAME,
            installed_manifest_path,
        )

        for artifact in manifest.artifacts:
            source_path = target_root / artifact.source_path
            if source_path.exists() or source_path.is_symlink():
                if _calculate_regular_file_sha256(source_path) != artifact.source_sha256:
                    raise CythonOverlayApplyError(
                        f"source changed during overlay application: {artifact.source_path}"
                    )
                source_path.unlink()
                _synchronize_directory(source_path.parent)
            _remove_module_bytecode_files(source_path)

        if wheel_record_update is not None:
            _atomic_replace_regular_file_bytes(
                wheel_record_update.record_path,
                expected_bytes=wheel_record_update.prepared_bytes,
                replacement_bytes=wheel_record_update.final_bytes,
                file_mode=wheel_record_update.original_file_mode,
            )
    finally:
        for temporary_path, _ in staged_artifacts:
            temporary_path.unlink(missing_ok=True)

    applied_verification = verify_cython_overlay(
        overlay_path,
        target_package_root=target_root,
    )
    if not applied_verification.succeeded:
        raise CythonOverlayApplyError(
            "applied native overlay failed verification: "
            + _format_verification_issues(applied_verification)
        )
    return manifest


def write_cython_overlay_manifest(
    manifest: CythonOverlayManifest,
    destination_path: Path,
) -> None:
    """Write one canonical manifest; the caller controls final publication."""

    validated_manifest = _validate_manifest(manifest)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_manifest = _serialize_manifest(validated_manifest)
    destination_path.write_bytes(serialized_manifest)
    destination_path.chmod(0o444)
    _synchronize_file(destination_path)


def load_cython_overlay_manifest(manifest_path: Path) -> CythonOverlayManifest:
    """Parse untrusted manifest bytes into the strict overlay domain model."""

    serialized_manifest = _read_regular_file_bytes(manifest_path, "native overlay manifest")
    try:
        raw_manifest = cast(
            object,
            json.loads(
                serialized_manifest.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CythonOverlayManifestError("native overlay manifest is not valid JSON") from error
    manifest_mapping = _require_mapping(raw_manifest, "manifest")
    _require_exact_fields(
        manifest_mapping,
        {
            "schema_version",
            "format",
            "package_name",
            "bundle_digest",
            "target",
            "toolchain",
            "artifacts",
        },
        "manifest",
    )
    target_mapping = _require_mapping(manifest_mapping["target"], "target")
    _require_exact_fields(
        target_mapping,
        {
            "python_implementation",
            "python_version",
            "python_abi_tag",
            "extension_suffix",
            "platform_tag",
            "operating_system",
            "machine",
        },
        "target",
    )
    toolchain_mapping = _require_mapping(manifest_mapping["toolchain"], "toolchain")
    _require_exact_fields(
        toolchain_mapping,
        {"cython_version", "setuptools_version"},
        "toolchain",
    )
    raw_artifacts = manifest_mapping["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise CythonOverlayManifestError("artifacts must be an array")
    artifacts: list[CythonOverlayArtifact] = []
    for raw_artifact in cast(Sequence[object], raw_artifacts):
        artifact_mapping = _require_mapping(raw_artifact, "artifact")
        _require_exact_fields(
            artifact_mapping,
            {
                "module_name",
                "source_path",
                "source_sha256",
                "source_size_bytes",
                "artifact_path",
                "artifact_sha256",
                "artifact_size_bytes",
            },
            "artifact",
        )
        artifacts.append(
            CythonOverlayArtifact(
                module_name=_require_nonempty_string(
                    artifact_mapping["module_name"], "module_name"
                ),
                source_path=_require_relative_path(artifact_mapping["source_path"], "source_path"),
                source_sha256=_require_sha256(artifact_mapping["source_sha256"], "source_sha256"),
                source_size_bytes=_require_nonnegative_integer(
                    artifact_mapping["source_size_bytes"], "source_size_bytes"
                ),
                artifact_path=_require_relative_path(
                    artifact_mapping["artifact_path"], "artifact_path"
                ),
                artifact_sha256=_require_sha256(
                    artifact_mapping["artifact_sha256"], "artifact_sha256"
                ),
                artifact_size_bytes=_require_positive_integer(
                    artifact_mapping["artifact_size_bytes"], "artifact_size_bytes"
                ),
            )
        )
    manifest = CythonOverlayManifest(
        schema_version=_require_integer(manifest_mapping["schema_version"], "schema_version"),
        format=_require_nonempty_string(manifest_mapping["format"], "format"),
        package_name=_require_nonempty_string(manifest_mapping["package_name"], "package_name"),
        bundle_digest=_require_sha256(manifest_mapping["bundle_digest"], "bundle_digest"),
        target=NativeBuildTarget(
            python_implementation=_require_nonempty_string(
                target_mapping["python_implementation"], "python_implementation"
            ),
            python_version=_require_nonempty_string(
                target_mapping["python_version"], "python_version"
            ),
            python_abi_tag=_require_nonempty_string(
                target_mapping["python_abi_tag"], "python_abi_tag"
            ),
            extension_suffix=_require_nonempty_string(
                target_mapping["extension_suffix"], "extension_suffix"
            ),
            platform_tag=_require_nonempty_string(target_mapping["platform_tag"], "platform_tag"),
            operating_system=_require_nonempty_string(
                target_mapping["operating_system"], "operating_system"
            ),
            machine=_require_nonempty_string(target_mapping["machine"], "machine"),
        ),
        toolchain=CythonToolchain(
            cython_version=_require_nonempty_string(
                toolchain_mapping["cython_version"], "cython_version"
            ),
            setuptools_version=_require_nonempty_string(
                toolchain_mapping["setuptools_version"], "setuptools_version"
            ),
        ),
        artifacts=tuple(artifacts),
    )
    validated_manifest = _validate_manifest(manifest)
    if serialized_manifest != _serialize_manifest(validated_manifest):
        raise CythonOverlayManifestError("native overlay manifest is not canonical JSON")
    return validated_manifest


def _resolve_build_inputs(
    config: CythonOverlayBuildConfig,
) -> tuple[Path, str, tuple[tuple[str, Path], ...]]:
    package_root = Path(config.package_root).expanduser()
    _require_unsymlinked_directory(package_root, "package root")
    package_name = config.package_name or package_root.name
    _require_dotted_name(package_name, "package_name")
    if config.module_names is None:
        module_sources = _discover_module_sources(package_root, package_name)
    else:
        if not config.module_names:
            raise CythonOverlayBuildError("module_names must not be empty")
        module_sources = tuple(
            (module_name, _source_path_for_module(package_root, package_name, module_name))
            for module_name in config.module_names
        )
    sorted_module_sources = tuple(sorted(module_sources, key=lambda item: item[0]))
    module_names = [module_name for module_name, _ in sorted_module_sources]
    if len(module_names) != len(set(module_names)):
        raise CythonOverlayBuildError("module_names must be unique")
    for module_name, source_path in sorted_module_sources:
        _require_dotted_name(module_name, "module_name")
        if source_path.name in {"__init__.py", "__main__.py"}:
            raise CythonOverlayBuildError(
                f"package adapters are preserved and cannot be compiled: {source_path}"
            )
        _require_regular_path_below_root(package_root, source_path, f"source for {module_name}")
    if not sorted_module_sources:
        raise CythonOverlayBuildError("package root contains no implementation modules")
    return package_root, package_name, sorted_module_sources


def _discover_module_sources(
    package_root: Path,
    package_name: str,
) -> tuple[tuple[str, Path], ...]:
    modules: list[tuple[str, Path]] = []
    for relative_path, entry_kind in _walk_directory_entries(package_root):
        if entry_kind == "symlink":
            raise CythonOverlayBuildError(
                f"package root must not contain symbolic links: {relative_path}"
            )
        if entry_kind != "file" or not relative_path.endswith(".py"):
            continue
        relative_source = PurePosixPath(relative_path)
        if relative_source.name in {"__init__.py", "__main__.py"}:
            continue
        module_parts = relative_source.with_suffix("").parts
        if not all(part.isidentifier() for part in module_parts):
            raise CythonOverlayBuildError(
                f"Python source path cannot become an import name: {relative_path}"
            )
        modules.append((".".join((package_name, *module_parts)), package_root / relative_path))
    return tuple(modules)


def _source_path_for_module(
    package_root: Path,
    package_name: str,
    module_name: str,
) -> Path:
    _require_dotted_name(module_name, "module_name")
    package_prefix = f"{package_name}."
    if not module_name.startswith(package_prefix):
        raise CythonOverlayBuildError(
            f"module {module_name!r} must be below package {package_name!r}"
        )
    relative_module_name = module_name.removeprefix(package_prefix)
    return package_root.joinpath(*relative_module_name.split(".")).with_suffix(".py")


def _compile_cython_modules(
    *,
    package_root: Path,
    module_sources: tuple[tuple[str, Path], ...],
    compiled_artifact_directory: Path,
    compiler_work_directory: Path,
    build_target: NativeBuildTarget,
) -> dict[str, Path]:
    try:
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension
        from setuptools.command.build_ext import build_ext
    except ModuleNotFoundError as error:
        raise CythonOverlayBuildError(
            'native overlay builds need the optional extra: install "hflow[native-build]"'
        ) from error

    package_parent = package_root.parent
    compiled_artifact_directory.mkdir()
    compiler_work_directory.mkdir()
    compile_prefix_map = f"-ffile-prefix-map={compiler_work_directory}=."
    package_prefix_map = f"-ffile-prefix-map={package_parent}=."
    extension_definitions = [
        Extension(
            module_name,
            [source_path.relative_to(package_parent).as_posix()],
            extra_compile_args=["-O2", "-g0", compile_prefix_map, package_prefix_map],
            extra_link_args=["-Wl,--build-id=none"],
        )
        for module_name, source_path in module_sources
    ]
    with contextlib.chdir(package_parent), _reproducible_build_environment():
        compiled_extensions = cythonize(
            extension_definitions,
            build_dir=str(compiler_work_directory / "cython"),
            compiler_directives={
                "annotation_typing": False,
                "binding": True,
                "emit_code_comments": False,
                "embedsignature": False,
                "language_level": 3,
            },
            force=True,
            quiet=True,
        )
        distribution = Distribution(
            {"name": "hflow-native-overlay-build", "ext_modules": compiled_extensions}
        )
        build_command = build_ext(distribution)
        build_command.build_lib = str(compiled_artifact_directory)
        build_command.build_temp = str(compiler_work_directory / "objects")
        build_command.force = True
        build_command.ensure_finalized()
        build_command.run()

    strip_executable = shutil.which("strip")
    if strip_executable is None:
        raise CythonOverlayBuildError("the system 'strip' executable is required")
    compiled_artifacts: dict[str, Path] = {}
    for module_name, _ in module_sources:
        artifact_path = compiled_artifact_directory.joinpath(
            *module_name.split(".")[:-1],
            f"{module_name.rsplit('.', maxsplit=1)[-1]}{build_target.extension_suffix}",
        )
        if not artifact_path.is_file():
            raise CythonOverlayBuildError(f"compiler did not produce an artifact for {module_name}")
        try:
            subprocess.run(
                [
                    strip_executable,
                    "--strip-unneeded",
                    "--remove-section=.note.gnu.build-id",
                    str(artifact_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise CythonOverlayBuildError(
                f"could not strip the native artifact for {module_name}"
            ) from error
        _synchronize_file(artifact_path)
        compiled_artifacts[module_name] = artifact_path
    return compiled_artifacts


@contextlib.contextmanager
def _reproducible_build_environment() -> Iterator[None]:
    """Keep compiler date/time macros stable without leaking process state."""

    previous_source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        yield
    finally:
        if previous_source_date_epoch is None:
            del os.environ["SOURCE_DATE_EPOCH"]
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous_source_date_epoch


def _validate_manifest(manifest: CythonOverlayManifest) -> CythonOverlayManifest:
    if manifest.schema_version != CYTHON_OVERLAY_SCHEMA_VERSION:
        raise CythonOverlayManifestError("unsupported native overlay schema version")
    if manifest.format != CYTHON_OVERLAY_FORMAT:
        raise CythonOverlayManifestError("unsupported native overlay format")
    _require_dotted_name(manifest.package_name, "package_name")
    _require_sha256(manifest.bundle_digest, "bundle_digest")
    _validate_target(manifest.target)
    _require_nonempty_string(manifest.toolchain.cython_version, "cython_version")
    _require_nonempty_string(manifest.toolchain.setuptools_version, "setuptools_version")
    if not manifest.artifacts:
        raise CythonOverlayManifestError("artifacts must not be empty")
    if (
        tuple(sorted(manifest.artifacts, key=lambda artifact: artifact.module_name))
        != manifest.artifacts
    ):
        raise CythonOverlayManifestError("artifacts must be sorted by module_name")
    module_names = [artifact.module_name for artifact in manifest.artifacts]
    if len(module_names) != len(set(module_names)):
        raise CythonOverlayManifestError("artifact module names must be unique")
    for artifact in manifest.artifacts:
        _require_dotted_name(artifact.module_name, "module_name")
        expected_source_path = _source_relative_path(manifest.package_name, artifact.module_name)
        if artifact.source_path != expected_source_path:
            raise CythonOverlayManifestError(
                f"source_path does not match module_name for {artifact.module_name}"
            )
        expected_artifact_path = (
            PurePosixPath(_ARTIFACT_DIRECTORY_NAME)
            / _installed_artifact_path(
                manifest.package_name,
                artifact.module_name,
                manifest.target.extension_suffix,
            )
        ).as_posix()
        if artifact.artifact_path != expected_artifact_path:
            raise CythonOverlayManifestError(
                f"artifact_path does not match module_name for {artifact.module_name}"
            )
        _require_relative_path(artifact.source_path, "source_path")
        _require_relative_path(artifact.artifact_path, "artifact_path")
        _require_sha256(artifact.source_sha256, "source_sha256")
        _require_nonnegative_integer(artifact.source_size_bytes, "source_size_bytes")
        _require_sha256(artifact.artifact_sha256, "artifact_sha256")
        _require_positive_integer(artifact.artifact_size_bytes, "artifact_size_bytes")
    expected_bundle_digest = _calculate_bundle_digest(
        package_name=manifest.package_name,
        target=manifest.target,
        toolchain=manifest.toolchain,
        artifacts=manifest.artifacts,
    )
    if manifest.bundle_digest != expected_bundle_digest:
        raise CythonOverlayManifestError("bundle_digest does not match manifest components")
    return manifest


def _validate_target(target: NativeBuildTarget) -> None:
    for field_name, value in target.to_json_value().items():
        _require_nonempty_string(value, field_name)
    python_major_version, separator, python_minor_version = target.python_version.partition(".")
    if (
        separator != "."
        or not python_major_version.isdecimal()
        or not python_minor_version.isdecimal()
    ):
        raise CythonOverlayManifestError("python_version must be major.minor")
    if "/" in target.extension_suffix or "\\" in target.extension_suffix:
        raise CythonOverlayManifestError("extension_suffix must not contain a path separator")


def _calculate_bundle_digest(
    *,
    package_name: str,
    target: NativeBuildTarget,
    toolchain: CythonToolchain,
    artifacts: tuple[CythonOverlayArtifact, ...],
) -> str:
    digest_payload = {
        "schema_version": CYTHON_OVERLAY_SCHEMA_VERSION,
        "format": CYTHON_OVERLAY_FORMAT,
        "package_name": package_name,
        "target": target.to_json_value(),
        "toolchain": toolchain.to_json_value(),
        "artifacts": [artifact.to_json_value() for artifact in artifacts],
    }
    canonical_bytes = json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical_bytes).hexdigest()


def _serialize_manifest(manifest: CythonOverlayManifest) -> bytes:
    return (json.dumps(manifest.to_json_value(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _append_target_issues(
    issues: list[CythonOverlayVerificationIssue],
    expected: NativeBuildTarget,
    actual: NativeBuildTarget,
) -> None:
    comparisons = (
        (
            CythonOverlayVerificationCode.PYTHON_IMPLEMENTATION_MISMATCH,
            expected.python_implementation,
            actual.python_implementation,
        ),
        (
            CythonOverlayVerificationCode.PYTHON_VERSION_MISMATCH,
            expected.python_version,
            actual.python_version,
        ),
        (
            CythonOverlayVerificationCode.PYTHON_ABI_MISMATCH,
            expected.python_abi_tag,
            actual.python_abi_tag,
        ),
        (
            CythonOverlayVerificationCode.EXTENSION_SUFFIX_MISMATCH,
            expected.extension_suffix,
            actual.extension_suffix,
        ),
        (
            CythonOverlayVerificationCode.PLATFORM_MISMATCH,
            expected.platform_tag,
            actual.platform_tag,
        ),
        (
            CythonOverlayVerificationCode.OPERATING_SYSTEM_MISMATCH,
            expected.operating_system,
            actual.operating_system,
        ),
        (
            CythonOverlayVerificationCode.MACHINE_MISMATCH,
            expected.machine,
            actual.machine,
        ),
    )
    for code, expected_value, actual_value in comparisons:
        if expected_value != actual_value:
            issues.append(CythonOverlayVerificationIssue(code))


def _append_artifact_file_issues(
    issues: list[CythonOverlayVerificationIssue],
    file_path: Path,
    serialized_path: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    missing_code: CythonOverlayVerificationCode,
    hash_mismatch_code: CythonOverlayVerificationCode,
    size_mismatch_code: CythonOverlayVerificationCode | None = None,
) -> None:
    try:
        file_status = file_path.lstat()
    except FileNotFoundError:
        issues.append(CythonOverlayVerificationIssue(missing_code, serialized_path))
        return
    except OSError:
        issues.append(
            CythonOverlayVerificationIssue(
                CythonOverlayVerificationCode.ARTIFACT_NOT_REGULAR,
                serialized_path,
            )
        )
        return
    if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
        issues.append(
            CythonOverlayVerificationIssue(
                CythonOverlayVerificationCode.ARTIFACT_NOT_REGULAR,
                serialized_path,
            )
        )
        return
    if size_mismatch_code is not None and file_status.st_size != expected_size_bytes:
        issues.append(CythonOverlayVerificationIssue(size_mismatch_code, serialized_path))
    try:
        actual_sha256 = _calculate_regular_file_sha256(file_path)
    except OSError:
        issues.append(
            CythonOverlayVerificationIssue(
                CythonOverlayVerificationCode.ARTIFACT_NOT_REGULAR,
                serialized_path,
            )
        )
        return
    if actual_sha256 != expected_sha256:
        issues.append(CythonOverlayVerificationIssue(hash_mismatch_code, serialized_path))


def _append_applied_overlay_issues(
    issues: list[CythonOverlayVerificationIssue],
    overlay_path: Path,
    target_root: Path,
    manifest: CythonOverlayManifest,
) -> None:
    installed_manifest_path = target_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME
    if not installed_manifest_path.is_file() or installed_manifest_path.is_symlink():
        issues.append(
            CythonOverlayVerificationIssue(
                CythonOverlayVerificationCode.INSTALLED_MANIFEST_MISSING,
                INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
            )
        )
    else:
        expected_manifest_bytes = _read_regular_file_bytes(
            overlay_path / CYTHON_OVERLAY_MANIFEST_FILE_NAME,
            "native overlay manifest",
        )
        installed_manifest_bytes = _read_regular_file_bytes(
            installed_manifest_path,
            "installed native overlay manifest",
        )
        if installed_manifest_bytes != expected_manifest_bytes:
            issues.append(
                CythonOverlayVerificationIssue(
                    CythonOverlayVerificationCode.INSTALLED_MANIFEST_MISMATCH,
                    INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME,
                )
            )
    for artifact in manifest.artifacts:
        installed_artifact_path = target_root / artifact.installed_artifact_path
        _append_artifact_file_issues(
            issues,
            installed_artifact_path,
            artifact.installed_artifact_path,
            expected_sha256=artifact.artifact_sha256,
            expected_size_bytes=artifact.artifact_size_bytes,
            missing_code=CythonOverlayVerificationCode.INSTALLED_ARTIFACT_MISSING,
            hash_mismatch_code=CythonOverlayVerificationCode.INSTALLED_ARTIFACT_HASH_MISMATCH,
        )
        source_path = target_root / artifact.source_path
        if source_path.exists() or source_path.is_symlink():
            issues.append(
                CythonOverlayVerificationIssue(
                    CythonOverlayVerificationCode.SOURCE_STILL_PRESENT,
                    artifact.source_path,
                )
            )

    try:
        wheel_record_update = _resolve_wheel_record_update(target_root, manifest)
    except CythonOverlayApplyError:
        issues.append(
            CythonOverlayVerificationIssue(
                CythonOverlayVerificationCode.INSTALLED_RECORD_MISMATCH,
            )
        )
    else:
        if (
            wheel_record_update is not None
            and wheel_record_update.state is not _WheelRecordState.FINAL
        ):
            issues.append(
                CythonOverlayVerificationIssue(
                    CythonOverlayVerificationCode.INSTALLED_RECORD_MISMATCH,
                    wheel_record_update.record_path.name,
                )
            )


def _preflight_installed_manifest(overlay_path: Path, target_root: Path) -> bool:
    installed_manifest_path = target_root / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME
    try:
        installed_manifest_status = installed_manifest_path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise CythonOverlayApplyError(
            f"could not inspect installed native overlay manifest: {installed_manifest_path}"
        ) from error
    if stat.S_ISLNK(installed_manifest_status.st_mode) or not stat.S_ISREG(
        installed_manifest_status.st_mode
    ):
        raise CythonOverlayApplyError(
            f"installed native overlay manifest is not a regular file: {installed_manifest_path}"
        )
    expected_manifest_bytes = _read_regular_file_bytes(
        overlay_path / CYTHON_OVERLAY_MANIFEST_FILE_NAME,
        "native overlay manifest",
    )
    if (
        _read_regular_file_bytes(
            installed_manifest_path,
            "installed native overlay manifest",
        )
        != expected_manifest_bytes
    ):
        raise CythonOverlayApplyError(
            "target package already contains a different native overlay manifest"
        )
    return True


def _resolve_wheel_record_update(
    target_root: Path,
    manifest: CythonOverlayManifest,
) -> _WheelRecordUpdate | None:
    absolute_target_root = target_root.absolute()
    package_name_parts = tuple(manifest.package_name.split("."))
    if tuple(absolute_target_root.parts[-len(package_name_parts) :]) != package_name_parts:
        raise CythonOverlayApplyError(
            "target package path does not match the manifest's dotted package name"
        )
    installation_root = absolute_target_root.parents[len(package_name_parts) - 1]
    package_relative_root = PurePosixPath(*package_name_parts).as_posix()

    expected_source_rows: dict[Path, _WheelRecordRow] = {}
    for artifact in manifest.artifacts:
        relative_path = (PurePosixPath(package_relative_root) / artifact.source_path).as_posix()
        resolved_path = _resolve_record_row_path(installation_root, relative_path)
        expected_source_rows[resolved_path] = _WheelRecordRow(
            relative_path,
            _wheel_record_hash_from_manifest_sha256(artifact.source_sha256),
            str(artifact.source_size_bytes),
            resolved_path,
        )

    expected_runtime_rows: dict[Path, _WheelRecordRow] = {}
    for artifact in manifest.artifacts:
        relative_path = (
            PurePosixPath(package_relative_root) / artifact.installed_artifact_path
        ).as_posix()
        resolved_path = _resolve_record_row_path(installation_root, relative_path)
        expected_runtime_rows[resolved_path] = _WheelRecordRow(
            relative_path,
            _wheel_record_hash_from_manifest_sha256(artifact.artifact_sha256),
            str(artifact.artifact_size_bytes),
            resolved_path,
        )
    serialized_manifest = _serialize_manifest(manifest)
    installed_manifest_record_path = (
        PurePosixPath(package_relative_root) / INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME
    ).as_posix()
    installed_manifest_resolved_path = _resolve_record_row_path(
        installation_root,
        installed_manifest_record_path,
    )
    expected_runtime_rows[installed_manifest_resolved_path] = _WheelRecordRow(
        installed_manifest_record_path,
        _wheel_record_hash_from_bytes(serialized_manifest),
        str(len(serialized_manifest)),
        installed_manifest_resolved_path,
    )

    source_paths = frozenset(expected_source_rows)
    runtime_paths = frozenset(expected_runtime_rows)
    candidate_records: list[_ParsedWheelRecord] = []
    parsed_records = _load_sibling_wheel_records(installation_root)
    for parsed_record in parsed_records:
        recorded_paths = {row.resolved_path for row in parsed_record.rows}
        matching_source_paths = source_paths & recorded_paths
        matching_runtime_paths = runtime_paths & recorded_paths
        if not matching_source_paths and not matching_runtime_paths:
            continue
        if matching_source_paths and matching_source_paths != source_paths:
            raise CythonOverlayApplyError(
                f"wheel RECORD owns only some selected sources: {parsed_record.record_path}"
            )
        if matching_runtime_paths and matching_runtime_paths != runtime_paths:
            raise CythonOverlayApplyError(
                f"wheel RECORD owns only some native overlay files: {parsed_record.record_path}"
            )
        candidate_records.append(parsed_record)

    if not candidate_records:
        if not parsed_records:
            return None
        raise CythonOverlayApplyError(
            "no wheel RECORD unambiguously owns every selected package source"
        )
    if len(candidate_records) != 1:
        raise CythonOverlayApplyError(
            "selected package files have ambiguous wheel RECORD ownership: "
            + ", ".join(str(record.record_path) for record in candidate_records)
        )

    owning_record = candidate_records[0]
    _refuse_signed_wheel_record(owning_record.record_path)
    _validate_wheel_record_metadata(owning_record)
    rows_by_resolved_path = {row.resolved_path: row for row in owning_record.rows}
    source_rows_are_valid = all(
        (recorded_row := rows_by_resolved_path.get(path)) is not None
        and _wheel_record_source_row_is_valid(
            recorded_row,
            expected_row,
        )
        for path, expected_row in expected_source_rows.items()
    )
    source_rows_are_normalized = all(
        rows_by_resolved_path.get(path) == expected_row
        for path, expected_row in expected_source_rows.items()
    )
    source_rows_are_absent = all(path not in rows_by_resolved_path for path in expected_source_rows)
    runtime_rows_are_exact = all(
        rows_by_resolved_path.get(path) == expected_row
        for path, expected_row in expected_runtime_rows.items()
    )
    runtime_rows_are_absent = all(
        path not in rows_by_resolved_path for path in expected_runtime_rows
    )
    recorded_bytecode_paths = {
        row.resolved_path
        for row in owning_record.rows
        if any(
            _is_recorded_bytecode_for_source(row.resolved_path, source_path)
            for source_path in source_paths
        )
    }

    if source_rows_are_valid and runtime_rows_are_absent:
        record_state = _WheelRecordState.ORIGINAL
    elif source_rows_are_normalized and runtime_rows_are_exact:
        record_state = _WheelRecordState.PREPARED
    elif source_rows_are_absent and runtime_rows_are_exact and not recorded_bytecode_paths:
        record_state = _WheelRecordState.FINAL
    else:
        raise CythonOverlayApplyError(
            f"wheel RECORD is not in an original, prepared, or final state: "
            f"{owning_record.record_path}"
        )

    prepared_rows_by_path = {
        row.path: row
        for row in owning_record.rows
        if row.resolved_path not in source_paths | runtime_paths
    }
    prepared_rows_by_path.update((row.path, row) for row in expected_source_rows.values())
    prepared_rows_by_path.update((row.path, row) for row in expected_runtime_rows.values())
    final_rows_by_path = {
        path: row
        for path, row in prepared_rows_by_path.items()
        if row.resolved_path not in source_paths | recorded_bytecode_paths
    }

    return _WheelRecordUpdate(
        record_path=owning_record.record_path,
        original_file_mode=owning_record.file_mode,
        current_bytes=owning_record.serialized_bytes,
        prepared_bytes=_serialize_wheel_record(prepared_rows_by_path.values()),
        final_bytes=_serialize_wheel_record(final_rows_by_path.values()),
        state=record_state,
    )


def _load_sibling_wheel_records(installation_root: Path) -> tuple[_ParsedWheelRecord, ...]:
    parsed_records: list[_ParsedWheelRecord] = []
    try:
        metadata_entries = sorted(os.scandir(installation_root), key=lambda entry: entry.name)
    except OSError as error:
        raise CythonOverlayApplyError(
            f"could not inspect package installation root: {installation_root}"
        ) from error
    for metadata_entry in metadata_entries:
        if not metadata_entry.name.endswith(".dist-info"):
            continue
        if metadata_entry.is_symlink() or not metadata_entry.is_dir(follow_symlinks=False):
            raise CythonOverlayApplyError(
                f"distribution metadata path is not a regular directory: {metadata_entry.path}"
            )
        record_path = Path(metadata_entry.path) / "RECORD"
        try:
            record_status = record_path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CythonOverlayApplyError(
                f"could not inspect wheel RECORD: {record_path}"
            ) from error
        if stat.S_ISLNK(record_status.st_mode) or not stat.S_ISREG(record_status.st_mode):
            raise CythonOverlayApplyError(f"wheel RECORD is not a regular file: {record_path}")
        serialized_record = _read_regular_file_bytes(record_path, "wheel RECORD")
        parsed_records.append(
            _ParsedWheelRecord(
                record_path=record_path,
                file_mode=stat.S_IMODE(record_status.st_mode),
                serialized_bytes=serialized_record,
                rows=_parse_wheel_record(
                    serialized_record,
                    installation_root=installation_root,
                    record_path=record_path,
                ),
            )
        )
    return tuple(parsed_records)


def _parse_wheel_record(
    serialized_record: bytes,
    *,
    installation_root: Path,
    record_path: Path,
) -> tuple[_WheelRecordRow, ...]:
    try:
        record_text = serialized_record.decode("utf-8")
        parsed_csv_rows = tuple(csv.reader(io.StringIO(record_text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise CythonOverlayApplyError("wheel RECORD is not valid UTF-8 CSV") from error
    parsed_rows: list[_WheelRecordRow] = []
    recorded_paths: set[str] = set()
    resolved_paths: set[Path] = set()
    for csv_row in parsed_csv_rows:
        if len(csv_row) != 3:
            raise CythonOverlayApplyError("every wheel RECORD row must contain three fields")
        recorded_path_text, hash_value, size_value = csv_row
        if not recorded_path_text or "\x00" in recorded_path_text:
            raise CythonOverlayApplyError(
                f"wheel RECORD contains an invalid path: {recorded_path_text!r}"
            )
        if recorded_path_text in recorded_paths:
            raise CythonOverlayApplyError(
                f"wheel RECORD contains a duplicate path: {recorded_path_text}"
            )
        recorded_paths.add(recorded_path_text)
        resolved_path = _resolve_record_row_path(
            installation_root,
            recorded_path_text,
        )
        if resolved_path in resolved_paths:
            raise CythonOverlayApplyError(
                f"wheel RECORD contains duplicate path aliases: {recorded_path_text}"
            )
        resolved_paths.add(resolved_path)
        parsed_rows.append(
            _WheelRecordRow(recorded_path_text, hash_value, size_value, resolved_path)
        )

    absolute_record_path = _lexical_absolute_path(record_path)
    self_rows = [row for row in parsed_rows if row.resolved_path == absolute_record_path]
    if len(self_rows) != 1 or self_rows[0].hash_value or self_rows[0].size_value:
        raise CythonOverlayApplyError("wheel RECORD must contain its own unhashed row")
    return tuple(parsed_rows)


def _validate_wheel_record_metadata(record: _ParsedWheelRecord) -> None:
    for row in record.rows:
        if row.hash_value:
            hash_algorithm, separator, encoded_digest = row.hash_value.partition("=")
            if (
                separator != "="
                or hash_algorithm not in hashlib.algorithms_guaranteed
                or not encoded_digest
                or any(
                    character
                    not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                    for character in encoded_digest
                )
            ):
                raise CythonOverlayApplyError(
                    f"wheel RECORD contains an invalid hash for {row.path}"
                )
            try:
                decoded_digest = _decode_wheel_record_digest(encoded_digest)
                expected_digest_size = hashlib.new(hash_algorithm).digest_size
            except (ValueError, TypeError) as error:
                raise CythonOverlayApplyError(
                    f"wheel RECORD contains an invalid hash for {row.path}"
                ) from error
            if not expected_digest_size or len(decoded_digest) != expected_digest_size:
                raise CythonOverlayApplyError(
                    f"wheel RECORD contains an invalid hash for {row.path}"
                )
        if row.size_value and (not row.size_value.isascii() or not row.size_value.isdecimal()):
            raise CythonOverlayApplyError(f"wheel RECORD contains an invalid size for {row.path}")


def _serialize_wheel_record(rows: Iterable[_WheelRecordRow]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for row in sorted(rows, key=lambda candidate: candidate.path):
        writer.writerow((row.path, row.hash_value, row.size_value))
    return output.getvalue().encode("utf-8")


def _wheel_record_hash_from_manifest_sha256(manifest_sha256: str) -> str:
    _, hexadecimal_digest = manifest_sha256.split(":", maxsplit=1)
    encoded_digest = base64.urlsafe_b64encode(bytes.fromhex(hexadecimal_digest)).rstrip(b"=")
    return "sha256=" + encoded_digest.decode("ascii")


def _wheel_record_hash_from_bytes(contents: bytes) -> str:
    encoded_digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=")
    return "sha256=" + encoded_digest.decode("ascii")


def _wheel_record_source_row_is_valid(
    recorded_row: _WheelRecordRow,
    normalized_expected_row: _WheelRecordRow,
) -> bool:
    if recorded_row.size_value and int(recorded_row.size_value) != int(
        normalized_expected_row.size_value
    ):
        return False
    if not recorded_row.hash_value:
        return True
    hash_algorithm, _, encoded_digest = recorded_row.hash_value.partition("=")
    recorded_digest = _decode_wheel_record_digest(encoded_digest)
    if hash_algorithm == "sha256":
        _, expected_hexadecimal_digest = normalized_expected_row.hash_value.split("=", maxsplit=1)
        return recorded_digest == _decode_wheel_record_digest(expected_hexadecimal_digest)
    try:
        actual_digest = _calculate_regular_file_digest(
            normalized_expected_row.resolved_path,
            hash_algorithm,
        )
    except OSError:
        return False
    return actual_digest == recorded_digest


def _decode_wheel_record_digest(encoded_digest: str) -> bytes:
    padding = "=" * (-len(encoded_digest) % 4)
    return base64.b64decode(
        encoded_digest + padding,
        altchars=b"-_",
        validate=True,
    )


def _resolve_record_row_path(installation_root: Path, serialized_path: str) -> Path:
    posix_path = PurePosixPath(serialized_path)
    if posix_path.is_absolute():
        return _lexical_absolute_path(Path(posix_path.as_posix()))
    return _lexical_absolute_path(installation_root.joinpath(*posix_path.parts))


def _lexical_absolute_path(path: Path) -> Path:
    # RECORD ownership is lexical; resolve() would follow attacker-controlled symlinks.
    return Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100


def _is_recorded_bytecode_for_source(record_path: Path, source_path: Path) -> bool:
    if record_path == source_path.with_suffix(".pyc"):
        return True
    return (
        record_path.parent == source_path.parent / "__pycache__"
        and record_path.name.startswith(f"{source_path.stem}.")
        and record_path.name.endswith(".pyc")
    )


def _refuse_signed_wheel_record(record_path: Path) -> None:
    for signature_name in ("RECORD.jws", "RECORD.p7s"):
        signature_path = record_path.parent / signature_name
        try:
            signature_path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CythonOverlayApplyError(
                f"could not inspect wheel RECORD signature: {signature_path}"
            ) from error
        raise CythonOverlayApplyError(f"cannot update a signed wheel RECORD: {signature_path}")


def _preflight_overlay_application(
    target_root: Path,
    manifest: CythonOverlayManifest,
) -> _WheelRecordUpdate | None:
    for artifact in manifest.artifacts:
        source_path = target_root / artifact.source_path
        destination_path = target_root / artifact.installed_artifact_path
        _require_relative_path_parents_unsymlinked(
            target_root,
            PurePosixPath(artifact.source_path),
            "target source",
        )
        _require_relative_path_parents_unsymlinked(
            target_root,
            PurePosixPath(artifact.installed_artifact_path),
            "target artifact",
        )
        _preflight_module_bytecode_directory(source_path)
        source_exists = source_path.exists() or source_path.is_symlink()
        destination_exists = destination_path.exists() or destination_path.is_symlink()
        if source_exists:
            try:
                source_sha256 = _calculate_regular_file_sha256(source_path)
            except OSError as error:
                raise CythonOverlayApplyError(
                    f"target source is not a regular file: {artifact.source_path}"
                ) from error
            if source_sha256 != artifact.source_sha256:
                raise CythonOverlayApplyError(
                    f"target source does not match the build receipt: {artifact.source_path}"
                )
        elif not destination_exists:
            raise CythonOverlayApplyError(
                f"target source is missing before native artifact installation: {artifact.source_path}"
            )
        if destination_exists:
            destination_status = destination_path.lstat()
            if stat.S_ISLNK(destination_status.st_mode) or not stat.S_ISREG(
                destination_status.st_mode
            ):
                raise CythonOverlayApplyError(
                    f"target artifact path is not a regular file: {artifact.installed_artifact_path}"
                )
            if _calculate_regular_file_sha256(destination_path) != artifact.artifact_sha256:
                raise CythonOverlayApplyError(
                    f"partially applied artifact does not match: {artifact.installed_artifact_path}"
                )
    wheel_record_update = _resolve_wheel_record_update(target_root, manifest)
    if wheel_record_update is not None and wheel_record_update.state is _WheelRecordState.ORIGINAL:
        for artifact in manifest.artifacts:
            source_path = target_root / artifact.source_path
            destination_path = target_root / artifact.installed_artifact_path
            if not source_path.is_file() or source_path.is_symlink():
                raise CythonOverlayApplyError(
                    f"original wheel RECORD has no selected source on disk: {artifact.source_path}"
                )
            if destination_path.exists() or destination_path.is_symlink():
                raise CythonOverlayApplyError(
                    "original wheel RECORD conflicts with an existing native artifact: "
                    f"{artifact.installed_artifact_path}"
                )
    if wheel_record_update is not None and wheel_record_update.state is _WheelRecordState.FINAL:
        for artifact in manifest.artifacts:
            source_path = target_root / artifact.source_path
            if source_path.exists() or source_path.is_symlink():
                raise CythonOverlayApplyError(
                    f"final wheel RECORD still has a selected source on disk: {artifact.source_path}"
                )
    return wheel_record_update


def _remove_module_bytecode_files(source_path: Path) -> None:
    legacy_bytecode_path = source_path.with_suffix(".pyc")
    try:
        legacy_bytecode_status = legacy_bytecode_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(legacy_bytecode_status.st_mode) or not stat.S_ISREG(
            legacy_bytecode_status.st_mode
        ):
            raise CythonOverlayApplyError(
                f"bytecode cache entry is not a regular file: {legacy_bytecode_path}"
            )
        legacy_bytecode_path.unlink()
        _synchronize_directory(legacy_bytecode_path.parent)
    bytecode_directory = source_path.parent / "__pycache__"
    try:
        directory_status = bytecode_directory.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(directory_status.st_mode):
        raise CythonOverlayApplyError(
            f"bytecode cache path is not a regular directory: {bytecode_directory}"
        )
    filename_prefix = f"{source_path.stem}."
    for entry in os.scandir(bytecode_directory):
        if not entry.name.startswith(filename_prefix) or not entry.name.endswith(".pyc"):
            continue
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise CythonOverlayApplyError(
                f"bytecode cache entry is not a regular file: {entry.path}"
            )
        Path(entry.path).unlink()
    _synchronize_directory(bytecode_directory)


def _preflight_module_bytecode_directory(source_path: Path) -> None:
    legacy_bytecode_path = source_path.with_suffix(".pyc")
    try:
        legacy_bytecode_status = legacy_bytecode_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(legacy_bytecode_status.st_mode) or not stat.S_ISREG(
            legacy_bytecode_status.st_mode
        ):
            raise CythonOverlayApplyError(
                f"bytecode cache entry is not a regular file: {legacy_bytecode_path}"
            )
    bytecode_directory = source_path.parent / "__pycache__"
    try:
        directory_status = bytecode_directory.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(directory_status.st_mode):
        raise CythonOverlayApplyError(
            f"bytecode cache path is not a regular directory: {bytecode_directory}"
        )
    filename_prefix = f"{source_path.stem}."
    for entry in os.scandir(bytecode_directory):
        if not entry.name.startswith(filename_prefix) or not entry.name.endswith(".pyc"):
            continue
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise CythonOverlayApplyError(
                f"bytecode cache entry is not a regular file: {entry.path}"
            )


def _atomic_copy_regular_file(source_path: Path, destination_path: Path) -> None:
    _require_unsymlinked_directory(destination_path.parent, "destination parent")
    temporary_file_descriptor, temporary_path_text = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    temporary_path = Path(temporary_path_text)
    try:
        with os.fdopen(temporary_file_descriptor, "wb") as temporary_file:
            _copy_regular_file_contents(source_path, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.chmod(0o444)
        temporary_path.replace(destination_path)
        _synchronize_directory(destination_path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_replace_regular_file_bytes(
    destination_path: Path,
    *,
    expected_bytes: bytes,
    replacement_bytes: bytes,
    file_mode: int,
) -> None:
    current_bytes = _read_regular_file_bytes(destination_path, "file to replace atomically")
    if current_bytes != expected_bytes:
        raise CythonOverlayApplyError(f"file changed after overlay preflight: {destination_path}")
    if current_bytes == replacement_bytes:
        return
    _require_unsymlinked_directory(destination_path.parent, "atomic replacement parent")
    temporary_file_descriptor, temporary_path_text = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    temporary_path = Path(temporary_path_text)
    try:
        with os.fdopen(temporary_file_descriptor, "wb") as temporary_file:
            temporary_file.write(replacement_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.chmod(file_mode)
        temporary_path.replace(destination_path)
        _synchronize_directory(destination_path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _assert_sources_unchanged(
    module_sources: tuple[tuple[str, Path], ...],
    source_hashes_before_build: Mapping[str, str],
) -> None:
    for module_name, source_path in module_sources:
        if _calculate_regular_file_sha256(source_path) != source_hashes_before_build[module_name]:
            raise CythonOverlayBuildError(f"source changed during build: {source_path}")


def _source_relative_path(package_name: str, module_name: str) -> str:
    prefix = f"{package_name}."
    if not module_name.startswith(prefix):
        raise CythonOverlayManifestError(
            f"module {module_name!r} is not below package {package_name!r}"
        )
    return PurePosixPath(*module_name.removeprefix(prefix).split(".")).with_suffix(".py").as_posix()


def _installed_artifact_path(
    package_name: str,
    module_name: str,
    extension_suffix: str,
) -> str:
    source_relative_path = PurePosixPath(_source_relative_path(package_name, module_name))
    return source_relative_path.with_suffix(extension_suffix).as_posix()


def _relative_parent_paths(relative_path: str) -> set[str]:
    parent_paths: set[str] = set()
    current_parent = PurePosixPath(relative_path).parent
    while current_parent != PurePosixPath("."):
        parent_paths.add(current_parent.as_posix())
        current_parent = current_parent.parent
    return parent_paths


def _walk_directory_entries(directory_path: Path) -> Iterator[tuple[str, str]]:
    """Yield normalized relative paths without following symbolic links."""

    _require_unsymlinked_directory(directory_path, "directory")

    def walk(current_directory: Path, relative_parent: PurePosixPath) -> Iterator[tuple[str, str]]:
        try:
            entries = sorted(os.scandir(current_directory), key=lambda entry: entry.name)
        except OSError as error:
            raise CythonOverlayManifestError(
                f"could not inspect directory {current_directory}"
            ) from error
        for entry in entries:
            relative_path = (relative_parent / entry.name).as_posix()
            if entry.is_symlink():
                yield relative_path, "symlink"
            elif entry.is_dir(follow_symlinks=False):
                yield relative_path, "directory"
                yield from walk(Path(entry.path), relative_parent / entry.name)
            elif entry.is_file(follow_symlinks=False):
                yield relative_path, "file"
            else:
                yield relative_path, "other"

    yield from walk(directory_path, PurePosixPath())


def _require_regular_path_below_root(root: Path, path: Path, label: str) -> None:
    try:
        relative_path = path.relative_to(root)
    except ValueError as error:
        raise CythonOverlayBuildError(f"{label} must remain below package root") from error
    current_path = root
    for path_part in relative_path.parts[:-1]:
        current_path = current_path / path_part
        _require_unsymlinked_directory(current_path, f"{label} parent")
    try:
        path_status = path.lstat()
    except OSError as error:
        raise CythonOverlayBuildError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise CythonOverlayBuildError(f"{label} must be a regular file: {path}")


def _require_relative_path_parents_unsymlinked(
    root: Path,
    relative_path: PurePosixPath,
    label: str,
) -> None:
    current_parent = root
    for path_part in relative_path.parts[:-1]:
        current_parent = current_parent / path_part
        try:
            _require_unsymlinked_directory(current_parent, f"{label} parent")
        except OSError as error:
            raise CythonOverlayApplyError(str(error)) from error


def _calculate_regular_file_sha256(file_path: Path) -> str:
    return "sha256:" + _calculate_regular_file_digest(file_path, "sha256").hex()


def _calculate_regular_file_digest(file_path: Path, hash_algorithm: str) -> bytes:
    file_descriptor = _open_regular_file(file_path, "file")
    digest = hashlib.new(hash_algorithm)
    try:
        while chunk := os.read(file_descriptor, _FILE_READ_CHUNK_SIZE_BYTES):
            digest.update(chunk)
    finally:
        os.close(file_descriptor)
    return digest.digest()


def _regular_file_size(file_path: Path) -> int:
    file_descriptor = _open_regular_file(file_path, "file")
    try:
        return os.fstat(file_descriptor).st_size
    finally:
        os.close(file_descriptor)


def _read_regular_file_bytes(file_path: Path, label: str) -> bytes:
    file_descriptor = _open_regular_file(file_path, label)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(file_descriptor, _FILE_READ_CHUNK_SIZE_BYTES):
            chunks.append(chunk)
    finally:
        os.close(file_descriptor)
    return b"".join(chunks)


def _copy_regular_file_contents(source_path: Path, destination_file: BinaryIO) -> None:
    """Copy from a no-follow source descriptor into an already-open binary file."""

    source_file_descriptor = _open_regular_file(source_path, "copy source")
    try:
        while chunk := os.read(source_file_descriptor, _FILE_READ_CHUNK_SIZE_BYTES):
            written_byte_count = destination_file.write(chunk)
            if written_byte_count != len(chunk):
                raise OSError("short write while copying native overlay file")
    finally:
        os.close(source_file_descriptor)


def _open_regular_file(file_path: Path, label: str) -> int:
    try:
        inspected_status = file_path.lstat()
    except OSError as error:
        raise OSError(f"{label} could not be inspected: {file_path}") from error
    if stat.S_ISLNK(inspected_status.st_mode) or not stat.S_ISREG(inspected_status.st_mode):
        raise OSError(f"{label} must be a regular file: {file_path}")
    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(file_path, open_flags)
    try:
        opened_status = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_status.st_mode) or (
            opened_status.st_dev,
            opened_status.st_ino,
        ) != (inspected_status.st_dev, inspected_status.st_ino):
            raise OSError(f"{label} changed while being opened: {file_path}")
    except BaseException:
        os.close(file_descriptor)
        raise
    return file_descriptor


def _require_unsymlinked_directory(directory_path: Path, label: str) -> None:
    try:
        directory_status = directory_path.lstat()
    except OSError as error:
        raise OSError(f"{label} could not be inspected: {directory_path}") from error
    if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(directory_status.st_mode):
        raise OSError(f"{label} must be a directory: {directory_path}")


def _synchronize_file(file_path: Path) -> None:
    file_descriptor = _open_regular_file(file_path, "file to synchronize")
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _synchronize_directory(directory_path: Path) -> None:
    directory_file_descriptor = os.open(
        directory_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_file_descriptor)
    finally:
        os.close(directory_file_descriptor)


def _paths_overlap(first_path: Path, second_path: Path) -> bool:
    try:
        first_path.relative_to(second_path)
    except ValueError:
        pass
    else:
        return True
    try:
        second_path.relative_to(first_path)
    except ValueError:
        return False
    return True


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CythonOverlayManifestError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _require_exact_fields(
    mapping: Mapping[str, object],
    expected_fields: set[str],
    label: str,
) -> None:
    if set(mapping) != expected_fields:
        raise CythonOverlayManifestError(f"{label} has unexpected fields")


def _require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CythonOverlayManifestError(f"{label} must be an integer")
    return value


def _require_positive_integer(value: object, label: str) -> int:
    parsed_value = _require_integer(value, label)
    if parsed_value <= 0:
        raise CythonOverlayManifestError(f"{label} must be positive")
    return parsed_value


def _require_nonnegative_integer(value: object, label: str) -> int:
    parsed_value = _require_integer(value, label)
    if parsed_value < 0:
        raise CythonOverlayManifestError(f"{label} must not be negative")
    return parsed_value


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise CythonOverlayManifestError(f"{label} must be a non-empty printable string")
    return value


def _require_dotted_name(value: object, label: str) -> str:
    parsed_value = _require_nonempty_string(value, label)
    name_parts = parsed_value.split(".")
    if any(
        not name_part.isidentifier() or keyword.iskeyword(name_part) for name_part in name_parts
    ):
        raise CythonOverlayManifestError(f"{label} must be a dotted Python identifier")
    return parsed_value


def _require_relative_path(value: object, label: str) -> str:
    parsed_value = _require_nonempty_string(value, label)
    if "\\" in parsed_value:
        raise CythonOverlayManifestError(f"{label} must use POSIX separators")
    relative_path = PurePosixPath(parsed_value)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(path_part in {".", ".."} for path_part in relative_path.parts)
        or relative_path.as_posix() != parsed_value
    ):
        raise CythonOverlayManifestError(f"{label} must remain within its root")
    return parsed_value


def _require_sha256(value: object, label: str) -> str:
    parsed_value = _require_nonempty_string(value, label)
    algorithm, separator, hexadecimal_digest = parsed_value.partition(":")
    if (
        algorithm != "sha256"
        or separator != ":"
        or len(hexadecimal_digest) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in hexadecimal_digest)
    ):
        raise CythonOverlayManifestError(f"{label} must be a lowercase SHA-256")
    return parsed_value


def _reject_duplicate_json_fields(field_pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed_fields: dict[str, object] = {}
    for field_name, field_value in field_pairs:
        if field_name in parsed_fields:
            raise CythonOverlayManifestError("native overlay manifest contains duplicate fields")
        parsed_fields[field_name] = field_value
    return parsed_fields


def _format_verification_issues(outcome: CythonOverlayVerificationOutcome) -> str:
    return ", ".join(
        issue.code.value + (f":{issue.path}" if issue.path is not None else "")
        for issue in outcome.issues
    )


__all__ = [
    "CYTHON_OVERLAY_FORMAT",
    "CYTHON_OVERLAY_MANIFEST_FILE_NAME",
    "CYTHON_OVERLAY_SCHEMA_VERSION",
    "INSTALLED_CYTHON_OVERLAY_MANIFEST_FILE_NAME",
    "CythonOverlayApplyError",
    "CythonOverlayArtifact",
    "CythonOverlayBuildConfig",
    "CythonOverlayBuildError",
    "CythonOverlayManifest",
    "CythonOverlayManifestError",
    "CythonOverlayVerificationCode",
    "CythonOverlayVerificationIssue",
    "CythonOverlayVerificationOutcome",
    "CythonToolchain",
    "NativeBuildTarget",
    "apply_cython_overlay",
    "build_cython_overlay",
    "current_native_build_target",
    "load_cython_overlay_manifest",
    "verify_cython_overlay",
]
