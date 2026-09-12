"""Offline verification of explicitly supplied media binaries."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hflow.ffmpeg._binary import PINNED_VERSION_LABEL
from hflow.ffmpeg._process import MediaToolError, run_media_command


@dataclass(frozen=True)
class MediaBinarySpec:
    name: Literal["ffmpeg", "ffprobe"]
    sha256: str
    version_label: str

    def __post_init__(self) -> None:
        if (
            self.name not in ("ffmpeg", "ffprobe")
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
            or not self.version_label
        ):
            raise ValueError("invalid media binary specification")


@dataclass(frozen=True)
class VerifiedMediaBinary:
    path: Path
    specification: MediaBinarySpec
    version: str


# Exact executables from the retained pinned Linux x86-64 distribution.
# Callers on other platforms supply the reviewed specification for that build.
PINNED_LINUX_X86_64_FFMPEG = MediaBinarySpec(
    "ffmpeg",
    "414037f32c343c3a1254a108e6ce6523770ce81e95ce64c67aa97ee3e612411f",
    PINNED_VERSION_LABEL,
)
PINNED_LINUX_X86_64_FFPROBE = MediaBinarySpec(
    "ffprobe",
    "91505681b7e1548c9754666e945cb4e53e7f0d66bc172838a18419f82992e62d",
    PINNED_VERSION_LABEL,
)


def verify_media_binary(path: Path, specification: MediaBinarySpec) -> VerifiedMediaBinary:
    """Verify bytes before executing -version. Never download or resolve a binary."""
    path = path.resolve(strict=True)
    with path.open("rb") as binary_stream:
        digest = hashlib.file_digest(binary_stream, "sha256").hexdigest()
    if digest != specification.sha256:
        raise MediaToolError("media binary checksum does not match")
    result = run_media_command([str(path), "-version"], timeout_seconds=10)
    fields = result.stdout.decode(errors="replace").split()
    if result.returncode or len(fields) < 3 or fields[:2] != [specification.name, "version"]:
        raise MediaToolError("media binary version is invalid")
    version = fields[2]
    if version != specification.version_label and not version.startswith(
        specification.version_label + "-"
    ):
        raise MediaToolError("media binary version does not match")
    return VerifiedMediaBinary(path, specification, version)
