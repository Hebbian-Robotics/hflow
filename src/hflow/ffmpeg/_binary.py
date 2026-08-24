"""Pinned ffmpeg/ffprobe management.

Different ffmpeg versions genuinely produce different measurements -- the
binary is a measuring instrument -- so the SDK never silently uses whatever is
on PATH. ffprobe ships in the same release archive and must match the ffmpeg
it probes for, so the pair is installed together and resolved by the same
policy. Resolution order for ffmpeg:

1. Explicit user override via ``HFLOW_FFMPEG`` (a deliberate choice;
   used by tests and users with their own managed builds).
2. The pinned static build, auto-downloaded once into the user cache dir and
   sha256-verified (Linux x86_64/aarch64 from BtbN/FFmpeg-Builds immutable
   release assets; Windows runs the Linux path under WSL2). On Linux there is
   deliberately NO PATH fallback: either the pinned build resolves or we raise
   with instructions, so two Linux machines can never silently disagree.
3. On platforms without a pinned build (macOS, native Windows), the PATH
   binary, with a loud warning. Whatever is resolved, its version string is
   recorded in every measurement identity (``provenance/v1``).

ffprobe mirrors that order via ``HFLOW_FFPROBE``, with one exception:
when the ffmpeg override is set but no ffprobe override is, the sibling
``ffprobe`` next to the overridden ffmpeg wins (the user's build ships its own
matching probe), falling back to PATH with the loud warning.

Cache layout: ``<user cache>/hflow/ffmpeg/<release tag>-<machine>/``
holds BOTH ``ffmpeg`` and ``ffprobe``; an install counts as complete only when
both binaries are present. Older installs placed a lone ``ffmpeg`` in the same
directory -- such incomplete directories are healed by re-extracting both
binaries from a fresh verified download.
"""

import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hflow._pinned_asset import (
    PinnedAssetError,
    download_url_to_file,
    user_cache_dir,
    verify_sha256,
)

logger = logging.getLogger(__name__)

FFMPEG_ENV_VAR = "HFLOW_FFMPEG"
FFPROBE_ENV_VAR = "HFLOW_FFPROBE"

# Both binaries come from one archive and are published together: presence of
# BOTH in the versioned install dir is the completion condition.
_PINNED_BINARY_NAMES = ("ffmpeg", "ffprobe")

# The pinned build is a dated BtbN/FFmpeg-Builds autobuild tag (NOT the
# floating "latest" tag): dated release assets are immutable, so the URL and
# bytes below can never change underneath the sha256 pins.
PINNED_RELEASE_TAG = "autobuild-2026-08-16-13-00"
PINNED_VERSION_LABEL = "n8.1.2-44-g7c533d0f86"

_RELEASE_DOWNLOAD_BASE_URL = (
    f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{PINNED_RELEASE_TAG}"
)


@dataclass(frozen=True)
class PinnedBuild:
    """One immutable release asset: where to get it and how to verify it."""

    url: str
    sha256_hex: str
    # Directory inside the tar.xz holding the binaries we extract
    # (``<archive_bin_dir>/ffmpeg`` and ``<archive_bin_dir>/ffprobe``).
    archive_bin_dir: str


PINNED_BUILDS_BY_MACHINE: dict[str, PinnedBuild] = {
    "x86_64": PinnedBuild(
        url=f"{_RELEASE_DOWNLOAD_BASE_URL}/ffmpeg-{PINNED_VERSION_LABEL}-linux64-gpl-8.1.tar.xz",
        sha256_hex="17780994c4679806fb227676f66a0af30c6379afc770324829f48f2a379be558",
        archive_bin_dir=f"ffmpeg-{PINNED_VERSION_LABEL}-linux64-gpl-8.1/bin",
    ),
    "aarch64": PinnedBuild(
        url=f"{_RELEASE_DOWNLOAD_BASE_URL}/ffmpeg-{PINNED_VERSION_LABEL}-linuxarm64-gpl-8.1.tar.xz",
        sha256_hex="e970a7dd450b440a21126a8bac3a1c95178b6ba05bee2465a4d2a586345c81ac",
        archive_bin_dir=f"ffmpeg-{PINNED_VERSION_LABEL}-linuxarm64-gpl-8.1/bin",
    ),
}


class FfmpegNotFoundError(RuntimeError):
    pass


class FfprobeNotFoundError(FfmpegNotFoundError):
    """ffprobe specifically could not be resolved."""


class PinnedDownloadError(FfmpegNotFoundError):
    """The pinned build could not be downloaded, verified, or extracted."""


def _pinned_install_dir(machine: str) -> Path:
    return user_cache_dir("ffmpeg") / f"{PINNED_RELEASE_TAG}-{machine}"


def _install_is_complete(install_dir: Path) -> bool:
    return all((install_dir / binary_name).is_file() for binary_name in _PINNED_BINARY_NAMES)


def _extract_archive_member(archive: Path, member_name: str, destination: Path) -> None:
    with tarfile.open(archive, mode="r:xz") as tar:
        try:
            member_file = tar.extractfile(member_name)
        except KeyError:
            member_file = None
        if member_file is None:
            raise PinnedDownloadError(f"archive {archive.name} has no member {member_name!r}")
        with member_file, destination.open("wb") as destination_file:
            shutil.copyfileobj(member_file, destination_file)
    destination.chmod(0o755)


def _install_pinned_build(build: PinnedBuild, install_dir: Path) -> Path:
    """Download once, verify, extract ffmpeg AND ffprobe, atomically publish.

    All temp files live next to ``install_dir`` contents so each final
    ``os.replace`` is an atomic same-filesystem rename: a concurrent second
    process either sees no file or a complete verified binary, never a partial
    write. Both binaries are extracted before either is published; a crash
    between the two renames leaves an incomplete dir that the next resolution
    heals from a fresh download (completion condition: both binaries present).
    """
    install_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=install_dir, prefix=".install-") as staging_dir_name:
        staging_dir = Path(staging_dir_name)
        archive_path = staging_dir / "download.tar.xz"
        # The shared mechanism raises its own error type; this module's callers
        # catch PinnedDownloadError (an FfmpegNotFoundError), so translate at
        # the boundary rather than widening what they must handle.
        try:
            download_url_to_file(build.url, archive_path)
            verify_sha256(archive_path, build.sha256_hex, build.url)
        except PinnedAssetError as error:
            raise PinnedDownloadError(str(error)) from error
        staged_binaries_by_name: dict[str, Path] = {}
        for binary_name in _PINNED_BINARY_NAMES:
            staged_binary = staging_dir / binary_name
            _extract_archive_member(
                archive_path, f"{build.archive_bin_dir}/{binary_name}", staged_binary
            )
            staged_binaries_by_name[binary_name] = staged_binary
        for binary_name, staged_binary in staged_binaries_by_name.items():
            staged_binary.replace(install_dir / binary_name)
    return install_dir


def _resolve_pinned_linux_install_dir() -> Path:
    """The versioned cache dir holding both pinned binaries, installing if incomplete."""
    machine = platform.machine()
    build = PINNED_BUILDS_BY_MACHINE.get(machine)
    if build is None:
        raise FfmpegNotFoundError(
            f"no pinned ffmpeg/ffprobe build exists for Linux/{machine} "
            f"(pinned architectures: {sorted(PINNED_BUILDS_BY_MACHINE)}). "
            f"Set {FFMPEG_ENV_VAR}=/path/to/ffmpeg (and optionally "
            f"{FFPROBE_ENV_VAR}=/path/to/ffprobe) to use your own binaries."
        )
    install_dir = _pinned_install_dir(machine)
    if _install_is_complete(install_dir):
        return install_dir
    logger.info(
        "downloading pinned ffmpeg+ffprobe %s (%s) to %s", PINNED_RELEASE_TAG, machine, install_dir
    )
    try:
        return _install_pinned_build(build, install_dir)
    except PinnedDownloadError:
        raise
    except OSError as error:  # urllib.error.URLError is an OSError subclass
        raise PinnedDownloadError(
            f"failed to download the pinned ffmpeg build from {build.url}: {error}. "
            f"Retry with network access, or set {FFMPEG_ENV_VAR}=/path/to/ffmpeg "
            "to use your own binary."
        ) from error


def _which_with_pinning_warning(binary_name: str, override_env_var: str) -> Path | None:
    """PATH lookup used only where policy allows it, always with the loud warning."""
    path_binary = shutil.which(binary_name)
    if path_binary is None:
        return None
    logger.warning(
        "using %s from PATH (%s); measurements are only comparable across "
        "runs using the same build -- pin one via %s",
        binary_name,
        path_binary,
        override_env_var,
    )
    return Path(path_binary)


@lru_cache(maxsize=1)
def ffmpeg_path() -> Path:
    """Resolve the ffmpeg binary per the policy above."""
    # (1) Explicit override always wins.
    override = os.environ.get(FFMPEG_ENV_VAR)
    if override:
        override_path = Path(override)
        if not override_path.is_file():
            raise FfmpegNotFoundError(f"{FFMPEG_ENV_VAR}={override} does not exist")
        return override_path
    # (2) Linux: the pinned build (auto-downloaded), never PATH.
    if platform.system() == "Linux":
        return _resolve_pinned_linux_install_dir() / "ffmpeg"
    # (3) Non-Linux only: PATH fallback with a loud warning.
    path_binary = _which_with_pinning_warning("ffmpeg", FFMPEG_ENV_VAR)
    if path_binary is None:
        raise FfmpegNotFoundError(
            f"No ffmpeg found. Set {FFMPEG_ENV_VAR}=/path/to/ffmpeg or install ffmpeg "
            f"(no pinned build exists for {platform.system()})."
        )
    return path_binary


@lru_cache(maxsize=1)
def ffprobe_path() -> Path:
    """Resolve the ffprobe binary per the policy above."""
    # (1) Explicit override always wins.
    override = os.environ.get(FFPROBE_ENV_VAR)
    if override:
        override_path = Path(override)
        if not override_path.is_file():
            raise FfprobeNotFoundError(f"{FFPROBE_ENV_VAR}={override} does not exist")
        return override_path
    # (2) When the user pinned their own ffmpeg, the pinned-cache ffprobe would
    # be a mismatched instrument: prefer the sibling ffprobe shipped next to
    # their ffmpeg, falling back to PATH with the loud warning.
    ffmpeg_override = os.environ.get(FFMPEG_ENV_VAR)
    if ffmpeg_override:
        sibling_ffprobe = Path(ffmpeg_override).parent / "ffprobe"
        if sibling_ffprobe.is_file():
            return sibling_ffprobe
        path_binary = _which_with_pinning_warning("ffprobe", FFPROBE_ENV_VAR)
        if path_binary is None:
            raise FfprobeNotFoundError(
                f"{FFMPEG_ENV_VAR} is set but no ffprobe exists next to it "
                f"({sibling_ffprobe}) or on PATH. Set {FFPROBE_ENV_VAR}=/path/to/ffprobe."
            )
        return path_binary
    # (3) Linux: the pinned build (auto-downloaded), never PATH.
    if platform.system() == "Linux":
        return _resolve_pinned_linux_install_dir() / "ffprobe"
    # (4) Non-Linux only: PATH fallback with a loud warning.
    path_binary = _which_with_pinning_warning("ffprobe", FFPROBE_ENV_VAR)
    if path_binary is None:
        raise FfprobeNotFoundError(
            f"No ffprobe found. Set {FFPROBE_ENV_VAR}=/path/to/ffprobe or install ffprobe "
            f"(no pinned build exists for {platform.system()})."
        )
    return path_binary


def _first_version_line(binary: Path) -> str:
    completed = subprocess.run(
        [str(binary), "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.splitlines()[0].strip()


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    """First line of ``ffmpeg -version`` for the resolved binary."""
    return _first_version_line(ffmpeg_path())


@lru_cache(maxsize=1)
def ffprobe_version() -> str:
    """First line of ``ffprobe -version`` for the resolved binary."""
    return _first_version_line(ffprobe_path())
