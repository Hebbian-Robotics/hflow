"""Pinned, digest-verified assets in the user cache.

An instrument's inputs are part of what its measurements MEAN -- the ffmpeg
build that decoded the frames, the weights a detector ran -- so hflow never
uses whatever happens to be on the machine. Each asset is pinned by URL and
sha256, downloaded once into the user cache, and verified before anything
reads it.

This module owns the mechanism only. Each caller keeps its own pins and its
own resolution policy (which override wins, whether a PATH fallback is
allowed), because those differ per instrument and are the interesting part.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


class PinnedAssetError(RuntimeError):
    """A pinned asset could not be fetched, or did not match its digest."""


def user_cache_dir(component: str) -> Path:
    """``<user cache>/hflow/<component>`` (respects ``XDG_CACHE_HOME``)."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_base = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return cache_base / "hflow" / component


def sha256_hex_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256_hex: str, url: str) -> None:
    """Raise unless ``path`` hashes to ``expected_sha256_hex``.

    Both hashes go in the message: the pin is the thing that was wrong or the
    bytes are, and the reader cannot tell which from one of them.
    """
    actual_sha256_hex = sha256_hex_of_file(path)
    if actual_sha256_hex != expected_sha256_hex:
        raise PinnedAssetError(
            f"sha256 mismatch for {url}: expected {expected_sha256_hex}, "
            f"got {actual_sha256_hex}. The pinned release asset should be immutable; "
            "this indicates a corrupted download or a tampered mirror."
        )


def download_url_to_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as destination_file:
        shutil.copyfileobj(response, destination_file)


def download_verified_asset(url: str, sha256_hex: str, destination: Path) -> Path:
    """Download to ``destination`` once, verified, and publish it atomically.

    The staging directory sits inside ``destination``'s parent so the final
    rename is a same-filesystem move: a concurrent second process sees either
    no file or a complete verified one, never a partial write. Unverified
    bytes never occupy the destination path even briefly.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".download-"
    ) as staging_dir_name:
        staged_asset = Path(staging_dir_name) / destination.name
        try:
            download_url_to_file(url, staged_asset)
        except OSError as error:  # urllib.error.URLError is an OSError subclass
            raise PinnedAssetError(f"failed to download {url}: {error}") from error
        verify_sha256(staged_asset, sha256_hex, url)
        staged_asset.replace(destination)
    return destination
