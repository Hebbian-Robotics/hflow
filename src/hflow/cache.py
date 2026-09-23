"""User cache directory resolution and legacy cache migration for HFlow."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)


def _migrate_directory(source: Path, destination: Path) -> None:
    """Safely migrate files and subdirectories from source to destination.

    Moves entries when possible. If moving across devices or permissions fails,
    attempts copying. Empty source directories are removed after their contents
    are migrated.
    """
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    try:
        entries = list(source.iterdir())
    except OSError:
        return

    for entry in entries:
        target_entry = destination / entry.name
        if not target_entry.exists():
            try:
                shutil.move(str(entry), str(target_entry))
            except OSError:
                with contextlib.suppress(OSError):
                    if entry.is_dir():
                        shutil.copytree(str(entry), str(target_entry))
                        shutil.rmtree(str(entry), ignore_errors=True)
                    else:
                        shutil.copy2(str(entry), str(target_entry))
                        entry.unlink(missing_ok=True)
        elif entry.is_dir() and target_entry.is_dir():
            _migrate_directory(entry, target_entry)

    with contextlib.suppress(OSError):
        source.rmdir()


def migrate_legacy_cache(
    target_root: Path | None = None,
    legacy_root: Path | None = None,
) -> None:
    """Migrate legacy ~/.cache/hflow contents to the native user cache location.

    Only operates when ``XDG_CACHE_HOME`` is unset and the target cache path
    differs from the legacy path. Existing files in the target location are
    never overwritten.
    """
    if os.environ.get("XDG_CACHE_HOME"):
        return

    target = target_root if target_root is not None else platformdirs.user_cache_path("hflow")
    legacy = legacy_root if legacy_root is not None else Path.home() / ".cache" / "hflow"

    if target == legacy or not legacy.is_dir():
        return

    _migrate_directory(legacy, target)


def user_cache_root() -> Path:
    """Return the HFlow user cache root directory.

    Resolves via ``platformdirs.user_cache_path("hflow")``, honoring
    ``$XDG_CACHE_HOME`` if set, and otherwise using the platform-native cache
    directory (e.g. ``~/Library/Caches/hflow`` on macOS, ``~/.cache/hflow`` on Linux).

    When ``XDG_CACHE_HOME`` is unset and the platform-native cache differs from
    the legacy ``~/.cache/hflow`` directory, existing contents in the legacy
    location are safely migrated so upgrades do not re-download large binaries,
    models, or bucket mirrors.
    """
    target = platformdirs.user_cache_path("hflow")
    migrate_legacy_cache(target_root=target)
    return target


def user_cache_dir(component: str | None = None) -> Path:
    """Return the user cache path for HFlow or an optional component subdirectory.

    Parameters
    ----------
    component : str or None, optional
        A component subdirectory name (e.g., ``"ffmpeg"``, ``"models"``,
        ``"mirrors"``). If omitted, returns the cache root itself.
    """
    root = user_cache_root()
    if component is None:
        return root
    target = root / component
    if not target.exists() and not os.environ.get("XDG_CACHE_HOME"):
        legacy_component = Path.home() / ".cache" / "hflow" / component
        if legacy_component.exists():
            _migrate_directory(legacy_component, target)
            if target.exists():
                return target
            return legacy_component
    return target
