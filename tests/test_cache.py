"""Behavior-focused tests for HFlow user cache resolution and migration."""

from pathlib import Path

import platformdirs
import platformdirs.macos
import platformdirs.unix
import pytest

from hflow._pinned_asset import user_cache_dir as pinned_user_cache_dir
from hflow.cache import migrate_legacy_cache, user_cache_dir, user_cache_root
from hflow.storage import _default_mirror_directory


def test_user_cache_path_default_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("platformdirs.PlatformDirs", platformdirs.macos.MacOS)

    root = user_cache_root()
    expected_root = fake_home / "Library" / "Caches" / "hflow"
    assert root == expected_root
    assert user_cache_dir() == expected_root
    assert user_cache_dir("ffmpeg") == expected_root / "ffmpeg"
    assert user_cache_dir("models") == expected_root / "models"
    assert user_cache_dir("mirrors") == expected_root / "mirrors"


def test_user_cache_path_default_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("platformdirs.PlatformDirs", platformdirs.unix.Unix)

    root = user_cache_root()
    expected_root = fake_home / ".cache" / "hflow"
    assert root == expected_root
    assert user_cache_dir() == expected_root
    assert user_cache_dir("ffmpeg") == expected_root / "ffmpeg"
    assert user_cache_dir("mirrors") == expected_root / "mirrors"


def test_user_cache_path_honors_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_xdg = tmp_path / "custom_xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(custom_xdg))

    for platform_class in (platformdirs.macos.MacOS, platformdirs.unix.Unix):
        monkeypatch.setattr("platformdirs.PlatformDirs", platform_class)
        expected_root = custom_xdg / "hflow"
        assert user_cache_root() == expected_root
        assert user_cache_dir("ffmpeg") == expected_root / "ffmpeg"
        assert user_cache_dir("mirrors") == expected_root / "mirrors"


def test_storage_mirror_directory_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_dir = tmp_path / "custom_mirrors"
    monkeypatch.setenv("HFLOW_MIRROR_DIR", str(override_dir))
    mirror_with_override = _default_mirror_directory("gs://robot-data/episodes")
    assert mirror_with_override.is_relative_to(override_dir)

    monkeypatch.delenv("HFLOW_MIRROR_DIR", raising=False)
    custom_xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(custom_xdg))
    mirror_without_override = _default_mirror_directory("gs://robot-data/episodes")
    assert mirror_without_override.is_relative_to(custom_xdg / "hflow" / "mirrors")


def test_pinned_asset_and_storage_share_same_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_xdg = tmp_path / "shared_xdg"
    monkeypatch.delenv("HFLOW_MIRROR_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(custom_xdg))

    pinned_dir = pinned_user_cache_dir("mirrors")
    storage_mirror = _default_mirror_directory("gs://bucket/path")
    assert storage_mirror.parent == pinned_dir


def test_legacy_cache_reuse_and_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("platformdirs.PlatformDirs", platformdirs.macos.MacOS)

    legacy_root = fake_home / ".cache" / "hflow"
    ffmpeg_dir = legacy_root / "ffmpeg" / "v0.2.14-arm64"
    ffmpeg_dir.mkdir(parents=True)
    (ffmpeg_dir / "ffmpeg").write_bytes(b"binary-ffmpeg-data")

    models_dir = legacy_root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "hand_landmarker.task").write_bytes(b"model-weights-data")

    mirror_sub = legacy_root / "mirrors" / "a1b2c3d4e5f6"
    mirror_sub.mkdir(parents=True)
    (mirror_sub / "events.parquet").write_bytes(b"parquet-mirror-data")

    native_root = fake_home / "Library" / "Caches" / "hflow"
    assert not native_root.exists()

    resolved_root = user_cache_root()
    assert resolved_root == native_root
    assert (
        native_root / "ffmpeg" / "v0.2.14-arm64" / "ffmpeg"
    ).read_bytes() == b"binary-ffmpeg-data"
    assert (native_root / "models" / "hand_landmarker.task").read_bytes() == b"model-weights-data"
    assert (
        native_root / "mirrors" / "a1b2c3d4e5f6" / "events.parquet"
    ).read_bytes() == b"parquet-mirror-data"
    assert not legacy_root.exists()


def test_legacy_cache_migration_merges_without_overwriting(
    tmp_path: Path,
) -> None:
    legacy_dir = tmp_path / "legacy" / "hflow"
    target_dir = tmp_path / "target" / "hflow"

    (legacy_dir / "ffmpeg").mkdir(parents=True)
    (legacy_dir / "ffmpeg" / "ffmpeg").write_bytes(b"old-ffmpeg")
    (legacy_dir / "ffmpeg" / "ffprobe").write_bytes(b"legacy-ffprobe")

    (target_dir / "ffmpeg").mkdir(parents=True)
    (target_dir / "ffmpeg" / "ffmpeg").write_bytes(b"existing-new-ffmpeg")

    migrate_legacy_cache(target_root=target_dir, legacy_root=legacy_dir)

    assert (target_dir / "ffmpeg" / "ffmpeg").read_bytes() == b"existing-new-ffmpeg"
    assert (target_dir / "ffmpeg" / "ffprobe").read_bytes() == b"legacy-ffprobe"


def test_legacy_cache_not_migrated_when_xdg_cache_home_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    custom_xdg = tmp_path / "custom_xdg"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(custom_xdg))
    monkeypatch.setattr("platformdirs.PlatformDirs", platformdirs.macos.MacOS)

    legacy_root = fake_home / ".cache" / "hflow"
    legacy_root.mkdir(parents=True)
    (legacy_root / "untouched_file").write_bytes(b"stay-here")

    resolved_root = user_cache_root()
    assert resolved_root == custom_xdg / "hflow"
    assert (legacy_root / "untouched_file").exists()
    assert (legacy_root / "untouched_file").read_bytes() == b"stay-here"
