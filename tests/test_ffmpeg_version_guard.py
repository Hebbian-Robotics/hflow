"""Regression tests for the suite-level ffmpeg version gate."""

from tests.conftest import _MIN_FFMPEG_VERSION, _ffmpeg_version_tuple


def test_parses_ubuntu_lts_and_git_n_prefix() -> None:
    assert _ffmpeg_version_tuple("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright") == (4, 4)
    assert _ffmpeg_version_tuple("ffmpeg version n5.1.2 Copyright") == (5, 1)
    assert _ffmpeg_version_tuple("ffmpeg version 6.1.1-3ubuntu5 Copyright") == (6, 1)


def test_old_distro_ffmpeg_is_below_fps_mode_floor() -> None:
    parsed = _ffmpeg_version_tuple("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright")
    assert parsed is not None
    assert parsed < _MIN_FFMPEG_VERSION
