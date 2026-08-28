"""Tests for the ffmpeg helpers: pinned binary, frame instrument, contact sheet.

The suite conftest pins ``HFLOW_FFMPEG`` to the system binary, so the
resolution tests below must clear the ``lru_cache`` around any environment
mutation. The one real-download test is opt-in via ``HFLOW_NETWORK_TESTS=1``.
"""

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from hflow._video_measurement_toolchain import (
    _frame_statistics_cache_path,
    measure_video_frame_statistics_for_hflow,
    resolved_video_measurement_toolchain,
)
from hflow._video_measurements import (
    FrameStatisticsExecutionError,
    FrameStatisticsParseError,
    FrameStatisticsSettings,
    LumaRangeEvidence,
    VideoFrameStatistics,
    VideoMeasurementToolchain,
    measure_video_frame_statistics,
)
from hflow._video_measurements._frame_statistics import (
    _FILTER_LIST_ENTRY_PATTERN,
    _aggregate_frame_statistics_output,
)
from hflow._video_measurements._raw_frames import (
    RawFrameError,
    _scaled_frame_shape,
    luma_frames,
    rgb_frames,
)
from hflow.episode import ExtractedFrame
from hflow.ffmpeg import _binary, _contact_sheet
from hflow.ffmpeg._binary import (
    FFMPEG_ENV_VAR,
    FFPROBE_ENV_VAR,
    PINNED_VERSION_LABEL,
    FfmpegNotFoundError,
    FfprobeNotFoundError,
    PinnedBuild,
    PinnedDownloadError,
    _install_pinned_build,
    _pinned_install_dir,
    ffmpeg_path,
    ffmpeg_version,
    ffprobe_path,
    ffprobe_version,
)
from hflow.ffmpeg._contact_sheet import (
    _find_usable_font_file,
    contact_sheet,
)


def _system_ffmpeg() -> str:
    return os.environ[FFMPEG_ENV_VAR]


def _measure_frame_statistics(
    video: Path,
    *,
    freeze_minimum_duration_seconds: float = 2.0,
    overexposed_average_luma_threshold: float = 235.0,
) -> VideoFrameStatistics:
    return measure_video_frame_statistics(
        video,
        toolchain=resolved_video_measurement_toolchain(),
        settings=FrameStatisticsSettings(
            freeze_minimum_duration_seconds=freeze_minimum_duration_seconds,
            overexposed_average_luma_threshold=overexposed_average_luma_threshold,
        ),
    )


def _clear_all_binary_caches() -> None:
    ffmpeg_path.cache_clear()
    ffmpeg_version.cache_clear()
    ffprobe_path.cache_clear()
    ffprobe_version.cache_clear()


@pytest.fixture
def cleared_binary_caches() -> Iterator[None]:
    _clear_all_binary_caches()
    yield None
    _clear_all_binary_caches()


def test_env_override_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleared_binary_caches: None
) -> None:
    override_ffmpeg = tmp_path / "ffmpeg"
    override_ffmpeg.touch()
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(override_ffmpeg))
    assert ffmpeg_path() == override_ffmpeg


def test_env_override_to_missing_file_raises(
    monkeypatch: pytest.MonkeyPatch, cleared_binary_caches: None
) -> None:
    monkeypatch.setenv(FFMPEG_ENV_VAR, "/nonexistent/ffmpeg")
    with pytest.raises(FfmpegNotFoundError, match="does not exist"):
        ffmpeg_path()


_FAKE_ARCHIVE_ROOT = "ffmpeg-fake-build"


def _build_fake_release_archive(directory: Path) -> tuple[PinnedBuild, str]:
    """A local tar.xz shaped like a BtbN release asset (ffmpeg AND ffprobe in
    ``bin/``), plus its true sha256."""
    fake_ffmpeg = directory / "fake-ffmpeg-script"
    fake_ffmpeg.write_text("#!/bin/sh\necho fake-ffmpeg-ok\n")
    fake_ffprobe = directory / "fake-ffprobe-script"
    fake_ffprobe.write_text("#!/bin/sh\necho fake-ffprobe-ok\n")
    archive = directory / "fake-release.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(fake_ffmpeg, arcname=f"{_FAKE_ARCHIVE_ROOT}/bin/ffmpeg")
        tar.add(fake_ffprobe, arcname=f"{_FAKE_ARCHIVE_ROOT}/bin/ffprobe")
    true_sha256_hex = hashlib.sha256(archive.read_bytes()).hexdigest()
    build = PinnedBuild(
        url=archive.as_uri(),
        sha256_hex=true_sha256_hex,
        archive_bin_dir=f"{_FAKE_ARCHIVE_ROOT}/bin",
    )
    return build, true_sha256_hex


def test_sha256_mismatch_names_both_hashes(tmp_path: Path) -> None:
    good_build, true_sha256_hex = _build_fake_release_archive(tmp_path)
    wrong_sha256_hex = "0" * 64
    tampered_build = PinnedBuild(
        url=good_build.url,
        sha256_hex=wrong_sha256_hex,
        archive_bin_dir=good_build.archive_bin_dir,
    )
    install_dir = tmp_path / "install"
    with pytest.raises(PinnedDownloadError) as exc_info:
        _install_pinned_build(tampered_build, install_dir)
    assert wrong_sha256_hex in str(exc_info.value)
    assert true_sha256_hex in str(exc_info.value)
    assert not (install_dir / "ffmpeg").exists()
    assert not (install_dir / "ffprobe").exists()


def test_local_install_extracts_both_binaries_and_executes(tmp_path: Path) -> None:
    build, _ = _build_fake_release_archive(tmp_path)
    install_dir = tmp_path / "install"
    installed_dir = _install_pinned_build(build, install_dir)
    assert installed_dir == install_dir
    expected_outputs_by_binary_name = {"ffmpeg": "fake-ffmpeg-ok", "ffprobe": "fake-ffprobe-ok"}
    for binary_name, expected_output in expected_outputs_by_binary_name.items():
        binary_path = install_dir / binary_name
        assert binary_path.is_file()
        assert os.access(binary_path, os.X_OK)
        completed = subprocess.run([str(binary_path)], capture_output=True, text=True, check=True)
        assert completed.stdout.strip() == expected_output
    # Reinstalling over an existing install must succeed (atomic os.replace).
    _install_pinned_build(build, install_dir)
    assert (install_dir / "ffmpeg").is_file()
    assert (install_dir / "ffprobe").is_file()


@pytest.mark.skipif(platform.system() != "Linux", reason="pinned-build path is Linux-only")
def test_pinned_build_resolution_without_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleared_binary_caches: None
) -> None:
    build, _ = _build_fake_release_archive(tmp_path)
    monkeypatch.delenv(FFMPEG_ENV_VAR, raising=False)
    monkeypatch.delenv(FFPROBE_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(_binary, "PINNED_BUILDS_BY_MACHINE", {platform.machine(): build})
    resolved = ffmpeg_path()
    assert resolved.is_relative_to(tmp_path / "cache")
    assert os.access(resolved, os.X_OK)
    # One download installs BOTH binaries into the same versioned dir.
    installed_ffprobe = resolved.with_name("ffprobe")
    assert os.access(installed_ffprobe, os.X_OK)
    # A second resolution against the populated cache must not reinstall:
    # deleting the source archive proves the cached binaries are used as-is.
    Path(build.url.removeprefix("file://")).unlink()
    ffmpeg_path.cache_clear()
    assert ffmpeg_path() == resolved
    assert ffprobe_path() == installed_ffprobe


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux never falls back to PATH")
def test_linux_unsupported_machine_raises_instead_of_path_fallback(
    monkeypatch: pytest.MonkeyPatch, cleared_binary_caches: None
) -> None:
    monkeypatch.delenv(FFMPEG_ENV_VAR, raising=False)
    monkeypatch.setattr(_binary, "PINNED_BUILDS_BY_MACHINE", {})
    # ffmpeg IS on PATH here; on Linux we still must refuse to use it.
    assert shutil.which("ffmpeg") is not None
    with pytest.raises(FfmpegNotFoundError, match=FFMPEG_ENV_VAR):
        ffmpeg_path()


def test_ffprobe_env_override_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleared_binary_caches: None
) -> None:
    override_ffprobe = tmp_path / "my-ffprobe"
    override_ffprobe.write_text("#!/bin/sh\n")
    monkeypatch.setenv(FFPROBE_ENV_VAR, str(override_ffprobe))
    assert ffprobe_path() == override_ffprobe


def test_ffprobe_env_override_to_missing_file_raises(
    monkeypatch: pytest.MonkeyPatch, cleared_binary_caches: None
) -> None:
    monkeypatch.setenv(FFPROBE_ENV_VAR, "/nonexistent/ffprobe")
    with pytest.raises(FfprobeNotFoundError, match="does not exist"):
        ffprobe_path()


def test_ffprobe_prefers_sibling_of_ffmpeg_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleared_binary_caches: None
) -> None:
    # A user-managed build ships its own matching ffprobe next to ffmpeg;
    # that sibling must beat both the pinned cache and PATH.
    user_bin_dir = tmp_path / "user-build" / "bin"
    user_bin_dir.mkdir(parents=True)
    (user_bin_dir / "ffmpeg").write_text("#!/bin/sh\n")
    sibling_ffprobe = user_bin_dir / "ffprobe"
    sibling_ffprobe.write_text("#!/bin/sh\n")
    monkeypatch.delenv(FFPROBE_ENV_VAR, raising=False)
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(user_bin_dir / "ffmpeg"))
    assert ffprobe_path() == sibling_ffprobe


def test_ffprobe_without_sibling_falls_back_to_path_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    cleared_binary_caches: None,
) -> None:
    lone_ffmpeg_dir = tmp_path / "lone"
    lone_ffmpeg_dir.mkdir()
    (lone_ffmpeg_dir / "ffmpeg").write_text("#!/bin/sh\n")
    monkeypatch.delenv(FFPROBE_ENV_VAR, raising=False)
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(lone_ffmpeg_dir / "ffmpeg"))
    system_ffprobe = shutil.which("ffprobe")
    assert system_ffprobe is not None, "ffprobe required on PATH for tests"
    with caplog.at_level(logging.WARNING, logger="hflow.ffmpeg._binary"):
        resolved = ffprobe_path()
    assert resolved == Path(system_ffprobe)
    warning_messages = [record.getMessage() for record in caplog.records]
    assert any(FFPROBE_ENV_VAR in message and "PATH" in message for message in warning_messages)


def test_ffprobe_without_sibling_or_path_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleared_binary_caches: None
) -> None:
    lone_ffmpeg_dir = tmp_path / "lone"
    lone_ffmpeg_dir.mkdir()
    (lone_ffmpeg_dir / "ffmpeg").write_text("#!/bin/sh\n")
    monkeypatch.delenv(FFPROBE_ENV_VAR, raising=False)
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(lone_ffmpeg_dir / "ffmpeg"))
    monkeypatch.setattr(shutil, "which", lambda _binary_name: None)
    with pytest.raises(FfprobeNotFoundError, match=FFPROBE_ENV_VAR):
        ffprobe_path()


@pytest.mark.skipif(platform.system() != "Linux", reason="pinned-build path is Linux-only")
def test_cache_dir_with_only_ffmpeg_is_healed_by_reinstall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleared_binary_caches: None
) -> None:
    """Older installs put a lone ffmpeg in the versioned dir; resolution must
    re-extract BOTH binaries from a fresh download (completion = both present)."""
    build, _ = _build_fake_release_archive(tmp_path)
    monkeypatch.delenv(FFMPEG_ENV_VAR, raising=False)
    monkeypatch.delenv(FFPROBE_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(_binary, "PINNED_BUILDS_BY_MACHINE", {platform.machine(): build})
    install_dir = _pinned_install_dir(platform.machine())
    install_dir.mkdir(parents=True)
    stale_ffmpeg_contents = "#!/bin/sh\necho stale-old-layout\n"
    (install_dir / "ffmpeg").write_text(stale_ffmpeg_contents)
    resolved_ffprobe = ffprobe_path()
    assert resolved_ffprobe == install_dir / "ffprobe"
    completed = subprocess.run([str(resolved_ffprobe)], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "fake-ffprobe-ok"
    # The heal reinstalled ffmpeg from the fresh archive too, not just ffprobe.
    assert (install_dir / "ffmpeg").read_text() != stale_ffmpeg_contents
    assert ffmpeg_path() == install_dir / "ffmpeg"


@pytest.mark.skipif(
    os.environ.get("HFLOW_NETWORK_TESTS") != "1",
    reason="network integration test; set HFLOW_NETWORK_TESTS=1 to run",
)
@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"),
    reason="pinned builds exist for Linux x86_64/aarch64 only",
)
def test_real_pinned_download_and_version(
    monkeypatch: pytest.MonkeyPatch, cleared_binary_caches: None
) -> None:
    monkeypatch.delenv(FFMPEG_ENV_VAR, raising=False)
    monkeypatch.delenv(FFPROBE_ENV_VAR, raising=False)
    resolved = ffmpeg_path()
    assert resolved.is_file()
    version_line = ffmpeg_version()
    assert PINNED_VERSION_LABEL in version_line
    # The same download must have produced a working ffprobe alongside.
    resolved_ffprobe = ffprobe_path()
    assert resolved_ffprobe == resolved.with_name("ffprobe")
    ffprobe_version_line = ffprobe_version()
    assert PINNED_VERSION_LABEL in ffprobe_version_line


def _synthetic_frames(
    *frame_signals: dict[str, float | str],
    interval_s: float = 0.5,
) -> str:
    """Instrument stdout for frames declaring only the signals a test is about.

    Every signal the parser hard-requires gets a benign limited-range default,
    so a test reads as the one fact it pins rather than a wall of keys -- and a
    signal added to the instrument later needs one default here, not an edit to
    every fixture.
    """
    defaults: dict[str, float | str] = {
        "lavfi.blackframe.pblack": 0,
        "lavfi.signalstats.YMIN": 16,
        "lavfi.signalstats.YLOW": 32,
        "lavfi.signalstats.YAVG": 100,
        "lavfi.signalstats.YHIGH": 200,
        "lavfi.signalstats.YMAX": 235,
        "lavfi.signalstats.YDIF": 0,
        "lavfi.signalstats.TOUT": 0,
        "lavfi.signalstats.BRNG": 0,
    }
    blocks: list[str] = []
    for frame_index, signals in enumerate(frame_signals):
        pts_time_s = frame_index * interval_s
        lines = [f"frame:{frame_index}    pts:{frame_index * 512}    pts_time:{pts_time_s}"]
        lines.extend(f"{key}={value}" for key, value in (defaults | signals).items())
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


_SYNTHETIC_OUTPUT = _synthetic_frames(
    {"lavfi.signalstats.YAVG": 100},
    {"lavfi.signalstats.YAVG": 10, "lavfi.blackframe.pblack": 99.5},
    {"lavfi.signalstats.YAVG": 40},
)


def test_synthetic_output_aggregates_exactly() -> None:
    stats = _aggregate_frame_statistics_output(_SYNTHETIC_OUTPUT)
    assert stats.decoded_frame_count == 3
    assert stats.duration_seconds == pytest.approx(1.5)  # last pts + median interval
    assert stats.black_frame_count == 1
    assert stats.black_frame_percent == pytest.approx(100.0 / 3.0)
    assert stats.average_luma_mean == pytest.approx(50.0)
    assert stats.average_luma_minimum == pytest.approx(10.0)
    assert stats.average_luma_maximum == pytest.approx(100.0)
    assert stats.freeze_intervals == ()
    assert stats.freeze_total_seconds == 0.0
    assert type(stats.freeze_total_seconds) is float
    # Default threshold 235.0: none of 100, 10, 40 are >= 235
    assert stats.overexposed_frame_count == 0
    assert stats.overexposed_frame_percent == 0.0


def test_synthetic_overexposed_aggregates_exactly() -> None:
    # YAVG: 100 (normal), 240 (overexposed), 250 (overexposed) at default threshold 235.0
    output = _synthetic_frames(
        {"lavfi.signalstats.YAVG": 100},
        {"lavfi.signalstats.YAVG": 240},
        {"lavfi.signalstats.YAVG": 250},
    )
    stats = _aggregate_frame_statistics_output(output)
    assert stats.decoded_frame_count == 3
    assert stats.overexposed_frame_count == 2
    assert stats.overexposed_frame_percent == pytest.approx(66.666, abs=0.1)
    assert stats.average_luma_minimum == pytest.approx(100.0)
    assert stats.average_luma_maximum == pytest.approx(250.0)


def test_synthetic_freeze_intervals_including_unterminated() -> None:
    output_text = _synthetic_frames(
        {"lavfi.signalstats.YAVG": 1},
        {"lavfi.signalstats.YAVG": 1, "lavfi.freezedetect.freeze_start": 0.5},
        {
            "lavfi.signalstats.YAVG": 1,
            "lavfi.freezedetect.freeze_duration": 1.5,
            "lavfi.freezedetect.freeze_end": 2,
        },
        {"lavfi.signalstats.YAVG": 1, "lavfi.freezedetect.freeze_start": 2.5},
        interval_s=1.0,
    )
    stats = _aggregate_frame_statistics_output(output_text)
    # Unterminated freeze closes at the duration (3.0 + 1.0 median interval).
    assert stats.duration_seconds == pytest.approx(4.0)
    assert [
        (interval.start_seconds, interval.end_seconds) for interval in stats.freeze_intervals
    ] == [(0.5, 2.0), (2.5, 4.0)]
    assert stats.freeze_total_seconds == pytest.approx(3.0)


def test_truncated_output_missing_yavg_raises() -> None:
    truncated = _synthetic_frames({}, {}).replace("lavfi.signalstats.YAVG=100\n", "", 1)
    with pytest.raises(FrameStatisticsParseError, match="YAVG"):
        _aggregate_frame_statistics_output(truncated)


def test_unparsable_line_raises() -> None:
    garbled = "frame:0    pts:0    pts_time:0\nlavfi.signalstats.YA\n"
    with pytest.raises(FrameStatisticsParseError, match="unparsable"):
        _aggregate_frame_statistics_output(garbled)


def test_nan_yavg_raises() -> None:
    nan_output = "frame:0    pts:0    pts_time:0\nlavfi.signalstats.YAVG=nan\n"
    with pytest.raises(FrameStatisticsParseError, match="non-finite"):
        _aggregate_frame_statistics_output(nan_output)


def test_empty_output_raises() -> None:
    with pytest.raises(FrameStatisticsParseError, match="no frames"):
        _aggregate_frame_statistics_output("")


@pytest.mark.parametrize(
    ("filter_listing_line", "expected_filter_name"),
    [
        (" T.. freezedetect      V->V       Detects frozen video input.", "freezedetect"),
        (" T. freezedetect       V->V      Detects frozen video input.", "freezedetect"),
    ],
)
def test_filter_listing_parser_accepts_supported_ffmpeg_layouts(
    filter_listing_line: str, expected_filter_name: str
) -> None:
    match = _FILTER_LIST_ENTRY_PATTERN.match(filter_listing_line)
    assert match is not None
    assert match.group("filter_name") == expected_filter_name


def test_invalid_freeze_interval_is_a_parse_error() -> None:
    output_text = _synthetic_frames(
        {"lavfi.freezedetect.freeze_start": 2.0},
        {"lavfi.freezedetect.freeze_end": 1.0},
    )
    with pytest.raises(FrameStatisticsParseError, match="invalid freeze interval"):
        _aggregate_frame_statistics_output(output_text)


@pytest.fixture(scope="module")
def black_tail_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4s of testsrc2 followed by 2s of black, 10 fps, h264: 60 frames total."""
    output = tmp_path_factory.mktemp("videos") / "black_tail.mp4"
    subprocess.run(
        [
            _system_ffmpeg(),
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=4,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=160x120:rate=10:duration=2,format=yuv420p",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(output),
        ],
        capture_output=True,
        check=True,
    )
    return output


@pytest.fixture(scope="module")
def bright_tail_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4s of testsrc2 followed by 2s of white, 10 fps, h264: 60 frames total."""
    output = tmp_path_factory.mktemp("videos") / "bright_tail.mp4"
    subprocess.run(
        [
            _system_ffmpeg(),
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=4,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "color=white:size=160x120:rate=10:duration=2,format=yuv420p",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(output),
        ],
        capture_output=True,
        check=True,
    )
    return output


@pytest.fixture(scope="module")
def frozen_tail_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2s of testsrc2 then the last frame held (cloned) for 3s, 10 fps."""
    output = tmp_path_factory.mktemp("videos") / "frozen_tail.mp4"
    subprocess.run(
        [
            _system_ffmpeg(),
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=2,format=yuv420p",
            "-vf",
            "tpad=stop_mode=clone:stop_duration=3",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(output),
        ],
        capture_output=True,
        check=True,
    )
    return output


def test_frame_stats_black_segment(black_tail_video: Path) -> None:
    # freeze_min_duration_s > the 2s black segment so no freeze fires here.
    stats = _measure_frame_statistics(black_tail_video, freeze_minimum_duration_seconds=3.0)
    assert stats.decoded_frame_count == 60
    assert stats.duration_seconds == pytest.approx(6.0, abs=0.3)
    # 20 of 60 frames are black; allow encoder edge effects.
    assert stats.black_frame_percent == pytest.approx(100.0 * 20 / 60, abs=7.0)
    # Encoded limited-range black lands around YAVG=16.
    assert stats.average_luma_minimum < 32.0
    assert stats.average_luma_minimum <= stats.average_luma_mean <= stats.average_luma_maximum
    assert stats.average_luma_maximum > 64.0
    assert stats.freeze_intervals == ()
    # Default threshold 235.0: testsrc2 max YAVG is well below.
    assert stats.overexposed_frame_count == 0
    assert stats.overexposed_frame_percent == 0.0
    assert stats.provenance.measurement_definition_version == "video-frame-statistics/v1"
    assert "blackframe=amount=0" in stats.provenance.filter_graph
    assert stats.provenance.settings.freeze_minimum_duration_seconds == 3.0


def test_frame_stats_bright_segment(bright_tail_video: Path) -> None:
    # freeze_min_duration_s > the 2s white segment so no freeze fires here.
    # Use a lower threshold to catch the white segment (limited-range white ~235).
    stats = _measure_frame_statistics(
        bright_tail_video,
        freeze_minimum_duration_seconds=3.0,
        overexposed_average_luma_threshold=200.0,
    )
    assert stats.decoded_frame_count == 60
    assert stats.duration_seconds == pytest.approx(6.0, abs=0.3)
    # 20 of 60 frames are white/overexposed; allow encoder edge effects.
    assert 10.0 < stats.overexposed_frame_percent < 50.0
    # Encoded limited-range white lands near YAVG=235.
    assert stats.average_luma_maximum > 200.0
    assert stats.average_luma_minimum <= stats.average_luma_mean <= stats.average_luma_maximum
    assert stats.freeze_intervals == ()


def test_frame_stats_freeze_interval(frozen_tail_video: Path) -> None:
    stats = _measure_frame_statistics(frozen_tail_video, freeze_minimum_duration_seconds=1.0)
    assert stats.decoded_frame_count == 50
    assert stats.duration_seconds == pytest.approx(5.0, abs=0.3)
    assert len(stats.freeze_intervals) == 1
    freeze_interval = stats.freeze_intervals[0]
    # The still segment spans roughly 2s..5s (the held frame displays from 1.9s).
    assert 1.5 <= freeze_interval.start_seconds <= 2.6
    assert 4.4 <= freeze_interval.end_seconds <= 5.4
    assert 2.0 <= stats.freeze_total_seconds <= 3.6


def test_frame_stats_truncated_video_file_raises(tmp_path: Path) -> None:
    not_a_video = tmp_path / "garbage.mp4"
    not_a_video.write_bytes(b"\x00\x01\x02not a video")
    with pytest.raises(RuntimeError):
        _measure_frame_statistics(not_a_video)


def test_luma_frames_streams_every_frame_at_the_coded_size(black_tail_video: Path) -> None:
    """Full rate and no re-encode, which is what a frame-to-frame measurement
    needs and what ``Episode.frames()`` deliberately does not give.
    """
    with luma_frames(black_tail_video, toolchain=resolved_video_measurement_toolchain()) as frames:
        shapes = [frame.shape for frame in frames]
    assert len(shapes) == 60
    assert set(shapes) == {(120, 160)}


def test_luma_frames_reaps_ffmpeg_when_the_caller_stops_early(
    black_tail_video: Path,
) -> None:
    """Abandoning the iterator closes the pipe, which makes ffmpeg fail its
    write. That is not a decode failure and must not be raised as one -- and the
    exit code cannot tell the two apart, so the helper tracks whether the stream
    was drained instead.
    """
    with luma_frames(black_tail_video, toolchain=resolved_video_measurement_toolchain()) as frames:
        first_frame = next(iter(frames))
    assert first_frame.shape == (120, 160)


def test_luma_frames_on_a_non_video_raises(tmp_path: Path) -> None:
    not_a_video = tmp_path / "garbage.mp4"
    not_a_video.write_bytes(b"\x00\x01\x02not a video")
    with (
        pytest.raises(RawFrameError),
        luma_frames(not_a_video, toolchain=resolved_video_measurement_toolchain()) as frames,
    ):
        list(frames)


def test_rgb_frames_streams_three_channels_at_the_coded_size(black_tail_video: Path) -> None:
    with rgb_frames(black_tail_video, toolchain=resolved_video_measurement_toolchain()) as frames:
        shapes = [frame.shape for frame in frames]
    assert len(shapes) == 60
    assert set(shapes) == {(120, 160, 3)}


def test_rgb_frames_resamples_and_resizes_in_one_decode(black_tail_video: Path) -> None:
    """A local model wants few frames at its own size; doing both in the filter
    graph converts one frame's pixels per output tick rather than every
    frame's.
    """
    with rgb_frames(
        black_tail_video,
        toolchain=resolved_video_measurement_toolchain(),
        frames_per_second=1.0,
        long_edge_pixels=80,
    ) as frames:
        shapes = [frame.shape for frame in frames]
    # 6 seconds at 10 fps sampled to 1 fps, and 160x120 scaled to an 80 long edge.
    assert len(shapes) == 6
    assert set(shapes) == {(60, 80, 3)}


def test_rgb_frames_are_contiguous_uint8(black_tail_video: Path) -> None:
    """The layout a local vision model takes directly; a non-contiguous or
    wider dtype would have to be copied and converted at every call site.
    """
    with rgb_frames(black_tail_video, toolchain=resolved_video_measurement_toolchain()) as frames:
        frame = next(iter(frames))
    assert frame.dtype == np.uint8
    assert frame.flags["C_CONTIGUOUS"]


def test_rgb_frames_reject_a_non_positive_rate(black_tail_video: Path) -> None:
    with (
        pytest.raises(ValueError, match="frames_per_second must be positive"),
        rgb_frames(
            black_tail_video,
            toolchain=resolved_video_measurement_toolchain(),
            frames_per_second=0.0,
        ),
    ):
        pass


@pytest.mark.parametrize(
    ("source_shape", "long_edge_pixels", "expected"),
    [
        ((120, 160), None, (120, 160)),  # unset leaves the coded size alone
        ((120, 160), 80, (60, 80)),
        ((160, 120), 80, (80, 60)),  # portrait: the long edge is the height
        ((120, 160), 320, (240, 320)),  # upscaling is allowed on purpose
        ((99, 160), 80, (50, 80)),  # 49.5 rounds half UP, not to even
    ],
)
def test_scaled_frame_shape_is_proportional_and_rounds_half_up(
    source_shape: tuple[int, int], long_edge_pixels: int | None, expected: tuple[int, int]
) -> None:
    assert _scaled_frame_shape(source_shape, long_edge_pixels) == expected


def test_scaled_frame_shape_refuses_a_degenerate_long_edge() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _scaled_frame_shape((120, 160), 1)


def test_missing_motion_extra_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented promise for the one built-in with an optional dependency:
    it fails with the command that fixes it, not a bare ImportError.

    Mocked at the import boundary, which is where the failure lives -- the suite
    installs the extra, so this case is otherwise unreachable.
    """
    import builtins

    from hflow._video_measurements import MotionExtraNotInstalledError
    from hflow._video_measurements._camera_motion import _import_cv2

    real_import = builtins.__import__

    def refuse_cv2(name: str, *arguments: object, **keywords: object) -> object:
        if name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'")
        return real_import(name, *arguments, **keywords)  # ty: ignore

    monkeypatch.setattr(builtins, "__import__", refuse_cv2)
    with pytest.raises(MotionExtraNotInstalledError, match=r"hflow\[motion\]"):
        _import_cv2()


def _probe_dimensions(image: Path) -> tuple[int, int]:
    ffprobe_binary = shutil.which("ffprobe")
    assert ffprobe_binary is not None, "ffprobe required on PATH for tests"
    completed = subprocess.run(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(image),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width_text, height_text = completed.stdout.strip().split(",")
    return int(width_text), int(height_text)


@pytest.fixture(scope="module")
def ten_extracted_frames(tmp_path_factory: pytest.TempPathFactory) -> list[ExtractedFrame]:
    """Ten 320x240 jpegs extracted from a testsrc2 clip, 1s apart."""
    directory = tmp_path_factory.mktemp("frames")
    subprocess.run(
        [
            _system_ffmpeg(),
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=1:duration=10",
            "-q:v",
            "2",
            str(directory / "frame_%02d.jpg"),
        ],
        capture_output=True,
        check=True,
    )
    frame_paths = sorted(directory.glob("frame_*.jpg"))
    assert len(frame_paths) == 10
    stream_start_ns = 1_755_000_000_000_000_000
    return [
        ExtractedFrame(path=frame_path, log_time_ns=stream_start_ns + index * 1_000_000_000)
        for index, frame_path in enumerate(frame_paths)
    ]


def test_contact_sheet_grid_geometry(
    ten_extracted_frames: list[ExtractedFrame], tmp_path: Path
) -> None:
    output = tmp_path / "sheet.jpg"
    sheet = contact_sheet(ten_extracted_frames, output, columns=4, tile_width=320)
    assert sheet.path == output
    assert output.is_file()
    assert sheet.columns == 4
    assert sheet.rows == 3  # ceil(10 / 4)
    assert sheet.frames_sampled_from == 10
    assert sheet.tile_log_times_ns == [frame.log_time_ns for frame in ten_extracted_frames]
    width, height = _probe_dimensions(output)
    assert width == 4 * 320
    assert height == 3 * 240  # 320x240 sources scaled to width 320 keep height 240
    assert sheet.timestamps_burned == (
        _find_usable_font_file() is not None
        and _contact_sheet._ffmpeg_supports_drawtext(ffmpeg_path())
    )


def test_contact_sheet_max_tiles_sampling(
    ten_extracted_frames: list[ExtractedFrame], tmp_path: Path
) -> None:
    output = tmp_path / "sampled_sheet.jpg"
    sheet = contact_sheet(ten_extracted_frames, output, columns=4, tile_width=160, max_tiles=6)
    assert output.is_file()
    assert sheet.frames_sampled_from == 10
    assert len(sheet.tile_log_times_ns) == 6
    assert sheet.tile_log_times_ns[0] == ten_extracted_frames[0].log_time_ns
    assert sheet.tile_log_times_ns[-1] == ten_extracted_frames[-1].log_time_ns
    assert sheet.tile_log_times_ns == sorted(set(sheet.tile_log_times_ns))
    assert sheet.rows == 2  # ceil(6 / 4); tile pads the two empty cells
    width, _height = _probe_dimensions(output)
    assert width == 4 * 160


def test_contact_sheet_without_drawtext_still_produces_sheet(
    ten_extracted_frames: list[ExtractedFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _contact_sheet,
        "_find_usable_font_file",
        lambda: Path("/a/usable/font.ttf"),
    )
    monkeypatch.setattr(
        _contact_sheet,
        "_ffmpeg_supports_drawtext",
        lambda _ffmpeg_binary: False,
    )

    output = tmp_path / "sheet-without-drawtext.jpg"
    sheet = contact_sheet(ten_extracted_frames[:2], output, columns=2)

    assert output.is_file()
    assert _probe_dimensions(output) == (640, 240)
    assert sheet.timestamps_burned is False


def test_contact_sheet_accepts_apostrophes_in_external_paths(
    ten_extracted_frames: list[ExtractedFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_font_path = _find_usable_font_file()
    if system_font_path is None:
        pytest.skip("requires a usable system font")
    if not _contact_sheet._ffmpeg_supports_drawtext(ffmpeg_path()):
        pytest.skip("requires ffmpeg drawtext support")

    external_path_directory = tmp_path / "wearer's assets"
    external_path_directory.mkdir()
    copied_frames: list[ExtractedFrame] = []
    for frame_index, source_frame in enumerate(ten_extracted_frames[:2]):
        copied_path = external_path_directory / f"frame-{frame_index}.jpg"
        copied_path.write_bytes(source_frame.path.read_bytes())
        copied_frames.append(ExtractedFrame(path=copied_path, log_time_ns=source_frame.log_time_ns))

    copied_font_path = external_path_directory / "worker's-font.ttf"
    copied_font_path.write_bytes(system_font_path.read_bytes())
    monkeypatch.setattr(
        _contact_sheet,
        "_find_usable_font_file",
        lambda: copied_font_path,
    )

    output = tmp_path / "quoted-path-sheet.jpg"
    sheet = contact_sheet(copied_frames, output, columns=2)

    assert output.is_file()
    assert _probe_dimensions(output) == (640, 240)
    assert sheet.timestamps_burned is True


def _render_with_font(
    frames: list[ExtractedFrame],
    font_path: Path,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    monkeypatch.setattr(_contact_sheet, "_find_usable_font_file", lambda: font_path)
    contact_sheet(frames, output, columns=2)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def test_an_apostrophe_font_path_reaches_drawtext_as_the_named_file(
    ten_extracted_frames: list[ExtractedFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escaping test above cannot fail on a host that quietly substitutes.

    When ffmpeg cannot open the ``fontfile`` it names, drawtext falls back to a
    default face and still reports ``timestamps_burned``. On this machine a
    deliberately missing path renders byte-identically to the real one, so every
    assertion above holds while the font path never arrived. Rendering a face
    that is visibly not the default is what makes the quoting observable.
    """
    if _find_usable_font_file() is None:
        pytest.skip("requires a usable system font")
    if not _contact_sheet._ffmpeg_supports_drawtext(ffmpeg_path()):
        pytest.skip("requires ffmpeg drawtext support")
    distinctive_font = next(
        (
            candidate
            for candidate in (
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
                Path("/usr/share/fonts/dejavu/DejaVuSerif.ttf"),
                Path("/usr/share/fonts/TTF/DejaVuSerif.ttf"),
                Path("/System/Library/Fonts/Times.ttc"),
            )
            if candidate.is_file()
        ),
        None,
    )
    if distinctive_font is None:
        pytest.skip("requires a second font distinguishable from the default")

    plain_directory = tmp_path / "plain assets"
    plain_directory.mkdir()
    quoted_directory = tmp_path / "wearer's assets"
    quoted_directory.mkdir()
    frames = ten_extracted_frames[:2]

    plain_font = plain_directory / "distinctive.ttf"
    plain_font.write_bytes(distinctive_font.read_bytes())
    quoted_font = quoted_directory / "worker's-distinctive.ttf"
    quoted_font.write_bytes(distinctive_font.read_bytes())

    plain_digest = _render_with_font(frames, plain_font, tmp_path / "plain.jpg", monkeypatch)
    default_digest = _render_with_font(
        frames, quoted_directory / "absent'-font.ttf", tmp_path / "absent.jpg", monkeypatch
    )
    # Only assert once this host can actually tell the two faces apart.
    if plain_digest == default_digest:
        pytest.skip("this host renders the distinctive font identically to its fallback")

    quoted_digest = _render_with_font(frames, quoted_font, tmp_path / "quoted.jpg", monkeypatch)

    assert quoted_digest == plain_digest, (
        "the apostrophe path rendered a different face than the same font at a plain path, "
        "so drawtext did not receive the file it was told to open"
    )


def test_contact_sheet_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        contact_sheet([], tmp_path / "never.jpg")


def _refuse_ffmpeg_for_invalid_contact_sheet_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_contact_sheet, "_find_usable_font_file", lambda: None)

    def fail_if_ffmpeg_runs(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid contact_sheet arguments must be refused before ffmpeg runs")

    monkeypatch.setattr(_contact_sheet.subprocess, "run", fail_if_ffmpeg_runs)


def _unreadable_frame(tmp_path: Path) -> ExtractedFrame:
    return ExtractedFrame(path=tmp_path / "must-not-be-read.jpg", log_time_ns=0)


@pytest.mark.parametrize("tile_width", [0, -320])
def test_contact_sheet_rejects_non_positive_tile_width_before_ffmpeg(
    tile_width: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_ffmpeg_for_invalid_contact_sheet_arguments(monkeypatch)
    with pytest.raises(ValueError, match=rf"tile_width must be >= 1, got {tile_width}"):
        contact_sheet([_unreadable_frame(tmp_path)], tmp_path / "never.jpg", tile_width=tile_width)


def test_contact_sheet_rejects_boolean_columns_before_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_ffmpeg_for_invalid_contact_sheet_arguments(monkeypatch)
    with pytest.raises(ValueError, match=r"columns must be an int, got True"):
        contact_sheet([_unreadable_frame(tmp_path)], tmp_path / "never.jpg", columns=True)


def test_contact_sheet_rejects_boolean_tile_width_before_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_ffmpeg_for_invalid_contact_sheet_arguments(monkeypatch)
    with pytest.raises(ValueError, match=r"tile_width must be an int, got True"):
        contact_sheet([_unreadable_frame(tmp_path)], tmp_path / "never.jpg", tile_width=True)


def test_contact_sheet_rejects_boolean_max_tiles_before_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_ffmpeg_for_invalid_contact_sheet_arguments(monkeypatch)
    with pytest.raises(ValueError, match=r"max_tiles must be an int, got True"):
        contact_sheet([_unreadable_frame(tmp_path)], tmp_path / "never.jpg", max_tiles=True)


def test_coding_range_is_derived_from_luma_and_selects_the_exposure_gates() -> None:
    """Trusting a container's declared range published a defect share off by a
    factor of hundreds on real footage, so the range is measured from the pixels.
    A frame leaving 16-235 proves the stream is full-range, which moves the
    clipping gate from 246 to 254 -- and a p90 of 250 flips from defect to fine.
    """
    limited = _aggregate_frame_statistics_output(
        _synthetic_frames({"lavfi.signalstats.YHIGH": 250}, {"lavfi.signalstats.YHIGH": 250})
    )
    assert limited.luma_range_evidence is LumaRangeEvidence.NOMINAL_LIMITED_RANGE_COMPATIBLE
    assert limited.clipped_highlight_frame_percent == pytest.approx(100.0)

    full = _aggregate_frame_statistics_output(
        _synthetic_frames(
            {"lavfi.signalstats.YHIGH": 250, "lavfi.signalstats.YMAX": 255},
            {"lavfi.signalstats.YHIGH": 250, "lavfi.signalstats.YMAX": 255},
        )
    )
    assert full.luma_range_evidence is LumaRangeEvidence.EXTENDS_BEYOND_NOMINAL_LIMITED_RANGE
    assert full.clipped_highlight_frame_percent == 0.0


def test_the_nominal_range_bounds_are_inclusive() -> None:
    """16 and 235 are themselves limited-range legal; only leaving them counts."""
    at_bounds = _aggregate_frame_statistics_output(
        _synthetic_frames({"lavfi.signalstats.YMIN": 16, "lavfi.signalstats.YMAX": 235})
    )
    assert at_bounds.luma_range_evidence is LumaRangeEvidence.NOMINAL_LIMITED_RANGE_COMPATIBLE
    below = _aggregate_frame_statistics_output(
        _synthetic_frames({"lavfi.signalstats.YMIN": 15, "lavfi.signalstats.YMAX": 235})
    )
    assert below.luma_range_evidence is LumaRangeEvidence.EXTENDS_BEYOND_NOMINAL_LIMITED_RANGE


def test_crushed_shadows_use_the_range_selected_gate() -> None:
    limited = _aggregate_frame_statistics_output(
        _synthetic_frames({"lavfi.signalstats.YLOW": 3}, {"lavfi.signalstats.YLOW": 200})
    )
    assert limited.crushed_shadow_frame_percent == pytest.approx(50.0)
    # The same p10 in a full-range stream is well inside the wider gate of 1.0.
    full = _aggregate_frame_statistics_output(
        _synthetic_frames(
            {"lavfi.signalstats.YLOW": 3, "lavfi.signalstats.YMAX": 255},
            {"lavfi.signalstats.YLOW": 200, "lavfi.signalstats.YMAX": 255},
        )
    )
    assert full.crushed_shadow_frame_percent == 0.0


def test_black_pixel_share_is_reported_over_every_frame() -> None:
    """Asking the filter for every frame's share, not just the flagged ones, is
    what keeps a half-covered lens visible instead of rounding to "not black".
    """
    stats = _aggregate_frame_statistics_output(
        _synthetic_frames(
            {"lavfi.blackframe.pblack": 50},
            {"lavfi.blackframe.pblack": 99},
            {"lavfi.blackframe.pblack": 0},
        )
    )
    # Only the 99% frame clears the 98% flag, but the 50% frame is still evidence.
    assert stats.black_frame_count == 1
    assert stats.black_pixel_share_maximum == pytest.approx(99.0)
    assert stats.black_pixel_share_mean == pytest.approx(149.0 / 3.0)


def test_luma_above_the_eight_bit_scale_raises() -> None:
    """A lost format pin would put a 10-bit source on a 0-1023 scale, where
    every threshold here is off by a factor of four. Fail loudly instead.
    """
    ten_bit = _synthetic_frames({"lavfi.signalstats.YMAX": 1023})
    with pytest.raises(FrameStatisticsParseError, match="outside the 8-bit range"):
        _aggregate_frame_statistics_output(ten_bit)


def test_first_frame_frame_difference_is_excluded_by_position() -> None:
    """The opening frame has no predecessor, so its YDIF is a sentinel zero.
    Dropping zeros by value would delete the real stillness this measures.
    """
    stats = _aggregate_frame_statistics_output(
        _synthetic_frames(
            {"lavfi.signalstats.YDIF": 0},
            {"lavfi.signalstats.YDIF": 4},
            {"lavfi.signalstats.YDIF": 6},
        )
    )
    # Mean over the two real observations, not three.
    assert stats.frame_difference_mean == pytest.approx(5.0)
    assert stats.frame_difference_maximum == pytest.approx(6.0)

    still = _aggregate_frame_statistics_output(
        _synthetic_frames({"lavfi.signalstats.YDIF": 0}, {"lavfi.signalstats.YDIF": 0})
    )
    assert still.frame_difference_mean == 0.0


def test_a_missing_required_signal_raises_instead_of_shrinking_a_denominator() -> None:
    """A frame silently dropped from a denominator turns "could not measure"
    into "measured and clean", which is the one answer this must never invent.
    """
    missing_tout = _synthetic_frames({"lavfi.signalstats.YAVG": 100})
    missing_tout = missing_tout.replace("lavfi.signalstats.TOUT=0\n", "")
    with pytest.raises(FrameStatisticsParseError, match="TOUT"):
        _aggregate_frame_statistics_output(missing_tout)


def test_hflow_cache_key_covers_graph_toolchain_and_video() -> None:
    video = Path("/work/wrist_cam.mp4")
    settings = FrameStatisticsSettings()
    toolchain = resolved_video_measurement_toolchain()
    baseline_cache_path = _frame_statistics_cache_path(video, settings, toolchain)
    assert baseline_cache_path is not None
    assert baseline_cache_path.parent == video.parent
    assert baseline_cache_path.name.startswith("wrist_cam.instrument.")

    changed_graph_cache_path = _frame_statistics_cache_path(
        video,
        FrameStatisticsSettings(black_pixel_luma_threshold=200),
        toolchain,
    )
    changed_toolchain_cache_path = _frame_statistics_cache_path(
        video,
        settings,
        VideoMeasurementToolchain(
            ffmpeg_executable=toolchain.ffmpeg_executable,
            ffprobe_executable=toolchain.ffprobe_executable,
            ffmpeg_version=toolchain.ffmpeg_version + "-different",
            ffprobe_version=toolchain.ffprobe_version,
        ),
    )
    assert changed_graph_cache_path != baseline_cache_path
    assert changed_toolchain_cache_path != baseline_cache_path
    assert _frame_statistics_cache_path(Path("/work/raw.h264"), settings, toolchain) is None


def test_cached_output_reaggregates_without_decoding_again(
    black_tail_video: Path, tmp_path: Path
) -> None:
    cached_video = tmp_path / "cached.mp4"
    shutil.copyfile(black_tail_video, cached_video)
    default_settings = FrameStatisticsSettings()
    first = measure_video_frame_statistics_for_hflow(cached_video, settings=default_settings)
    toolchain = resolved_video_measurement_toolchain()
    cache_path = _frame_statistics_cache_path(cached_video, default_settings, toolchain)
    assert cache_path is not None and cache_path.is_file()

    # A cache hit must remain usable even when another decode is impossible.
    cached_video.write_bytes(b"not a video anymore")
    tightened_settings = FrameStatisticsSettings(overexposed_average_luma_threshold=0.0)
    tightened = measure_video_frame_statistics_for_hflow(cached_video, settings=tightened_settings)
    assert tightened.decoded_frame_count == first.decoded_frame_count
    assert tightened.overexposed_frame_count == tightened.decoded_frame_count
    assert _frame_statistics_cache_path(cached_video, tightened_settings, toolchain) == cache_path
    assert not list(tmp_path.glob(".*.tmp"))


def test_graph_change_does_not_reuse_an_incompatible_cache(
    black_tail_video: Path, tmp_path: Path
) -> None:
    cached_video = tmp_path / "cached.mp4"
    shutil.copyfile(black_tail_video, cached_video)
    measure_video_frame_statistics_for_hflow(cached_video, settings=FrameStatisticsSettings())
    cached_video.write_bytes(b"not a video anymore")

    with pytest.raises(FrameStatisticsExecutionError):
        measure_video_frame_statistics_for_hflow(
            cached_video,
            settings=FrameStatisticsSettings(black_pixel_luma_threshold=200),
        )


def test_corrupt_cache_is_replaced_from_a_fresh_decode(
    black_tail_video: Path, tmp_path: Path
) -> None:
    cached_video = tmp_path / "cached.mp4"
    shutil.copyfile(black_tail_video, cached_video)
    settings = FrameStatisticsSettings()
    first = measure_video_frame_statistics_for_hflow(cached_video, settings=settings)
    cache_path = _frame_statistics_cache_path(
        cached_video, settings, resolved_video_measurement_toolchain()
    )
    assert cache_path is not None
    cache_path.write_text("not valid instrument output\n", encoding="utf-8")

    repaired = measure_video_frame_statistics_for_hflow(cached_video, settings=settings)
    assert repaired.decoded_frame_count == first.decoded_frame_count
    assert "lavfi.signalstats.YAVG" in cache_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))
