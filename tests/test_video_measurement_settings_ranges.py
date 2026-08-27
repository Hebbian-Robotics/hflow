"""Range and value refusals on the video-measurement dataclasses (#200).

Every constructor in ``hflow._video_measurements`` refuses an out-of-range or
non-finite value, but the refusals themselves were never exercised - a
regression here would only surface as a confusing FFmpeg or filter-graph
error far from the constructor that let a bad value through. These tests are
dependency-free (no ffmpeg, no video fixtures, no ``cv2`` extra) except for
the ``VideoMeasurementToolchain`` executable-path checks, which only need a
``tmp_path`` file to stand in for a binary.

``_raw_frames.py``'s two refusals (``long_edge_pixels`` below 2,
``frames_per_second`` not positive in ``rgb_frames``) already have coverage
in ``test_ffmpeg.py`` (``test_scaled_frame_shape_refuses_a_degenerate_long_edge``,
``test_rgb_frames_reject_a_non_positive_rate``) and are not duplicated here.
"""

from pathlib import Path

import pytest

from hflow._video_measurements._camera_motion import CameraMotionSettings
from hflow._video_measurements._frame_statistics import (
    FrameStatisticsSettings,
    VideoTimeInterval,
)
from hflow._video_measurements._toolchain import VideoMeasurementToolchain


@pytest.fixture
def real_binaries(tmp_path: Path) -> tuple[Path, Path]:
    ffmpeg_executable = tmp_path / "ffmpeg"
    ffprobe_executable = tmp_path / "ffprobe"
    ffmpeg_executable.touch()
    ffprobe_executable.touch()
    return ffmpeg_executable, ffprobe_executable


def test_toolchain_refuses_a_missing_ffmpeg_executable(
    real_binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    _, ffprobe_executable = real_binaries
    with pytest.raises(ValueError, match="ffmpeg executable does not exist"):
        VideoMeasurementToolchain(
            ffmpeg_executable=tmp_path / "no-such-ffmpeg",
            ffprobe_executable=ffprobe_executable,
            ffmpeg_version="7.0",
            ffprobe_version="7.0",
        )


def test_toolchain_refuses_a_missing_ffprobe_executable(
    real_binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    ffmpeg_executable, _ = real_binaries
    with pytest.raises(ValueError, match="ffprobe executable does not exist"):
        VideoMeasurementToolchain(
            ffmpeg_executable=ffmpeg_executable,
            ffprobe_executable=tmp_path / "no-such-ffprobe",
            ffmpeg_version="7.0",
            ffprobe_version="7.0",
        )


@pytest.mark.parametrize("empty_version", ["", "   "])
def test_toolchain_refuses_an_empty_ffmpeg_version(
    real_binaries: tuple[Path, Path], empty_version: str
) -> None:
    ffmpeg_executable, ffprobe_executable = real_binaries
    with pytest.raises(ValueError, match="ffmpeg_version must not be empty"):
        VideoMeasurementToolchain(
            ffmpeg_executable=ffmpeg_executable,
            ffprobe_executable=ffprobe_executable,
            ffmpeg_version=empty_version,
            ffprobe_version="7.0",
        )


@pytest.mark.parametrize("empty_version", ["", "   "])
def test_toolchain_refuses_an_empty_ffprobe_version(
    real_binaries: tuple[Path, Path], empty_version: str
) -> None:
    ffmpeg_executable, ffprobe_executable = real_binaries
    with pytest.raises(ValueError, match="ffprobe_version must not be empty"):
        VideoMeasurementToolchain(
            ffmpeg_executable=ffmpeg_executable,
            ffprobe_executable=ffprobe_executable,
            ffmpeg_version="7.0",
            ffprobe_version=empty_version,
        )


def test_toolchain_accepts_real_paths_and_versions(real_binaries: tuple[Path, Path]) -> None:
    ffmpeg_executable, ffprobe_executable = real_binaries
    toolchain = VideoMeasurementToolchain(
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        ffmpeg_version="7.0",
        ffprobe_version="7.0",
    )
    assert toolchain.ffmpeg_executable == ffmpeg_executable


@pytest.mark.parametrize(
    "start_seconds",
    [-1.0, float("-inf"), float("inf"), float("nan")],
    ids=["negative", "-inf", "+inf", "nan"],
)
def test_video_time_interval_refuses_a_bad_start(start_seconds: float) -> None:
    with pytest.raises(ValueError, match="start_seconds must be finite and nonnegative"):
        VideoTimeInterval(start_seconds=start_seconds, end_seconds=10.0)


@pytest.mark.parametrize(
    "end_seconds",
    [0.5, float("-inf"), float("inf"), float("nan")],
    ids=["before_start", "-inf", "+inf", "nan"],
)
def test_video_time_interval_refuses_a_bad_end(end_seconds: float) -> None:
    with pytest.raises(
        ValueError, match="end_seconds must be finite and no earlier than start_seconds"
    ):
        VideoTimeInterval(start_seconds=1.0, end_seconds=end_seconds)


def test_video_time_interval_accepts_a_zero_length_interval() -> None:
    interval = VideoTimeInterval(start_seconds=0.0, end_seconds=0.0)
    assert interval.end_seconds == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"black_frame_minimum_pixel_share_percent": -1},
            "black_frame_minimum_pixel_share_percent must be between 0 and 100",
            id="black_frame_minimum_pixel_share_percent-below",
        ),
        pytest.param(
            {"black_frame_minimum_pixel_share_percent": 101},
            "black_frame_minimum_pixel_share_percent must be between 0 and 100",
            id="black_frame_minimum_pixel_share_percent-above",
        ),
        pytest.param(
            {"black_pixel_luma_threshold": -1},
            "black_pixel_luma_threshold must be between 0 and 255",
            id="black_pixel_luma_threshold-below",
        ),
        pytest.param(
            {"black_pixel_luma_threshold": 256},
            "black_pixel_luma_threshold must be between 0 and 255",
            id="black_pixel_luma_threshold-above",
        ),
        pytest.param(
            {"freeze_noise_tolerance_decibels": float("nan")},
            "freeze_noise_tolerance_decibels must be finite",
            id="freeze_noise_tolerance_decibels-nan",
        ),
        pytest.param(
            {"freeze_noise_tolerance_decibels": float("inf")},
            "freeze_noise_tolerance_decibels must be finite",
            id="freeze_noise_tolerance_decibels-inf",
        ),
        pytest.param(
            {"freeze_minimum_duration_seconds": 0.0},
            "freeze_minimum_duration_seconds must be finite and positive",
            id="freeze_minimum_duration_seconds-zero",
        ),
        pytest.param(
            {"freeze_minimum_duration_seconds": -1.0},
            "freeze_minimum_duration_seconds must be finite and positive",
            id="freeze_minimum_duration_seconds-negative",
        ),
        pytest.param(
            {"freeze_minimum_duration_seconds": float("nan")},
            "freeze_minimum_duration_seconds must be finite and positive",
            id="freeze_minimum_duration_seconds-nan",
        ),
        pytest.param(
            {"overexposed_average_luma_threshold": -1.0},
            "overexposed_average_luma_threshold must be between 0 and 255",
            id="overexposed_average_luma_threshold-below",
        ),
        pytest.param(
            {"overexposed_average_luma_threshold": 256.0},
            "overexposed_average_luma_threshold must be between 0 and 255",
            id="overexposed_average_luma_threshold-above",
        ),
        pytest.param(
            {"overexposed_average_luma_threshold": float("nan")},
            "overexposed_average_luma_threshold must be between 0 and 255",
            id="overexposed_average_luma_threshold-nan",
        ),
    ],
)
def test_frame_statistics_settings_refuses_out_of_range_fields(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        FrameStatisticsSettings(**kwargs)  # ty: ignore


def test_frame_statistics_settings_defaults_are_valid() -> None:
    settings = FrameStatisticsSettings()
    assert settings.black_pixel_luma_threshold == 17


def test_frame_statistics_settings_accepts_zero_where_in_range() -> None:
    settings = FrameStatisticsSettings(
        black_frame_minimum_pixel_share_percent=0,
        black_pixel_luma_threshold=0,
        overexposed_average_luma_threshold=0.0,
    )
    assert settings.black_frame_minimum_pixel_share_percent == 0
    assert settings.black_pixel_luma_threshold == 0
    assert settings.overexposed_average_luma_threshold == 0.0


@pytest.mark.parametrize(
    "frames_per_second",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "inf"],
)
def test_camera_motion_settings_refuses_a_bad_frame_rate(frames_per_second: float) -> None:
    with pytest.raises(ValueError, match="frames_per_second must be finite and positive"):
        CameraMotionSettings(frames_per_second=frames_per_second)


@pytest.mark.parametrize(
    "horizontal_field_of_view_degrees",
    [0.0, -1.0, 360.1, float("nan"), float("inf")],
    ids=["zero", "negative", "above_360", "nan", "inf"],
)
def test_camera_motion_settings_refuses_a_bad_field_of_view(
    horizontal_field_of_view_degrees: float,
) -> None:
    with pytest.raises(
        ValueError, match=r"horizontal_field_of_view_degrees must be finite and in \(0, 360\]"
    ):
        CameraMotionSettings(
            frames_per_second=30.0,
            horizontal_field_of_view_degrees=horizontal_field_of_view_degrees,
        )


def test_camera_motion_settings_accepts_the_boundary_of_360() -> None:
    settings = CameraMotionSettings(frames_per_second=30.0, horizontal_field_of_view_degrees=360.0)
    assert settings.horizontal_field_of_view_degrees == 360.0
