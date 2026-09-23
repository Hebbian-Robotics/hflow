"""Type, range, and value refusals on the video-measurement dataclasses (#199, #200).

Every constructor in ``hflow._video_measurements`` refuses an out-of-range or
non-finite value, but the refusals themselves were never exercised - a
regression here would only surface as a confusing FFmpeg or filter-graph
error far from the constructor that let a bad value through.

Range checks alone (``0 <= x <= 100``) do not reject the wrong *type*: ``bool``
subclasses ``int``, so ``True``/``False`` satisfy every numeric comparison
these settings run, and a value of the wrong type entirely (``str``, ``None``)
previously raised a bare ``TypeError`` from the comparison rather than a clear
message naming the field.

These tests are dependency-free (no ffmpeg, no video fixtures, no ``cv2``
extra) except for the ``VideoMeasurementToolchain`` executable-path checks,
which only need a ``tmp_path`` file to stand in for a binary.

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

# Stands in for "an executable path that does not exist" in the toolchain
# refusal table, since the real missing path depends on ``tmp_path``.
_MISSING_EXECUTABLE = "missing"


@pytest.fixture
def real_binaries(tmp_path: Path) -> tuple[Path, Path]:
    ffmpeg_executable = tmp_path / "ffmpeg"
    ffprobe_executable = tmp_path / "ffprobe"
    ffmpeg_executable.touch()
    ffprobe_executable.touch()
    return ffmpeg_executable, ffprobe_executable


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        pytest.param(
            "ffmpeg_executable",
            _MISSING_EXECUTABLE,
            "ffmpeg executable does not exist",
            id="missing-ffmpeg",
        ),
        pytest.param(
            "ffprobe_executable",
            _MISSING_EXECUTABLE,
            "ffprobe executable does not exist",
            id="missing-ffprobe",
        ),
        pytest.param(
            "ffmpeg_version", "", "ffmpeg_version must not be empty", id="empty-ffmpeg-version"
        ),
        pytest.param(
            "ffmpeg_version", "   ", "ffmpeg_version must not be empty", id="blank-ffmpeg-version"
        ),
        pytest.param(
            "ffprobe_version", "", "ffprobe_version must not be empty", id="empty-ffprobe-version"
        ),
        pytest.param(
            "ffprobe_version",
            "   ",
            "ffprobe_version must not be empty",
            id="blank-ffprobe-version",
        ),
    ],
)
def test_toolchain_refuses_a_missing_executable_or_empty_version(
    real_binaries: tuple[Path, Path], tmp_path: Path, field_name: str, value: str, message: str
) -> None:
    ffmpeg_executable, ffprobe_executable = real_binaries
    toolchain_arguments: dict[str, object] = {
        "ffmpeg_executable": ffmpeg_executable,
        "ffprobe_executable": ffprobe_executable,
        "ffmpeg_version": "7.0",
        "ffprobe_version": "7.0",
    }
    toolchain_arguments[field_name] = (
        tmp_path / f"no-such-{field_name}" if value == _MISSING_EXECUTABLE else value
    )
    with pytest.raises(ValueError, match=message):
        VideoMeasurementToolchain(**toolchain_arguments)


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


# (constructor kwargs, expected error text) for every bool-typed refusal.
_BOOL_REFUSALS = [
    pytest.param(
        FrameStatisticsSettings,
        {"black_frame_minimum_pixel_share_percent": True},
        "black_frame_minimum_pixel_share_percent",
        id="frame_statistics-black_frame_minimum_pixel_share_percent",
    ),
    pytest.param(
        FrameStatisticsSettings,
        {"black_pixel_luma_threshold": True},
        "black_pixel_luma_threshold",
        id="frame_statistics-black_pixel_luma_threshold",
    ),
    pytest.param(
        FrameStatisticsSettings,
        {"freeze_noise_tolerance_decibels": True},
        "freeze_noise_tolerance_decibels",
        id="frame_statistics-freeze_noise_tolerance_decibels",
    ),
    pytest.param(
        FrameStatisticsSettings,
        {"freeze_minimum_duration_seconds": True},
        "freeze_minimum_duration_seconds",
        id="frame_statistics-freeze_minimum_duration_seconds",
    ),
    pytest.param(
        FrameStatisticsSettings,
        {"overexposed_average_luma_threshold": True},
        "overexposed_average_luma_threshold",
        id="frame_statistics-overexposed_average_luma_threshold",
    ),
    pytest.param(
        VideoTimeInterval,
        {"start_seconds": True, "end_seconds": 1.0},
        "start_seconds",
        id="video_time_interval-start_seconds",
    ),
    pytest.param(
        VideoTimeInterval,
        {"start_seconds": 0.0, "end_seconds": True},
        "end_seconds",
        id="video_time_interval-end_seconds",
    ),
    pytest.param(
        CameraMotionSettings,
        {"frames_per_second": True},
        "frames_per_second",
        id="camera_motion-frames_per_second",
    ),
    pytest.param(
        CameraMotionSettings,
        {"frames_per_second": 30.0, "horizontal_field_of_view_degrees": True},
        "horizontal_field_of_view_degrees",
        id="camera_motion-horizontal_field_of_view_degrees",
    ),
]


@pytest.mark.parametrize(("settings_cls", "kwargs", "field_name"), _BOOL_REFUSALS)
def test_bool_is_refused_for_every_field(
    settings_cls: type, kwargs: dict[str, object], field_name: str
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name}.*bool"):
        settings_cls(**kwargs)


@pytest.mark.parametrize(
    ("settings_cls", "kwargs", "message"),
    [
        pytest.param(
            FrameStatisticsSettings,
            {"black_pixel_luma_threshold": 17.5},
            r"black_pixel_luma_threshold.*float",
            id="int-field-float",
        ),
        pytest.param(
            FrameStatisticsSettings,
            {"black_frame_minimum_pixel_share_percent": "98"},
            r"black_frame_minimum_pixel_share_percent.*str",
            id="int-field-str",
        ),
        pytest.param(
            CameraMotionSettings,
            {"frames_per_second": "30"},
            "frames_per_second",
            id="float-field-str",
        ),
        pytest.param(
            CameraMotionSettings,
            {"frames_per_second": None},
            "frames_per_second",
            id="float-field-none",
        ),
    ],
)
def test_a_value_of_the_wrong_type_is_refused_naming_the_field(
    settings_cls: type, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        settings_cls(**kwargs)


def test_float_field_accepts_a_plain_int() -> None:
    # An int is a perfectly good float value here; it must not be coerced or
    # rejected, and 0 (falsy but real) must still be accepted where in range.
    settings = FrameStatisticsSettings(freeze_minimum_duration_seconds=2)
    assert settings.freeze_minimum_duration_seconds == 2

    interval = VideoTimeInterval(start_seconds=0, end_seconds=1)
    assert interval.start_seconds == 0
