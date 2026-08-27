"""Type guards on the video-measurement settings dataclasses (#199).

Range checks alone (``0 <= x <= 100``) do not reject the wrong *type*: ``bool``
subclasses ``int``, so ``True``/``False`` satisfy every numeric comparison
these settings already run, and a value of the wrong type entirely (``str``,
``None``) previously raised a bare ``TypeError`` from the comparison rather
than a clear message naming the field. These tests are dependency-free
(no ffmpeg, no video fixtures, no ``cv2`` extra) since they only exercise
dataclass construction.
"""

import pytest

from hflow._video_measurements._camera_motion import CameraMotionSettings
from hflow._video_measurements._frame_statistics import (
    FrameStatisticsSettings,
    VideoTimeInterval,
)

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


def test_int_field_refuses_a_float() -> None:
    with pytest.raises(ValueError, match=r"black_pixel_luma_threshold.*float"):
        FrameStatisticsSettings(black_pixel_luma_threshold=17.5)  # ty: ignore


def test_int_field_refuses_a_str() -> None:
    with pytest.raises(ValueError, match=r"black_frame_minimum_pixel_share_percent.*str"):
        FrameStatisticsSettings(black_frame_minimum_pixel_share_percent="98")  # ty: ignore


@pytest.mark.parametrize(
    ("settings_cls", "kwargs"),
    [
        pytest.param(CameraMotionSettings, {"frames_per_second": "30"}, id="camera_motion-str"),
        pytest.param(CameraMotionSettings, {"frames_per_second": None}, id="camera_motion-none"),
    ],
)
def test_float_field_refuses_non_numeric_types(
    settings_cls: type, kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="frames_per_second"):
        settings_cls(**kwargs)


def test_float_field_accepts_a_plain_int() -> None:
    # An int is a perfectly good float value here; it must not be coerced or
    # rejected, and 0 (falsy but real) must still be accepted where in range.
    settings = FrameStatisticsSettings(freeze_minimum_duration_seconds=2)
    assert settings.freeze_minimum_duration_seconds == 2

    interval = VideoTimeInterval(start_seconds=0, end_seconds=1)
    assert interval.start_seconds == 0


def test_int_field_accepts_zero() -> None:
    settings = FrameStatisticsSettings(black_pixel_luma_threshold=0)
    assert settings.black_pixel_luma_threshold == 0


def test_defaults_are_unaffected() -> None:
    # No behavior change for the values every caller already relies on.
    assert FrameStatisticsSettings() == FrameStatisticsSettings()
    assert VideoTimeInterval(start_seconds=0.0, end_seconds=1.0).end_seconds == 1.0
    assert CameraMotionSettings(frames_per_second=30.0).frames_per_second == 30.0
