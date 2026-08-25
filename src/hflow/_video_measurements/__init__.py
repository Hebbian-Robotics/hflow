"""Incubating, format-independent measurements over individual video files.

This package is private while its result model and measurement definitions
settle. It deliberately has no dependency on HFlow episodes, checks, catalog
keys, gates, or binary-download policy so it can become a standalone package.
"""

from ._camera_motion import (
    CAMERA_MOTION_DEFINITION_VERSION,
    DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES,
    CameraMotionMeasurements,
    CameraMotionProvenance,
    CameraMotionResult,
    CameraMotionSettings,
    InsufficientVideoFrames,
    MotionExtraNotInstalledError,
    measure_camera_motion,
)
from ._frame_statistics import (
    FRAME_STATISTICS_DEFINITION_VERSION,
    FrameStatisticsExecutionError,
    FrameStatisticsParseError,
    FrameStatisticsProvenance,
    FrameStatisticsSettings,
    LumaRangeEvidence,
    UnsupportedVideoMeasurementToolchainError,
    VideoFrameStatistics,
    VideoTimeInterval,
    measure_video_frame_statistics,
)
from ._toolchain import VideoMeasurementToolchain

__all__ = [
    "CAMERA_MOTION_DEFINITION_VERSION",
    "DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES",
    "FRAME_STATISTICS_DEFINITION_VERSION",
    "CameraMotionMeasurements",
    "CameraMotionProvenance",
    "CameraMotionResult",
    "CameraMotionSettings",
    "FrameStatisticsExecutionError",
    "FrameStatisticsParseError",
    "FrameStatisticsProvenance",
    "FrameStatisticsSettings",
    "InsufficientVideoFrames",
    "LumaRangeEvidence",
    "MotionExtraNotInstalledError",
    "UnsupportedVideoMeasurementToolchainError",
    "VideoFrameStatistics",
    "VideoMeasurementToolchain",
    "VideoTimeInterval",
    "measure_camera_motion",
    "measure_video_frame_statistics",
]
