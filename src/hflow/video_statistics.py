"""Public file-level frame statistics with explicit range and cache policy.

Measurements use original framing: callers should not add model-input padding.
Defaults retain the existing instrument's range handling; FULL explicitly converts
declared input range to full-range luma before measurement. Neither mode changes
the source file.
"""

from pathlib import Path

from hflow._video_measurement_toolchain import resolved_video_measurement_toolchain
from hflow._video_measurements._frame_statistics import (
    FRAME_STATISTICS_DEFINITION_VERSION,
    FrameStatisticsExecutionError,
    FrameStatisticsParseError,
    FrameStatisticsProvenance,
    FrameStatisticsSettings,
    LumaRangeEvidence,
    LumaRangePolicy,
    VideoFrameStatistics,
    VideoTimeInterval,
)
from hflow._video_measurements._frame_statistics import (
    measure_video_frame_statistics as _measure_video_frame_statistics,
)
from hflow._video_measurements._toolchain import VideoMeasurementToolchain

__all__ = [
    "FRAME_STATISTICS_DEFINITION_VERSION",
    "FrameStatisticsExecutionError",
    "FrameStatisticsParseError",
    "FrameStatisticsProvenance",
    "FrameStatisticsSettings",
    "LumaRangeEvidence",
    "LumaRangePolicy",
    "VideoFrameStatistics",
    "VideoMeasurementToolchain",
    "VideoTimeInterval",
    "measure_video_frame_statistics",
]


def measure_video_frame_statistics(
    video: Path,
    *,
    settings: FrameStatisticsSettings = FrameStatisticsSettings(),
    toolchain: VideoMeasurementToolchain | None = None,
    instrument_output_cache_path: Path | None = None,
) -> VideoFrameStatistics:
    """Measure without persistent output unless a cache path is explicitly supplied.

    Toolchain resolution follows HFlow's binary policy and may download binaries.
    The result records effective settings, filter graph and binary version.
    """
    return _measure_video_frame_statistics(
        video,
        settings=settings,
        toolchain=toolchain if toolchain is not None else resolved_video_measurement_toolchain(),
        instrument_output_cache_path=instrument_output_cache_path,
    )
