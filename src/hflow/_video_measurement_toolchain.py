"""HFlow's toolchain and workdir-cache video-measurement adapters."""

import hashlib
from pathlib import Path

from hflow._video_measurements import (
    FRAME_STATISTICS_DEFINITION_VERSION,
    FrameStatisticsSettings,
    VideoFrameStatistics,
    VideoMeasurementToolchain,
    measure_video_frame_statistics,
)
from hflow._video_measurements._frame_statistics import frame_statistics_filter_graph
from hflow.ffmpeg import _binary


def resolved_video_measurement_toolchain() -> VideoMeasurementToolchain:
    """Describe the exact FFmpeg pair selected by HFlow's binary policy."""
    return VideoMeasurementToolchain(
        ffmpeg_executable=_binary.ffmpeg_path(),
        ffprobe_executable=_binary.ffprobe_path(),
        ffmpeg_version=_binary.ffmpeg_version(),
        ffprobe_version=_binary.ffprobe_version(),
    )


def _frame_statistics_cache_path(
    video: Path,
    settings: FrameStatisticsSettings,
    toolchain: VideoMeasurementToolchain,
) -> Path | None:
    """Return HFlow's workdir cache path for one effective instrument."""
    if video.suffix.lower() != ".mp4":
        return None
    cache_identity = "\0".join(
        (
            FRAME_STATISTICS_DEFINITION_VERSION,
            toolchain.ffmpeg_version,
            frame_statistics_filter_graph(settings),
        )
    )
    cache_digest = hashlib.sha256(cache_identity.encode()).hexdigest()[:16]
    return video.with_name(f"{video.stem}.instrument.{cache_digest}.txt")


def measure_video_frame_statistics_for_hflow(
    video: Path, *, settings: FrameStatisticsSettings
) -> VideoFrameStatistics:
    """Measure with HFlow's pinned toolchain and persistent workdir cache."""
    toolchain = resolved_video_measurement_toolchain()
    return measure_video_frame_statistics(
        video,
        toolchain=toolchain,
        settings=settings,
        instrument_output_cache_path=_frame_statistics_cache_path(video, settings, toolchain),
    )
