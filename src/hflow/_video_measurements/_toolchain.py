"""External executables used to decode and inspect video files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMeasurementToolchain:
    """The caller-owned FFmpeg toolchain behind a measurement.

    The measurement package does not download or select binaries. Callers make
    that policy decision and pass the exact executables and version strings in,
    which keeps licensing, release cadence, and deployment policy outside the
    measurement definitions.
    """

    ffmpeg_executable: Path
    ffprobe_executable: Path
    ffmpeg_version: str
    ffprobe_version: str

    def __post_init__(self) -> None:
        if not self.ffmpeg_executable.is_file():
            raise ValueError(f"ffmpeg executable does not exist: {self.ffmpeg_executable}")
        if not self.ffprobe_executable.is_file():
            raise ValueError(f"ffprobe executable does not exist: {self.ffprobe_executable}")
        if not self.ffmpeg_version.strip():
            raise ValueError("ffmpeg_version must not be empty")
        if not self.ffprobe_version.strip():
            raise ValueError("ffprobe_version must not be empty")
