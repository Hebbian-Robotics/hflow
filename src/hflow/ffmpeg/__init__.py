"""HFlow's pinned FFmpeg binaries and contact-sheet rendering.

Public surface used in checks::

    sheet = hflow.ffmpeg.contact_sheet(ep.frames(fps=0.5), out_path)
"""

from hflow.ffmpeg._binary import (
    FFMPEG_ENV_VAR,
    FFPROBE_ENV_VAR,
    FfmpegNotFoundError,
    FfprobeNotFoundError,
    ffmpeg_path,
    ffmpeg_version,
    ffprobe_path,
    ffprobe_version,
)
from hflow.ffmpeg._contact_sheet import ContactSheet, contact_sheet

__all__ = [
    "FFMPEG_ENV_VAR",
    "FFPROBE_ENV_VAR",
    "ContactSheet",
    "FfmpegNotFoundError",
    "FfprobeNotFoundError",
    "contact_sheet",
    "ffmpeg_path",
    "ffmpeg_version",
    "ffprobe_path",
    "ffprobe_version",
]
