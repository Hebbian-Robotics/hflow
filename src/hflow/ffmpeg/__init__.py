"""ffmpeg-backed helpers: the pinned binary, the frame instrument, the contact sheet.

Public surface used in checks::

    stats = hflow.ffmpeg.frame_stats(ep.video("wrist_cam"))
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
from hflow.ffmpeg._instrument import FrameStats, InstrumentParseError, frame_stats
from hflow.ffmpeg._raw_frames import RawFrameError, luma_frames, rgb_frames, scaled_frame_shape

__all__ = [
    "FFMPEG_ENV_VAR",
    "FFPROBE_ENV_VAR",
    "ContactSheet",
    "FfmpegNotFoundError",
    "FfprobeNotFoundError",
    "FrameStats",
    "InstrumentParseError",
    "RawFrameError",
    "contact_sheet",
    "ffmpeg_path",
    "ffmpeg_version",
    "ffprobe_path",
    "ffprobe_version",
    "frame_stats",
    "luma_frames",
    "rgb_frames",
    "scaled_frame_shape",
]
