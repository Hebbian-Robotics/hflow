"""Adapt HFlow's media limits to the shared optional MCAP camera exporter."""

from pathlib import Path

from hebbian_mcap.video import (
    CameraExportLimits,
)
from hebbian_mcap.video import (
    McapMediaError as McapMediaError,
)
from hebbian_mcap.video import (
    PreparedMcapVideo as PreparedMcapVideo,
)
from hebbian_mcap.video import (
    export_mcap_camera as _export_mcap_camera,
)

from hflow.media import VideoLimits


def export_mcap_camera(
    source_path: Path,
    output_path: Path,
    *,
    camera_topic: str | None = None,
    limits: VideoLimits = VideoLimits(),
) -> PreparedMcapVideo:
    """Export one camera using the caller's established media bounds."""
    return _export_mcap_camera(
        source_path,
        output_path,
        camera_topic=camera_topic,
        limits=CameraExportLimits(
            maximum_frame_pixels=limits.maximum_frame_pixels,
            maximum_frames_per_second=limits.maximum_frames_per_second,
            minimum_duration_seconds=limits.minimum_duration_seconds,
            maximum_duration_seconds=limits.maximum_duration_seconds,
        ),
    )
