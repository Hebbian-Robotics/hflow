"""Continuous camera motion; not a perceptual stability verdict.

Use stream_camera_motion as a context manager to own the decoder lifecycle.
The underlying frame iterator and temporal filter also work without HFlow
episodes, catalogs, or orchestration. See docs/how-to/stream-camera-motion.md.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hflow._video_measurement_toolchain import resolved_video_measurement_toolchain
from hflow._video_measurements._motion_fit import (
    CameraMotionTransform,
    MeasuredCameraMotion,
    MotionFitEvidence,
    UnmeasuredCameraMotion,
)
from hflow._video_measurements._motion_settings import CameraMotionAlgorithmSettings
from hflow._video_measurements._motion_stream import (
    CAMERA_MOTION_STREAM_DEFINITION_VERSION,
    AngularMotionRate,
    CameraMotionObservation,
    CameraMotionStreamSettings,
    CameraShakeObservation,
    CameraShakeSettings,
    MeasuredCameraShake,
    UnavailableCameraShake,
    filter_camera_shake,
    iter_frame_motion,
)
from hflow._video_measurements._raw_frames import luma_frames
from hflow._video_measurements._toolchain import VideoMeasurementToolchain

__all__ = [
    "CAMERA_MOTION_STREAM_DEFINITION_VERSION",
    "AngularMotionRate",
    "CameraMotionAlgorithmSettings",
    "CameraMotionObservation",
    "CameraMotionStreamSettings",
    "CameraMotionTransform",
    "CameraShakeObservation",
    "CameraShakeSettings",
    "MeasuredCameraMotion",
    "MeasuredCameraShake",
    "MotionFitEvidence",
    "UnavailableCameraShake",
    "UnmeasuredCameraMotion",
    "VideoMeasurementToolchain",
    "filter_camera_shake",
    "iter_frame_motion",
    "stream_camera_motion",
]


@contextmanager
def stream_camera_motion(
    video: Path,
    *,
    settings: CameraMotionStreamSettings,
    toolchain: VideoMeasurementToolchain | None = None,
) -> Iterator[Iterator[CameraMotionObservation]]:
    """Decode a fixed-frame-rate file incrementally at the caller-supplied rate.

    Times are relative to the first decoded frame, not container presentation
    timestamps. Variable-frame-rate input must be normalized by the caller.
    HFlow resolves (and may download) its managed FFmpeg unless given a toolchain.
    Leaving the context closes the iterator and reaps FFmpeg, including on break.
    """
    resolved_toolchain = (
        toolchain if toolchain is not None else resolved_video_measurement_toolchain()
    )
    with luma_frames(video, toolchain=resolved_toolchain) as frames:
        observations = iter_frame_motion(frames, settings=settings)
        try:
            yield observations
        finally:
            observations.close()
