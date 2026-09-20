"""Continuous camera motion; not a perceptual stability verdict.

Use stream_camera_motion as a context manager to own the decoder lifecycle.
The underlying frame iterator and temporal filter also work without HFlow
episodes, catalogs, or orchestration. See docs/how-to/stream-camera-motion.md.
"""

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

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
    "CameraShakeSummary",
    "MeasuredCameraMotion",
    "MeasuredCameraShake",
    "MotionFitEvidence",
    "UnavailableCameraShake",
    "UnmeasuredCameraMotion",
    "VideoMeasurementToolchain",
    "filter_camera_shake",
    "iter_frame_motion",
    "stream_camera_motion",
    "summarize_camera_shake",
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


@dataclass(frozen=True, slots=True)
class CameraShakeSummary:
    """Continuous residual rates and their observed frame-pair coverage.

    Angular rates use the caller's approximate field of view. A fitted motion
    estimate can still be unreliable; coverage does not establish accuracy.
    The final frame's display duration is outside the observed pair intervals.
    """

    pair_count: int
    measured_motion_pair_count: int
    measured_shake_pair_count: int
    insufficient_context_pair_count: int
    unmeasured_context_pair_count: int
    observed_seconds: float
    assessed_seconds: float
    mean_shake_degrees_per_second: float | None
    rms_shake_degrees_per_second: float | None
    maximum_shake_degrees_per_second: float | None

    @property
    def unassessed_seconds(self) -> float:
        return self.observed_seconds - self.assessed_seconds

    @property
    def assessed_fraction(self) -> float | None:
        if self.observed_seconds == 0:
            return None
        return self.assessed_seconds / self.observed_seconds


def summarize_camera_shake(observations: Iterable[CameraShakeObservation]) -> CameraShakeSummary:
    """Reduce a complete filtered motion stream with constant summary memory.

    Missing motion and filter context stay unassessed, never zero shake. Rates
    are weighted by the adjacent-pair durations; no video frames or rate history
    are retained. Ordering and cadence belong to HFlow's filter contract.
    """
    pair_count = 0
    measured_motion_pair_count = 0
    measured_shake_pair_count = 0
    insufficient_context_pair_count = 0
    unmeasured_context_pair_count = 0
    observed_seconds = 0.0
    assessed_seconds = 0.0
    mean_shake_degrees_per_second = 0.0
    rms_shake_degrees_per_second = 0.0
    maximum_shake_degrees_per_second = 0.0

    for observation in observations:
        pair_count += 1
        pair_seconds = observation.motion.end_seconds - observation.motion.start_seconds
        observed_seconds += pair_seconds
        if isinstance(observation.motion.measurement, MeasuredCameraMotion):
            measured_motion_pair_count += 1

        shake = observation.shake
        if isinstance(shake, MeasuredCameraShake):
            measured_shake_pair_count += 1
            assessed_seconds += pair_seconds
            shake_degrees_per_second = shake.residual.magnitude_degrees_per_second
            duration_share = pair_seconds / assessed_seconds
            mean_shake_degrees_per_second += duration_share * (
                shake_degrees_per_second - mean_shake_degrees_per_second
            )
            # The weighted Euclidean norm avoids squaring large finite rates.
            rms_shake_degrees_per_second = math.hypot(
                rms_shake_degrees_per_second * math.sqrt(1 - duration_share),
                shake_degrees_per_second * math.sqrt(duration_share),
            )
            maximum_shake_degrees_per_second = max(
                maximum_shake_degrees_per_second, shake_degrees_per_second
            )
        else:
            match shake.reason:
                case "insufficient_context":
                    insufficient_context_pair_count += 1
                case "unmeasured_context":
                    unmeasured_context_pair_count += 1
                case unknown_reason:
                    assert_never(unknown_reason)

    return CameraShakeSummary(
        pair_count=pair_count,
        measured_motion_pair_count=measured_motion_pair_count,
        measured_shake_pair_count=measured_shake_pair_count,
        insufficient_context_pair_count=insufficient_context_pair_count,
        unmeasured_context_pair_count=unmeasured_context_pair_count,
        observed_seconds=observed_seconds,
        assessed_seconds=assessed_seconds,
        mean_shake_degrees_per_second=(
            mean_shake_degrees_per_second if measured_shake_pair_count else None
        ),
        rms_shake_degrees_per_second=(
            rms_shake_degrees_per_second if measured_shake_pair_count else None
        ),
        maximum_shake_degrees_per_second=(
            maximum_shake_degrees_per_second if measured_shake_pair_count else None
        ),
    )
