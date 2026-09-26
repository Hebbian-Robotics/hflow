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
    "CameraShakeRateBin",
    "CameraShakeSettings",
    "CameraShakeSummary",
    "MeasuredCameraMotion",
    "MeasuredCameraShake",
    "MotionFitEvidence",
    "UnavailableCameraShake",
    "UnmeasuredCameraMotion",
    "VideoMeasurementToolchain",
    "camera_shake_rate_percentile",
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
class CameraShakeRateBin:
    """Assessed duration in one bounded-width residual-rate interval."""

    rate_floor_degrees_per_second: int
    assessed_seconds: float

    def __post_init__(self) -> None:
        if (
            type(self.rate_floor_degrees_per_second) is not int
            or self.rate_floor_degrees_per_second < 0
        ):
            raise ValueError("rate floor must be a nonnegative integer")
        if self.rate_floor_degrees_per_second >= 4096:
            octave = self.rate_floor_degrees_per_second.bit_length() - 1
            bin_width = 1 << (octave - 6)
            if self.rate_floor_degrees_per_second % bin_width:
                raise ValueError("rate floor does not align with a logarithmic bin")
        if not math.isfinite(self.assessed_seconds) or self.assessed_seconds <= 0:
            raise ValueError("assessed duration must be positive and finite")


def _camera_shake_rate_floor(rate_degrees_per_second: float) -> int:
    integer_rate = math.floor(rate_degrees_per_second)
    if integer_rate < 4096:
        return integer_rate
    octave = integer_rate.bit_length() - 1
    bin_width = 1 << (octave - 6)
    return integer_rate // bin_width * bin_width


def _camera_shake_rate_bin_upper_bound(rate_floor_degrees_per_second: int) -> float:
    bin_width = (
        1
        if rate_floor_degrees_per_second < 4096
        else 1 << (rate_floor_degrees_per_second.bit_length() - 7)
    )
    return float(rate_floor_degrees_per_second + bin_width)


def camera_shake_rate_percentile(
    rate_bins: Iterable[CameraShakeRateBin],
    *,
    percentile: float,
    maximum_shake_degrees_per_second: float,
) -> float:
    """Return a duration-weighted quantile upper bound from mergeable bins.

    Bins can be merged by floor and duration across windows before calling this.
    Error is at most one degree/second below 4096 and at most 1.5625% above it.
    The observed maximum caps the upper edge of the selected bin.
    """
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered_bins = tuple(
        sorted(rate_bins, key=lambda rate_bin: rate_bin.rate_floor_degrees_per_second)
    )
    bin_durations = tuple(rate_bin.assessed_seconds for rate_bin in ordered_bins)
    assessed_seconds = math.fsum(bin_durations)
    if assessed_seconds <= 0:
        raise ValueError("rate bins must contain assessed duration")
    target_seconds = assessed_seconds * percentile / 100
    lower_index = 0
    upper_index = len(ordered_bins) - 1
    while lower_index < upper_index:
        middle_index = (lower_index + upper_index) // 2
        prefix_seconds = math.fsum(bin_durations[: middle_index + 1])
        if prefix_seconds >= target_seconds or (
            percentile < 100 and math.isclose(prefix_seconds, target_seconds, rel_tol=1e-15)
        ):
            upper_index = middle_index
        else:
            lower_index = middle_index + 1
    selected_bin = ordered_bins[lower_index]
    try:
        bin_upper_bound = _camera_shake_rate_bin_upper_bound(
            selected_bin.rate_floor_degrees_per_second
        )
    except OverflowError:
        return maximum_shake_degrees_per_second
    return min(bin_upper_bound, maximum_shake_degrees_per_second)


@dataclass(slots=True)
class _DurationAccumulator:
    """Accumulate repeated frame durations without drifting across quantile edges."""

    total_seconds: float = 0.0
    correction_seconds: float = 0.0

    def add(self, duration_seconds: float) -> None:
        corrected_duration = duration_seconds - self.correction_seconds
        updated_total = self.total_seconds + corrected_duration
        self.correction_seconds = (updated_total - self.total_seconds) - corrected_duration
        self.total_seconds = updated_total


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
    rate_bins: tuple[CameraShakeRateBin, ...]

    @property
    def unassessed_seconds(self) -> float:
        return self.observed_seconds - self.assessed_seconds

    @property
    def assessed_fraction(self) -> float | None:
        if self.observed_seconds == 0:
            return None
        return self.assessed_seconds / self.observed_seconds

    @property
    def p99_shake_degrees_per_second(self) -> float | None:
        """Duration-weighted p99 upper bound over the bounded rate bins."""
        if self.maximum_shake_degrees_per_second is None:
            return None
        return camera_shake_rate_percentile(
            self.rate_bins,
            percentile=99,
            maximum_shake_degrees_per_second=self.maximum_shake_degrees_per_second,
        )


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
    rate_duration_by_floor: dict[int, _DurationAccumulator] = {}

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
            rate_floor = _camera_shake_rate_floor(shake_degrees_per_second)
            rate_duration_by_floor.setdefault(rate_floor, _DurationAccumulator()).add(pair_seconds)
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
        rate_bins=tuple(
            CameraShakeRateBin(rate_floor, duration_accumulator.total_seconds)
            for rate_floor, duration_accumulator in sorted(rate_duration_by_floor.items())
        ),
    )
