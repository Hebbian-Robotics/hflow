"""Continuous motion and gap-aware shake filtering."""

import math
from collections import deque
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from hflow._field_guards import (
    require_finite_float,
    require_int_in_range,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
)

from ._camera_motion import _import_cv2
from ._motion_fit import (
    FrameMotionResult,
    UnmeasuredCameraMotion,
    fit_frame_motion,
)
from ._motion_settings import CameraMotionAlgorithmSettings

CAMERA_MOTION_STREAM_DEFINITION_VERSION = "camera-motion-stream/v1"


@dataclass(frozen=True, slots=True)
class CameraMotionStreamSettings:
    """Fixed-cadence input; fit evidence is interpreted by the consumer."""

    frames_per_second: float
    algorithm: CameraMotionAlgorithmSettings = field(default_factory=CameraMotionAlgorithmSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, CameraMotionAlgorithmSettings):
            raise ValueError("algorithm must be CameraMotionAlgorithmSettings")
        require_positive_float(self.frames_per_second, "frames_per_second")
        if not math.isfinite(1 / self.frames_per_second):
            raise ValueError("frames_per_second must have a finite reciprocal")


@dataclass(frozen=True, slots=True)
class CameraMotionObservation:
    pair_index: int
    start_seconds: float
    end_seconds: float
    frame_width_pixels: int
    measurement: FrameMotionResult
    settings: CameraMotionStreamSettings
    definition_version: str = field(default=CAMERA_MOTION_STREAM_DEFINITION_VERSION, init=False)

    def __post_init__(self) -> None:
        require_non_negative_int(self.pair_index, "pair_index")
        require_positive_int(self.frame_width_pixels, "frame_width_pixels")
        require_finite_float(self.start_seconds, "start_seconds")
        require_finite_float(self.end_seconds, "end_seconds")
        if (
            self.start_seconds != self.pair_index / self.settings.frames_per_second
            or self.end_seconds != (self.pair_index + 1) / self.settings.frames_per_second
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("pair timestamps must match the declared fixed frame cadence")


def iter_frame_motion(
    frames: Iterable[np.ndarray], *, settings: CameraMotionStreamSettings
) -> Generator[CameraMotionObservation, None, None]:
    """Consume fixed-cadence uint8 grayscale frames; retain no motion history.

    A pair has an observation even when no finite transform can be estimated.
    Frames may reuse a buffer: the previous image is copied before requesting
    the next. Empty and single-frame inputs yield no adjacent-pair observations.
    The caller owns the input iterator and must close it if it owns resources.
    """
    cv2 = _import_cv2()
    previous_frame: np.ndarray | None = None
    for frame_index, current_frame in enumerate(frames):
        if (
            not isinstance(current_frame, np.ndarray)
            or current_frame.dtype != np.uint8
            or current_frame.ndim != 2
            or min(current_frame.shape) < 1
        ):
            raise ValueError("frames must be nonempty two-dimensional uint8 grayscale arrays")
        if previous_frame is not None:
            if current_frame.shape != previous_frame.shape:
                raise ValueError("frame dimensions must stay constant throughout the stream")
            result = fit_frame_motion(
                cv2, previous_frame, current_frame, settings=settings.algorithm
            )
            observation = CameraMotionObservation(
                pair_index=frame_index - 1,
                start_seconds=(frame_index - 1) / settings.frames_per_second,
                end_seconds=frame_index / settings.frames_per_second,
                frame_width_pixels=current_frame.shape[1],
                measurement=result,
                settings=settings,
            )
            previous_frame = current_frame.copy()
            yield observation
        else:
            previous_frame = current_frame.copy()


@dataclass(frozen=True, slots=True)
class CameraShakeSettings:
    """Subtract a centered mean over 2*half_window_pairs+1 valid motion rates.

    The default uses 31 pairs and 15 pairs of lookahead (0.5 seconds at 30 fps).
    The field of view is an explicit approximate pixel-to-angle conversion,
    not a calibrated camera model. The residual excludes scale changes.
    """

    half_window_pairs: int = 15
    horizontal_field_of_view_degrees: float = 90.0

    def __post_init__(self) -> None:
        require_int_in_range(self.half_window_pairs, "half_window_pairs", minimum=1, maximum=4096)
        field_of_view = require_positive_float(
            self.horizontal_field_of_view_degrees, "horizontal_field_of_view_degrees"
        )
        if field_of_view > 360:
            raise ValueError("horizontal_field_of_view_degrees must be <= 360")


@dataclass(frozen=True, slots=True)
class AngularMotionRate:
    rotation_degrees_per_second: float
    horizontal_degrees_per_second: float
    vertical_degrees_per_second: float

    def __post_init__(self) -> None:
        for field_name in (
            "rotation_degrees_per_second",
            "horizontal_degrees_per_second",
            "vertical_degrees_per_second",
        ):
            require_finite_float(getattr(self, field_name), field_name)
        if not math.isfinite(self.magnitude_degrees_per_second):
            raise ValueError("angular motion rate must have a finite magnitude")

    @property
    def magnitude_degrees_per_second(self) -> float:
        return math.hypot(
            self.rotation_degrees_per_second,
            self.horizontal_degrees_per_second,
            self.vertical_degrees_per_second,
        )


@dataclass(frozen=True, slots=True)
class MeasuredCameraShake:
    residual: AngularMotionRate
    smoothed_motion: AngularMotionRate
    status: Literal["measured"] = field(default="measured", init=False)


@dataclass(frozen=True, slots=True)
class UnavailableCameraShake:
    reason: Literal["insufficient_context", "unmeasured_context"]
    status: Literal["unavailable"] = field(default="unavailable", init=False)


@dataclass(frozen=True, slots=True)
class CameraShakeObservation:
    motion: CameraMotionObservation
    shake: MeasuredCameraShake | UnavailableCameraShake
    settings: CameraShakeSettings


def _angular_rate(
    observation: CameraMotionObservation, settings: CameraShakeSettings
) -> AngularMotionRate | None:
    if isinstance(observation.measurement, UnmeasuredCameraMotion):
        return None
    transform = observation.measurement.transform
    degrees_per_pixel = settings.horizontal_field_of_view_degrees / observation.frame_width_pixels
    frames_per_second = observation.settings.frames_per_second
    return AngularMotionRate(
        transform.rotation_degrees * frames_per_second,
        transform.horizontal_translation_pixels * degrees_per_pixel * frames_per_second,
        transform.vertical_translation_pixels * degrees_per_pixel * frames_per_second,
    )


def _filtered_observation(
    observation: CameraMotionObservation,
    window: deque[CameraMotionObservation],
    settings: CameraShakeSettings,
) -> CameraShakeObservation:
    half_window = settings.half_window_pairs
    if (
        observation.pair_index - window[0].pair_index < half_window
        or window[-1].pair_index - observation.pair_index < half_window
    ):
        return CameraShakeObservation(
            observation, UnavailableCameraShake("insufficient_context"), settings
        )
    rates: list[AngularMotionRate] = []
    for neighbor in window:
        rate = _angular_rate(neighbor, settings)
        if rate is None:
            return CameraShakeObservation(
                observation, UnavailableCameraShake("unmeasured_context"), settings
            )
        rates.append(rate)
    center_rate = rates[half_window]
    smoothed_rate = AngularMotionRate(
        math.fsum(rate.rotation_degrees_per_second / len(rates) for rate in rates),
        math.fsum(rate.horizontal_degrees_per_second / len(rates) for rate in rates),
        math.fsum(rate.vertical_degrees_per_second / len(rates) for rate in rates),
    )
    residual = AngularMotionRate(
        center_rate.rotation_degrees_per_second - smoothed_rate.rotation_degrees_per_second,
        center_rate.horizontal_degrees_per_second - smoothed_rate.horizontal_degrees_per_second,
        center_rate.vertical_degrees_per_second - smoothed_rate.vertical_degrees_per_second,
    )
    return CameraShakeObservation(
        observation, MeasuredCameraShake(residual, smoothed_rate), settings
    )


def filter_camera_shake(
    observations: Iterable[CameraMotionObservation], *, settings: CameraShakeSettings
) -> Iterator[CameraShakeObservation]:
    """Yield one result per pair using bounded lookahead, with no gap filling.

    Input must be the complete ordered motion stream, including unmeasured
    pairs. Discontinuous or mixed input raises rather than joining across gaps.
    The first/last half-window cannot be assessed, nor can any window containing
    an unmeasured pair. The caller retains ownership of the input iterator.
    """
    window: deque[CameraMotionObservation] = deque()
    next_output_index = 0
    previous_observation: CameraMotionObservation | None = None
    for observation in observations:
        expected_pair_index = (
            0 if previous_observation is None else previous_observation.pair_index + 1
        )
        if observation.pair_index != expected_pair_index:
            raise ValueError("motion pairs must start at zero and be consecutive")
        if previous_observation is not None and (
            observation.settings != previous_observation.settings
            or observation.frame_width_pixels != previous_observation.frame_width_pixels
            or observation.start_seconds != previous_observation.end_seconds
        ):
            raise ValueError("motion pairs must have consistent settings, geometry, and timestamps")
        window.append(observation)
        previous_observation = observation
        if observation.pair_index >= next_output_index + settings.half_window_pairs:
            center = window[next_output_index - window[0].pair_index]
            yield _filtered_observation(center, window, settings)
            next_output_index += 1
            while window and window[0].pair_index < next_output_index - settings.half_window_pairs:
                window.popleft()
    # Right-edge observations are unavailable: do not invent future rates.
    for observation in window:
        if observation.pair_index >= next_output_index:
            yield CameraShakeObservation(
                observation, UnavailableCameraShake("insufficient_context"), settings
            )
