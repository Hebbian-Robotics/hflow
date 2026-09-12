"""Typed views of deterministic camera evidence, with explicit measurement units."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from hflow.steps import MeasurementValue


def finite_measurement(value: object, *, minimum: float = 0.0, maximum: float = math.inf) -> float:
    """Parse a numeric measurement without accepting booleans or nonfinite values."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("measurement must be numeric")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("measurement is outside its range") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError("measurement is outside its range")
    return number


@dataclass(frozen=True)
class CameraQualityEvidence:
    """Sampled-frame percentages and detected freeze duration, without QC policy.

    These measurements describe the processed camera stream. They do not imply
    complete coverage of the original recording or define assessment duration.
    """

    black_frame_percent: float
    clipped_highlight_percent: float
    crushed_shadow_percent: float
    frozen_seconds: float

    def __post_init__(self) -> None:
        for value in (
            self.black_frame_percent,
            self.clipped_highlight_percent,
            self.crushed_shadow_percent,
        ):
            finite_measurement(value, maximum=100.0)
        finite_measurement(self.frozen_seconds)

    @staticmethod
    def measurement_names(camera_topic: str) -> tuple[str, ...]:
        return tuple(
            f"{camera_topic}/{name}"
            for name in (
                "black_frame_pct",
                "clipped_highlight_pct",
                "crushed_shadow_pct",
                "freeze_total_s",
            )
        )

    @classmethod
    def from_measurements(
        cls, measurements: Mapping[str, MeasurementValue], camera_topic: str
    ) -> "CameraQualityEvidence":
        black, highlights, shadows, frozen = (
            measurements[name] for name in cls.measurement_names(camera_topic)
        )
        return cls(
            finite_measurement(black, maximum=100.0),
            finite_measurement(highlights, maximum=100.0),
            finite_measurement(shadows, maximum=100.0),
            finite_measurement(frozen),
        )
