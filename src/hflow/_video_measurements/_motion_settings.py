"""Validated numerical parameters for the similarity-motion estimator."""

from dataclasses import dataclass

from hflow._field_guards import (
    require_int_in_range,
    require_non_negative_float,
    require_positive_float,
)

_MAXIMUM_NATIVE_INTEGER = 2**31 - 1


@dataclass(frozen=True, slots=True)
class CameraMotionAlgorithmSettings:
    """Sampling, pyramidal tracking, and robust similarity fitting parameters.

    These define the measurement, not an acceptance policy. Tracking stops at
    the iteration limit or convergence tolerance, whichever is reached first.
    The two-correspondence solver minimum is a mathematical requirement.
    """

    target_grid_rows: int = 20
    minimum_track_spacing_pixels: int = 8
    tracking_window_width_pixels: int = 21
    tracking_window_height_pixels: int = 21
    maximum_pyramid_level: int = 3
    tracking_maximum_iterations: int = 30
    tracking_convergence_epsilon_pixels: float = 0.01
    tracking_minimum_eigenvalue: float = 1e-4
    forward_backward_tolerance_pixels: float = 1.0
    ransac_reprojection_tolerance_pixels: float = 3.0
    ransac_maximum_iterations: int = 2000
    ransac_confidence: float = 0.99
    refinement_maximum_iterations: int = 10

    def __post_init__(self) -> None:
        for field_name in (
            "target_grid_rows",
            "minimum_track_spacing_pixels",
            "ransac_maximum_iterations",
        ):
            require_int_in_range(
                getattr(self, field_name), field_name, minimum=1, maximum=_MAXIMUM_NATIVE_INTEGER
            )
        for field_name in ("tracking_window_width_pixels", "tracking_window_height_pixels"):
            require_int_in_range(
                getattr(self, field_name), field_name, minimum=3, maximum=_MAXIMUM_NATIVE_INTEGER
            )
        # Grayscale LK allocates area * (one intensity + two derivatives)
        # using signed integer arithmetic before converting to an allocation size.
        if (
            self.tracking_window_width_pixels * self.tracking_window_height_pixels
            > _MAXIMUM_NATIVE_INTEGER // 3
        ):
            raise ValueError(
                "tracking_window_width_pixels * tracking_window_height_pixels exceeds native scratch-buffer arithmetic"
            )
        # OpenCV uses signed integer pyramid scaling and clamps LK termination
        # internally. Reject out-of-range values so recorded settings stay true.
        require_int_in_range(
            self.maximum_pyramid_level, "maximum_pyramid_level", minimum=0, maximum=30
        )
        require_int_in_range(
            self.tracking_maximum_iterations, "tracking_maximum_iterations", minimum=1, maximum=100
        )
        require_int_in_range(
            self.refinement_maximum_iterations,
            "refinement_maximum_iterations",
            minimum=0,
            maximum=_MAXIMUM_NATIVE_INTEGER,
        )
        for field_name in (
            "tracking_convergence_epsilon_pixels",
            "tracking_minimum_eigenvalue",
            "forward_backward_tolerance_pixels",
        ):
            require_non_negative_float(getattr(self, field_name), field_name)
        if self.tracking_convergence_epsilon_pixels > 10:
            raise ValueError("tracking_convergence_epsilon_pixels must be <= 10")
        require_positive_float(
            self.ransac_reprojection_tolerance_pixels, "ransac_reprojection_tolerance_pixels"
        )
        require_positive_float(self.ransac_confidence, "ransac_confidence")
        if self.ransac_confidence >= 1:
            raise ValueError("ransac_confidence must be < 1")

    def validate_frame_dimensions(self, frame_height: int, frame_width: int) -> None:
        """Check the image-dependent padded dimensions before native allocation."""
        if (
            frame_width + 2 * self.tracking_window_width_pixels > _MAXIMUM_NATIVE_INTEGER
            or frame_height + 2 * self.tracking_window_height_pixels > _MAXIMUM_NATIVE_INTEGER
        ):
            raise ValueError("tracking_window padding exceeds native frame dimensions")
