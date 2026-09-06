"""Camera-motion measurements that separate deliberate movement from shake."""

import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from ._field_guards import require_float
from ._raw_frames import LUMA_FRAME_FILTER_GRAPH, luma_frames
from ._toolchain import VideoMeasurementToolchain

CAMERA_MOTION_DEFINITION_VERSION = "camera-motion/v1"
DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES = 90.0

_SHAKE_HIGH_PASS_HZ = 1.0
_TRACK_ROWS_TARGET = 20
_MINIMUM_TRACK_SPACING_PIXELS = 8
_TRACK_WINDOW_PIXELS = 21
_TRACK_PYRAMID_LEVELS = 3
_FORWARD_BACKWARD_TOLERANCE_PIXELS = 1.0
_MINIMUM_TRACKS_FOR_A_FIT = 12
_RANSAC_REPROJECTION_TOLERANCE_PIXELS = 3.0
_MINIMUM_VIDEO_FRAME_COUNT = 2


class MotionExtraNotInstalledError(RuntimeError):
    """OpenCV is unavailable, so camera motion cannot be measured."""


def _import_cv2() -> ModuleType:
    """Import OpenCV only when camera motion is requested."""
    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by hand
        raise MotionExtraNotInstalledError(
            "camera-motion measurement needs OpenCV, which ships in HFlow's "
            "optional 'motion' extra: pip install 'hflow[motion]' (or uv add "
            "'hflow[motion]'). Pyramidal Lucas-Kanade optical flow and the "
            "RANSAC similarity fit have no numpy-only equivalent that would "
            "measure the same thing."
        ) from error
    return cv2


@dataclass(frozen=True)
class CameraMotionSettings:
    """Inputs that define how pixel movement becomes an angular rate."""

    frames_per_second: float
    horizontal_field_of_view_degrees: float = DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES
    # A pair counts as unstable only when its shake rate clears this as well as
    # the instrument's resolution floor. Zero keeps the floor alone.
    minimum_shake_degrees_per_second: float = 0.0

    def __post_init__(self) -> None:
        require_float(self.frames_per_second, "frames_per_second")
        require_float(self.horizontal_field_of_view_degrees, "horizontal_field_of_view_degrees")
        require_float(self.minimum_shake_degrees_per_second, "minimum_shake_degrees_per_second")
        if (
            not math.isfinite(self.minimum_shake_degrees_per_second)
            or self.minimum_shake_degrees_per_second < 0
        ):
            raise ValueError(
                "minimum_shake_degrees_per_second must be finite and non-negative, "
                f"got {self.minimum_shake_degrees_per_second}"
            )
        if not math.isfinite(self.frames_per_second) or self.frames_per_second <= 0:
            raise ValueError(
                f"frames_per_second must be finite and positive, got {self.frames_per_second}"
            )
        if (
            not math.isfinite(self.horizontal_field_of_view_degrees)
            or not 0 < self.horizontal_field_of_view_degrees <= 360
        ):
            raise ValueError(
                "horizontal_field_of_view_degrees must be finite and in (0, 360], "
                f"got {self.horizontal_field_of_view_degrees}"
            )


@dataclass(frozen=True)
class CameraMotionProvenance:
    """The definition, settings, and decoder implementations behind a result."""

    measurement_definition_version: str
    ffmpeg_version: str
    ffprobe_version: str
    opencv_version: str
    luma_decode_filter_graph: str
    settings: CameraMotionSettings


@dataclass(frozen=True)
class InsufficientVideoFrames:
    """A recoverable result for a video with no adjacent frame pair."""

    observed_frame_count: int
    minimum_required_frame_count: int
    provenance: CameraMotionProvenance


@dataclass(frozen=True)
class CameraMotionMeasurements:
    """Angular camera movement over every adjacent frame pair in a video."""

    measured_seconds: float
    unclassified_seconds: float
    unstable_seconds: float
    unstable_share: float
    shake_rate_p50_degrees_per_second: float
    shake_rate_p95_degrees_per_second: float
    intentional_rate_p50_degrees_per_second: float
    resolution_floor_degrees_per_second: float
    median_inlier_ratio: float
    degrees_per_pixel: float
    frames_per_second: float
    unstable_pair_indices: tuple[int, ...]
    provenance: CameraMotionProvenance


CameraMotionResult = CameraMotionMeasurements | InsufficientVideoFrames


def _grid_points(frame_shape: tuple[int, int]) -> np.ndarray:
    frame_height, frame_width = frame_shape
    track_spacing_pixels = max(
        _MINIMUM_TRACK_SPACING_PIXELS,
        min(frame_height, frame_width) // _TRACK_ROWS_TARGET,
    )
    row_coordinates = np.arange(track_spacing_pixels // 2, frame_height, track_spacing_pixels)
    column_coordinates = np.arange(track_spacing_pixels // 2, frame_width, track_spacing_pixels)
    grid_points = np.stack(np.meshgrid(column_coordinates, row_coordinates), axis=-1).reshape(-1, 2)
    return grid_points.astype(np.float32).reshape(-1, 1, 2)


def _fit_frame_pair(
    cv2: ModuleType, earlier_frame: np.ndarray, later_frame: np.ndarray
) -> tuple[float, float, float, float]:
    """Fit rotation, translation, and inlier coverage for one frame pair."""
    source_points = _grid_points(earlier_frame.shape)
    tracked_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        earlier_frame,
        later_frame,
        source_points,
        None,
        winSize=(_TRACK_WINDOW_PIXELS, _TRACK_WINDOW_PIXELS),
        maxLevel=_TRACK_PYRAMID_LEVELS,
    )
    back_tracked_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        later_frame,
        earlier_frame,
        tracked_points,
        None,
        winSize=(_TRACK_WINDOW_PIXELS, _TRACK_WINDOW_PIXELS),
        maxLevel=_TRACK_PYRAMID_LEVELS,
    )
    retained_track_mask = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & (
            np.linalg.norm((back_tracked_points - source_points).reshape(-1, 2), axis=1)
            <= _FORWARD_BACKWARD_TOLERANCE_PIXELS
        )
    )
    if int(np.count_nonzero(retained_track_mask)) < _MINIMUM_TRACKS_FOR_A_FIT:
        raise ValueError("too few surviving tracks to fit a transform")

    retained_source_points = source_points.reshape(-1, 2)[retained_track_mask]
    retained_destination_points = tracked_points.reshape(-1, 2)[retained_track_mask]
    transformation_matrix, inliers = cv2.estimateAffinePartial2D(
        retained_source_points,
        retained_destination_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJECTION_TOLERANCE_PIXELS,
    )
    if transformation_matrix is None:
        raise ValueError("no similarity transform fitted the surviving tracks")
    rotation_degrees = float(
        np.degrees(np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0]))
    )
    inlier_ratio = (
        float(np.count_nonzero(inliers) / len(retained_source_points))
        if inliers is not None
        else 1.0
    )
    return (
        rotation_degrees,
        float(transformation_matrix[0, 2]),
        float(transformation_matrix[1, 2]),
        inlier_ratio,
    )


def _high_pass_filter(angular_rates: np.ndarray, frames_per_second: float) -> np.ndarray:
    """Return the part of a signed rate series above the shake boundary."""
    moving_average_window = max(1, round(frames_per_second / _SHAKE_HIGH_PASS_HZ))
    if moving_average_window < 2:
        return angular_rates
    moving_average_kernel = np.ones(moving_average_window) / moving_average_window
    padded_angular_rates = np.pad(
        angular_rates,
        (
            moving_average_window // 2,
            moving_average_window - 1 - moving_average_window // 2,
        ),
        mode="edge",
    )
    deliberate_angular_rates = np.convolve(
        padded_angular_rates, moving_average_kernel, mode="valid"
    )
    return angular_rates - deliberate_angular_rates


def _opencv_version(cv2: ModuleType) -> str:
    opencv_version = getattr(cv2, "__version__", None)
    if not isinstance(opencv_version, str) or not opencv_version.strip():
        raise RuntimeError("OpenCV did not report a version string")
    return opencv_version


def measure_camera_motion(
    video: Path,
    *,
    toolchain: VideoMeasurementToolchain,
    settings: CameraMotionSettings,
) -> CameraMotionResult:
    """Measure camera shake or report that no adjacent frame pair existed."""
    cv2 = _import_cv2()
    provenance = CameraMotionProvenance(
        measurement_definition_version=CAMERA_MOTION_DEFINITION_VERSION,
        ffmpeg_version=toolchain.ffmpeg_version,
        ffprobe_version=toolchain.ffprobe_version,
        opencv_version=_opencv_version(cv2),
        luma_decode_filter_graph=LUMA_FRAME_FILTER_GRAPH,
        settings=settings,
    )
    signed_angular_rate_axes: list[tuple[float, float, float]] = []
    inlier_ratios: list[float] = []
    measurable_pairs: list[bool] = []
    degrees_per_pixel = 0.0
    observed_frame_count = 0

    with luma_frames(video, toolchain=toolchain) as frames:
        previous_frame: np.ndarray | None = None
        for frame in frames:
            observed_frame_count += 1
            if previous_frame is None:
                previous_frame = frame
                degrees_per_pixel = settings.horizontal_field_of_view_degrees / frame.shape[1]
                continue
            try:
                rotation_degrees, horizontal_pixels, vertical_pixels, inlier_ratio = (
                    _fit_frame_pair(cv2, previous_frame, frame)
                )
            except ValueError:
                measurable_pairs.append(False)
                signed_angular_rate_axes.append((0.0, 0.0, 0.0))
            else:
                measurable_pairs.append(True)
                signed_angular_rate_axes.append(
                    (
                        rotation_degrees * settings.frames_per_second,
                        horizontal_pixels * degrees_per_pixel * settings.frames_per_second,
                        vertical_pixels * degrees_per_pixel * settings.frames_per_second,
                    )
                )
                inlier_ratios.append(inlier_ratio)
            previous_frame = frame

    if not measurable_pairs:
        return InsufficientVideoFrames(
            observed_frame_count=observed_frame_count,
            minimum_required_frame_count=_MINIMUM_VIDEO_FRAME_COUNT,
            provenance=provenance,
        )

    measurable_pair_mask = np.asarray(measurable_pairs, dtype=bool)
    seconds_per_pair = 1.0 / settings.frames_per_second
    resolution_floor = degrees_per_pixel * settings.frames_per_second
    if not np.any(measurable_pair_mask):
        return CameraMotionMeasurements(
            measured_seconds=0.0,
            unclassified_seconds=len(measurable_pairs) * seconds_per_pair,
            unstable_seconds=0.0,
            unstable_share=0.0,
            shake_rate_p50_degrees_per_second=0.0,
            shake_rate_p95_degrees_per_second=0.0,
            intentional_rate_p50_degrees_per_second=0.0,
            resolution_floor_degrees_per_second=resolution_floor,
            median_inlier_ratio=0.0,
            degrees_per_pixel=degrees_per_pixel,
            frames_per_second=settings.frames_per_second,
            unstable_pair_indices=(),
            provenance=provenance,
        )

    angular_rate_axes = np.asarray(signed_angular_rate_axes)
    shake_rate_axes = np.stack(
        [
            _high_pass_filter(angular_rate_axes[:, axis_index], settings.frames_per_second)
            for axis_index in range(3)
        ],
        axis=1,
    )
    shake_rates = np.linalg.norm(shake_rate_axes, axis=1)
    intentional_rates = np.linalg.norm(angular_rate_axes - shake_rate_axes, axis=1)
    # The floor a pair's shake must clear is the instrument's own resolution
    # (one pixel per frame), raised to the caller's minimum when they set one:
    # sub-degree jitter is real but not what anyone means by a shaky camera.
    shake_floor = max(resolution_floor, settings.minimum_shake_degrees_per_second)
    unstable_pair_mask = (
        measurable_pair_mask & (shake_rates > intentional_rates) & (shake_rates > shake_floor)
    )
    measured_pair_count = int(np.count_nonzero(measurable_pair_mask))
    unstable_pair_count = int(np.count_nonzero(unstable_pair_mask))
    return CameraMotionMeasurements(
        measured_seconds=measured_pair_count * seconds_per_pair,
        unclassified_seconds=(len(measurable_pairs) - measured_pair_count) * seconds_per_pair,
        unstable_seconds=unstable_pair_count * seconds_per_pair,
        unstable_share=float(unstable_pair_count / measured_pair_count),
        shake_rate_p50_degrees_per_second=float(
            np.percentile(shake_rates[measurable_pair_mask], 50)
        ),
        shake_rate_p95_degrees_per_second=float(
            np.percentile(shake_rates[measurable_pair_mask], 95)
        ),
        intentional_rate_p50_degrees_per_second=float(
            np.percentile(intentional_rates[measurable_pair_mask], 50)
        ),
        resolution_floor_degrees_per_second=resolution_floor,
        median_inlier_ratio=float(np.median(inlier_ratios)) if inlier_ratios else 0.0,
        degrees_per_pixel=degrees_per_pixel,
        frames_per_second=settings.frames_per_second,
        unstable_pair_indices=tuple(int(index) for index in np.flatnonzero(unstable_pair_mask)),
        provenance=provenance,
    )
