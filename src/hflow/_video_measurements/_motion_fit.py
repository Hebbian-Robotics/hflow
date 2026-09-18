"""Shared frame-pair estimator and evidence, independent of acceptance policy."""

import math
from dataclasses import dataclass, field
from types import ModuleType
from typing import Literal

import numpy as np

from ._motion_settings import CameraMotionAlgorithmSettings

_DEFAULT_ALGORITHM_SETTINGS = CameraMotionAlgorithmSettings()
_MINIMUM_TRACKS_FOR_A_FIT = 2


@dataclass(frozen=True, slots=True)
class CameraMotionTransform:
    """Earlier-to-later image similarity transform, about the image origin."""

    horizontal_translation_pixels: float
    vertical_translation_pixels: float
    rotation_degrees: float
    scale: float


@dataclass(frozen=True, slots=True)
class MotionFitEvidence:
    attempted_track_count: int
    retained_track_count: int
    inlier_count: int
    median_inlier_reprojection_error_pixels: float | None

    @property
    def track_retention_ratio(self) -> float:
        return (
            self.retained_track_count / self.attempted_track_count
            if self.attempted_track_count
            else 0.0
        )

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.retained_track_count if self.retained_track_count else 0.0


@dataclass(frozen=True, slots=True)
class MeasuredCameraMotion:
    transform: CameraMotionTransform
    evidence: MotionFitEvidence
    status: Literal["measured"] = field(default="measured", init=False)


MotionUnavailableReason = Literal["insufficient_tracks", "fit_failed", "nonfinite_fit"]


@dataclass(frozen=True, slots=True)
class UnmeasuredCameraMotion:
    reason: MotionUnavailableReason
    evidence: MotionFitEvidence
    status: Literal["unmeasured"] = field(default="unmeasured", init=False)


FrameMotionResult = MeasuredCameraMotion | UnmeasuredCameraMotion


def _grid_points(
    frame_shape: tuple[int, int], settings: CameraMotionAlgorithmSettings
) -> np.ndarray:
    frame_height, frame_width = frame_shape
    track_spacing_pixels = max(
        settings.minimum_track_spacing_pixels,
        min(frame_height, frame_width) // settings.target_grid_rows,
    )
    row_coordinates = np.arange(track_spacing_pixels // 2, frame_height, track_spacing_pixels)
    column_coordinates = np.arange(track_spacing_pixels // 2, frame_width, track_spacing_pixels)
    grid_points = np.stack(np.meshgrid(column_coordinates, row_coordinates), axis=-1).reshape(-1, 2)
    return grid_points.astype(np.float32).reshape(-1, 1, 2)


def fit_frame_motion(
    cv2: ModuleType,
    earlier_frame: np.ndarray,
    later_frame: np.ndarray,
    *,
    settings: CameraMotionAlgorithmSettings = _DEFAULT_ALGORITHM_SETTINGS,
    minimum_track_count: int = _MINIMUM_TRACKS_FOR_A_FIT,
) -> FrameMotionResult:
    """Track in both directions and fit a robust similarity with observable evidence."""
    settings.validate_frame_dimensions(*earlier_frame.shape)
    source_points = _grid_points(earlier_frame.shape, settings)
    attempted_track_count = len(source_points)
    empty_evidence = MotionFitEvidence(attempted_track_count, 0, 0, None)
    if attempted_track_count < minimum_track_count:
        return UnmeasuredCameraMotion("insufficient_tracks", empty_evidence)
    tracked_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        earlier_frame,
        later_frame,
        source_points,
        None,
        winSize=(settings.tracking_window_width_pixels, settings.tracking_window_height_pixels),
        maxLevel=settings.maximum_pyramid_level,
        criteria=(
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            settings.tracking_maximum_iterations,
            settings.tracking_convergence_epsilon_pixels,
        ),
        flags=0,
        minEigThreshold=settings.tracking_minimum_eigenvalue,
    )
    if tracked_points is None or forward_status is None:
        return UnmeasuredCameraMotion("insufficient_tracks", empty_evidence)
    back_tracked_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        later_frame,
        earlier_frame,
        tracked_points,
        None,
        winSize=(settings.tracking_window_width_pixels, settings.tracking_window_height_pixels),
        maxLevel=settings.maximum_pyramid_level,
        criteria=(
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            settings.tracking_maximum_iterations,
            settings.tracking_convergence_epsilon_pixels,
        ),
        flags=0,
        minEigThreshold=settings.tracking_minimum_eigenvalue,
    )
    if back_tracked_points is None or backward_status is None:
        return UnmeasuredCameraMotion("insufficient_tracks", empty_evidence)
    retained_track_mask = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.all(np.isfinite(tracked_points.reshape(-1, 2)), axis=1)
        & (
            np.linalg.norm((back_tracked_points - source_points).reshape(-1, 2), axis=1)
            <= settings.forward_backward_tolerance_pixels
        )
    )
    retained_track_count = int(np.count_nonzero(retained_track_mask))
    evidence = MotionFitEvidence(attempted_track_count, retained_track_count, 0, None)
    if retained_track_count < minimum_track_count:
        return UnmeasuredCameraMotion("insufficient_tracks", evidence)
    retained_source_points = source_points.reshape(-1, 2)[retained_track_mask]
    retained_destination_points = tracked_points.reshape(-1, 2)[retained_track_mask]
    transformation_matrix, inliers = cv2.estimateAffinePartial2D(
        retained_source_points,
        retained_destination_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=settings.ransac_reprojection_tolerance_pixels,
        maxIters=settings.ransac_maximum_iterations,
        confidence=settings.ransac_confidence,
        refineIters=settings.refinement_maximum_iterations,
    )
    if transformation_matrix is None:
        return UnmeasuredCameraMotion("fit_failed", evidence)
    if not np.all(np.isfinite(transformation_matrix)):
        return UnmeasuredCameraMotion("nonfinite_fit", evidence)
    inlier_mask = (
        inliers.reshape(-1).astype(bool)
        if inliers is not None
        else np.ones(retained_track_count, dtype=bool)
    )
    predicted_points = (
        retained_source_points @ transformation_matrix[:, :2].T + transformation_matrix[:, 2]
    )
    inlier_errors = np.linalg.norm(
        predicted_points[inlier_mask] - retained_destination_points[inlier_mask], axis=1
    )
    evidence = MotionFitEvidence(
        attempted_track_count,
        retained_track_count,
        int(np.count_nonzero(inlier_mask)),
        float(np.median(inlier_errors)) if len(inlier_errors) else None,
    )
    return MeasuredCameraMotion(
        CameraMotionTransform(
            horizontal_translation_pixels=float(transformation_matrix[0, 2]),
            vertical_translation_pixels=float(transformation_matrix[1, 2]),
            rotation_degrees=float(
                np.degrees(np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0]))
            ),
            scale=math.hypot(transformation_matrix[0, 0], transformation_matrix[1, 0]),
        ),
        evidence,
    )
