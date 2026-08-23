"""Camera-shake measurement: separating how the camera moved from what moved.

Frame differencing cannot answer this. A static camera watching a busy scene and
a shaking camera watching a still one both produce large per-pixel differences,
and ITU-T P.910's own reference numbers rank a static-camera talking head above
a genuine camera pan -- so a difference metric asked to judge stability answers
about the subject instead.

This measures the camera. Sparse features are tracked between adjacent frames,
a similarity transform is fitted to the tracks (RANSAC discarding the ones that
disagree, which is what rejects independently moving subjects), and the fitted
rotation and translation are converted to angular rates using the lens's field
of view. Those rates are then split in time: motion below roughly 1 Hz is where
deliberate camera movement lives, and what remains above it is shake.

A pair is unstable when its shake exceeds both the deliberate motion in the same
pair and the instrument's own resolution floor. That is threshold-free on
purpose: the footage is compared against itself and against what the instrument
can actually resolve, because no standards body publishes a shake limit in
degrees per second and any number chosen here would be taste wearing a unit.

Requires the ``motion`` extra (``pip install 'hflow[motion]'``) for OpenCV.
"""

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from hflow.ffmpeg import luma_frames

# The high-pass boundary between deliberate camera movement and shake. 1 Hz is
# the figure the camera-stability literature converged on (Niskanen et al.,
# ICME 2006) and it is not tunable: moving it changes what the words mean.
_SHAKE_HIGH_PASS_HZ = 1.0

# Feature-tracking geometry. A grid rather than corner detection so coverage is
# uniform across the frame -- corners cluster on whatever is textured, which on
# egocentric footage is often the moving subject.
#
# Spacing scales with the frame so the number of tracks, and so the chance of
# clearing the minimum under heavy motion, does not depend on resolution. At a
# fixed 32 px a 320-wide frame yielded only 80 tracks and lost four fifths of
# its pairs to the minimum, while the same footage at 16 px kept six in seven.
_TRACK_ROWS_TARGET = 20
_MINIMUM_TRACK_SPACING_PX = 8
_TRACK_WINDOW_PX = 21
_TRACK_PYRAMID_LEVELS = 3
# A track is kept only if following it back lands within this distance of where
# it started. Forward-backward disagreement is the cheapest reliable way to
# discard a track that latched onto the wrong texture.
_FORWARD_BACKWARD_TOLERANCE_PX = 1.0
_MINIMUM_TRACKS_FOR_A_FIT = 12
_RANSAC_REPROJECTION_TOLERANCE_PX = 3.0

DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES = 90.0


class MotionExtraNotInstalledError(RuntimeError):
    pass


def _import_cv2() -> ModuleType:
    """OpenCV, imported on use so the core install never needs it.

    Typed as the module rather than the package so a missing extra is one clear
    error at the call site instead of an ImportError at ``import hflow``.
    """
    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by hand
        raise MotionExtraNotInstalledError(
            "camera-motion measurement needs OpenCV, which ships in the optional "
            "'motion' extra: pip install 'hflow[motion]' (or uv add "
            "'hflow[motion]'). Pyramidal Lucas-Kanade optical flow and the "
            "RANSAC similarity fit have no numpy-only equivalent that would "
            "measure the same thing."
        ) from error
    return cv2


@dataclass(frozen=True)
class CameraMotion:
    """Angular motion of the camera over one clip, split into intent and shake.

    ``unstable_share`` is over ``measured_s`` -- the footage a transform could
    actually be fitted to. ``unclassified_s`` is the rest: fit failures and the
    seed pairs the high-pass filter needs before it can report. Both are
    published because a share over an unstated denominator is not a measurement,
    and low coverage is the case where the share means least.
    """

    measured_s: float
    unclassified_s: float
    unstable_s: float
    unstable_share: float
    shake_rate_p50_deg_per_s: float
    shake_rate_p95_deg_per_s: float
    intentional_rate_p50_deg_per_s: float
    resolution_floor_deg_per_s: float
    median_inlier_ratio: float
    degrees_per_pixel: float
    frames_per_second: float
    unstable_pair_indices: tuple[int, ...]


def _grid_points(frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    spacing = max(_MINIMUM_TRACK_SPACING_PX, min(height, width) // _TRACK_ROWS_TARGET)
    rows = np.arange(spacing // 2, height, spacing)
    columns = np.arange(spacing // 2, width, spacing)
    grid = np.stack(np.meshgrid(columns, rows), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32).reshape(-1, 1, 2)


def _fit_pair(
    cv2: ModuleType, earlier: np.ndarray, later: np.ndarray
) -> tuple[float, float, float, float]:
    """(rotation_degrees, dx_px, dy_px, inlier_ratio) for one frame pair.

    Translation is returned signed, per axis. A magnitude here would be a bug:
    the high-pass downstream needs the sign to tell an oscillation from a drift,
    and a rectified series has almost no energy left at the shake frequency.

    Raises :class:`ValueError` when no transform can be fitted, which the caller
    records as unclassified footage rather than as zero motion -- imputing zero
    would report a shaking camera as steady exactly where measurement failed.
    """
    points = _grid_points(earlier.shape)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        earlier,
        later,
        points,
        None,
        winSize=(_TRACK_WINDOW_PX, _TRACK_WINDOW_PX),
        maxLevel=_TRACK_PYRAMID_LEVELS,
    )
    back_tracked, back_status, _ = cv2.calcOpticalFlowPyrLK(
        later,
        earlier,
        tracked,
        None,
        winSize=(_TRACK_WINDOW_PX, _TRACK_WINDOW_PX),
        maxLevel=_TRACK_PYRAMID_LEVELS,
    )
    kept = (
        status.reshape(-1).astype(bool)
        & back_status.reshape(-1).astype(bool)
        & (
            np.linalg.norm((back_tracked - points).reshape(-1, 2), axis=1)
            <= _FORWARD_BACKWARD_TOLERANCE_PX
        )
    )
    if int(np.count_nonzero(kept)) < _MINIMUM_TRACKS_FOR_A_FIT:
        raise ValueError("too few surviving tracks to fit a transform")

    source = points.reshape(-1, 2)[kept]
    destination = tracked.reshape(-1, 2)[kept]
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJECTION_TOLERANCE_PX,
    )
    if matrix is None:
        raise ValueError("no similarity transform fitted the surviving tracks")
    # A partial affine is [s*cos, -s*sin, tx; s*sin, s*cos, ty], so the rotation
    # comes straight out of the top row and the scale divides out.
    rotation_degrees = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    inlier_ratio = float(np.count_nonzero(inliers) / len(source)) if inliers is not None else 1.0
    return rotation_degrees, float(matrix[0, 2]), float(matrix[1, 2]), inlier_ratio


def _high_pass(rates: np.ndarray, frames_per_second: float) -> np.ndarray:
    """What is left of a rate series above the shake boundary.

    A moving average over one high-pass period is the deliberate component, and
    subtracting it leaves the shake. Crude by design: a sharper filter would
    need a longer seed and the boundary is a soft one anyway.
    """
    window = max(1, round(frames_per_second / _SHAKE_HIGH_PASS_HZ))
    if window < 2:
        return rates
    kernel = np.ones(window) / window
    padded = np.pad(rates, (window // 2, window - 1 - window // 2), mode="edge")
    deliberate = np.convolve(padded, kernel, mode="valid")
    return rates - deliberate


def measure_camera_motion(
    video: Path,
    *,
    frames_per_second: float,
    horizontal_field_of_view_degrees: float = DEFAULT_HORIZONTAL_FIELD_OF_VIEW_DEGREES,
) -> CameraMotion | None:
    """Measure one clip's camera motion. ``None`` when nothing could be fitted.

    ``frames_per_second`` converts per-pair motion into a rate, so pass the
    stream's real rate (its median log-time interval) rather than a nominal one.
    ``horizontal_field_of_view_degrees`` converts pixels to degrees; the default
    suits an action camera, and a phone-like 65 degrees or a 118-degree wide lens
    scales the translation term by about 1.35x either way. Rotation is unaffected
    by it, so a wrong value changes the numbers without inverting the comparison.
    """
    cv2 = _import_cv2()
    if frames_per_second <= 0:
        raise ValueError(f"frames_per_second must be positive, got {frames_per_second}")

    # Signed, per axis: the high-pass below needs the sign to tell an
    # oscillation from a drift. Rectifying first leaves a series whose energy
    # sits at twice the shake frequency and is far smaller, which reads heavy
    # shake as steady footage.
    signed_axes: list[tuple[float, float, float]] = []
    inlier_ratios: list[float] = []
    measurable: list[bool] = []
    degrees_per_pixel = 0.0

    with luma_frames(video) as frames:
        previous: np.ndarray | None = None
        for frame in frames:
            if previous is None:
                previous = frame
                degrees_per_pixel = horizontal_field_of_view_degrees / frame.shape[1]
                continue
            try:
                rotation_degrees, dx_px, dy_px, inlier_ratio = _fit_pair(cv2, previous, frame)
            except ValueError:
                measurable.append(False)
                signed_axes.append((0.0, 0.0, 0.0))
            else:
                measurable.append(True)
                signed_axes.append(
                    (
                        rotation_degrees * frames_per_second,
                        dx_px * degrees_per_pixel * frames_per_second,
                        dy_px * degrees_per_pixel * frames_per_second,
                    )
                )
                inlier_ratios.append(inlier_ratio)
            previous = frame

    if not measurable:
        return None
    measurable_mask = np.asarray(measurable, dtype=bool)
    seconds_per_pair = 1.0 / frames_per_second
    if not np.any(measurable_mask):
        return CameraMotion(
            measured_s=0.0,
            unclassified_s=len(measurable) * seconds_per_pair,
            unstable_s=0.0,
            unstable_share=0.0,
            shake_rate_p50_deg_per_s=0.0,
            shake_rate_p95_deg_per_s=0.0,
            intentional_rate_p50_deg_per_s=0.0,
            resolution_floor_deg_per_s=degrees_per_pixel * frames_per_second,
            median_inlier_ratio=0.0,
            degrees_per_pixel=degrees_per_pixel,
            frames_per_second=frames_per_second,
            unstable_pair_indices=(),
        )

    # High-pass each signed axis independently, then take the magnitude of what
    # is left. Filtering the axes and combining afterwards is what keeps an 8 Hz
    # wobble visible: combining first would rectify it into a near-constant.
    axis_rates = np.asarray(signed_axes)
    shake_axes = np.stack(
        [_high_pass(axis_rates[:, axis], frames_per_second) for axis in range(3)], axis=1
    )
    shake_rates = np.linalg.norm(shake_axes, axis=1)
    intentional_rates = np.linalg.norm(axis_rates - shake_axes, axis=1)

    # The floor is the instrument stating its own resolution: motion smaller
    # than one pixel between frames is not something this can distinguish from
    # nothing, so it must not be reported as instability.
    resolution_floor = degrees_per_pixel * frames_per_second
    unstable = (
        measurable_mask & (shake_rates > intentional_rates) & (shake_rates > resolution_floor)
    )
    measured_pair_count = int(np.count_nonzero(measurable_mask))
    measured_s = measured_pair_count * seconds_per_pair
    return CameraMotion(
        measured_s=measured_s,
        unclassified_s=(len(measurable) - measured_pair_count) * seconds_per_pair,
        unstable_s=int(np.count_nonzero(unstable)) * seconds_per_pair,
        unstable_share=float(np.count_nonzero(unstable) / measured_pair_count),
        shake_rate_p50_deg_per_s=float(np.percentile(shake_rates[measurable_mask], 50)),
        shake_rate_p95_deg_per_s=float(np.percentile(shake_rates[measurable_mask], 95)),
        intentional_rate_p50_deg_per_s=float(np.percentile(intentional_rates[measurable_mask], 50)),
        resolution_floor_deg_per_s=resolution_floor,
        median_inlier_ratio=float(np.median(inlier_ratios)) if inlier_ratios else 0.0,
        degrees_per_pixel=degrees_per_pixel,
        frames_per_second=frames_per_second,
        unstable_pair_indices=tuple(int(index) for index in np.flatnonzero(unstable)),
    )
