# Stream continuous camera-motion measurements

Use `hflow.camera_motion` to measure a fixed-frame-rate video while decoding it.
Each adjacent frame pair produces a typed observation with image translation,
rotation, scale, and tracking evidence, or an explicit unmeasured outcome. A
separate filter produces a continuous shake residual with bounded lookahead.
Neither API classifies footage as stable or unstable.

## Run on a local video

Install the `motion` extra, or use the repository development environment:

```bash
uv add 'hflow[motion]'
```

These APIs are implemented in the current source checkout; a previously
published HFlow version may not include them. From the repository root, use
`uv sync --locked` and run the [example](../../examples/camera_motion.py):

```bash
uv run python examples/camera_motion.py recording.mp4 --fps 30
uv run python examples/camera_motion.py recording.mp4 --fps 30 --shake
```

The example prints one JSON object per frame pair. It reads the video without
modifying it, uses CPU OpenCV, and requires FFmpeg. HFlow may download its managed
FFmpeg build. There are no model calls, credentials, or remote video uploads.
The first video stream is selected explicitly. Image coordinates follow its
coded frames; display-rotation metadata is not
applied. This keeps decoder geometry and reported motion in the same coordinate
system, including for rotation-tagged phone videos.

`--fps` must be the actual constant input frame rate. Timestamps start at the
first decoded frame and are derived from that rate, not container presentation
timestamps. Normalize variable-frame-rate footage before using this API. A
wrong frame rate changes both reported times and rates.

## Consume raw observations

```python
from pathlib import Path

from hflow.camera_motion import (
    CameraMotionStreamSettings,
    MeasuredCameraMotion,
    stream_camera_motion,
)

settings = CameraMotionStreamSettings(frames_per_second=30)
with stream_camera_motion(Path("recording.mp4"), settings=settings) as observations:
    for observation in observations:
        measurement = observation.measurement
        if isinstance(measurement, MeasuredCameraMotion):
            print(observation.start_seconds, measurement.transform)
            print(measurement.evidence.inlier_ratio)
        else:
            print(observation.start_seconds, "unmeasured", measurement.reason)
```

Always use the context manager: leaving it closes the iterator and reaps FFmpeg,
including when you break early or a consumer raises an exception. An optional
`VideoMeasurementToolchain` lets the caller supply explicit FFmpeg/FFprobe paths
and version strings instead of invoking HFlow's binary resolution.

Already-decoded frames can be passed to `iter_frame_motion(frames,
settings=settings)`. They must be nonempty two-dimensional `uint8` grayscale
arrays of constant dimensions and cadence. The producer may reuse its frame
buffer; the estimator copies the previous image before requesting another.
The caller owns any resources associated with this frame iterable.

An input with N frames yields max(0, N−1) observations. Empty or single-frame
inputs have no adjacent pairs. A pair's interval runs from its earlier frame to
its later frame; the final frame's display duration is not an extra measured
interval. Invalid frame shapes and decoding errors raise rather than being
reported as camera stability evidence.

## Interpret motion and fit evidence

`CameraMotionObservation` carries the pair index, relative start/end seconds,
frame width, frame-cadence and complete algorithm settings, and definition version
`camera-motion-stream/v1`. Its `measurement` is one of:

- `MeasuredCameraMotion`: a `CameraMotionTransform` and `MotionFitEvidence`.
- `UnmeasuredCameraMotion`: a reason and available fit evidence, with no zero
  substitute for the unavailable transform.

The transform maps earlier-image coordinates to later-image coordinates. It
reports x/y translation in pixels, rotation in degrees, and dimensionless scale.
Translation is measured about the image origin: rotation about the image center
also produces translation terms. These are image-motion parameters, not an
independent physical camera pose or calibrated yaw/pitch/roll.

Evidence includes attempted tracks, tracks retained after forward/backward
consistency checks, RANSAC inlier count, and median inlier reprojection error.
`track_retention_ratio` and `inlier_ratio` are computed properties. A zero
denominator gives ratio zero; a missing reprojection error remains null.

The estimator attempts a similarity fit with at least two retained tracks, the
solver's minimum sample size. It returns every finite fitted transform with its
evidence, including low track retention or low inlier ratios. `measured` means an
estimate exists; it does not mean the estimate is accurate or the camera is
stable. Consumers decide how much evidence they require. Failures distinguish
insufficient tracks and failed/nonfinite fits. A coherent moving foreground can
still produce a strong fit; consistency does not establish camera attribution.

## Parameter provenance

The estimator uses established pyramidal Lucas–Kanade tracking and RANSAC
similarity fitting. That does not make every parameter a research-validated
default for camera shake. The current choices are:

| Parameter | Current value | Basis |
| --- | --- | --- |
| Tracking window and maximum pyramid index | 21 × 21 pixels; index 3 (levels 0–3) | Matches [OpenCV tracking defaults](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html). |
| Tracking termination and minimum eigenvalue | 30 iterations or 0.01-pixel convergence; `1e-4` | Matches OpenCV tracking defaults, now passed explicitly; not calibrated on our footage. |
| Forward/backward tracking tolerance | Euclidean distance ≤ 1 pixel | Engineering choice with precedent in the [OpenCV tracking example](https://raw.githubusercontent.com/opencv/opencv/4.13.0/samples/python/lk_track.py). That example uses maximum coordinate error < 1, so its rule is similar, not identical. |
| RANSAC reprojection tolerance | 3 pixels | Matches [OpenCV's fitting default](https://docs.opencv.org/3.4.20/d9/d0c/group__calib3d.html). This tolerance defines which tracks support a model; it does not classify camera stability. |
| RANSAC iteration limit, confidence, refinement | 2000; 0.99; 10 iterations | Matches OpenCV fitting defaults, now passed explicitly. Confidence is a solver setting, not confidence that the camera estimate is correct. |
| Minimum correspondences | 2 | [OpenCV's similarity solver sample size](https://raw.githubusercontent.com/opencv/opencv/4.13.0/modules/calib3d/src/ptsetreg.cpp). A solvable fit can still be unreliable. |
| Grid spacing | `max(8, min(height, width) // 20)` pixels | Our sampling/computation heuristic; no task-specific calibration. |
| Shake smoothing | 31 pairs, centered | Our explicit time-scale choice; no universal research-backed boundary between intentional movement and shake. |
| Horizontal field of view | 90 degrees | Assumption for approximate angular conversion; replace with camera information when available. |

The previous 25% retention and 50% inlier acceptance gates were uncalibrated
policy choices and are absent from this raw API. The legacy aggregate's
12-track minimum is also a policy choice and remains only in that adapter.
Numerical tracking tolerances remain part of how the continuous estimate is
computed. Deterministic does not mean parameter-free: compare measurements
using the same definition, settings, and dependency versions, and record those
versions when evaluating reproducibility. Synthetic tests validate known-motion
outcomes; they do not establish optimal defaults. Parameter calibration would
require representative footage with reference motion and sensitivity testing
across resolutions, frame rates, textures, and motion magnitudes.

## Configure the estimator

All numerical tuning parameters of the current estimator are exposed through
the immutable, validated `CameraMotionAlgorithmSettings`:

```python
from hflow.camera_motion import CameraMotionAlgorithmSettings, CameraMotionStreamSettings

algorithm = CameraMotionAlgorithmSettings(
    target_grid_rows=30,
    tracking_window_width_pixels=25,
    tracking_window_height_pixels=25,
    maximum_pyramid_level=4,
    forward_backward_tolerance_pixels=0.75,
    ransac_reprojection_tolerance_pixels=2,
)
settings = CameraMotionStreamSettings(frames_per_second=30, algorithm=algorithm)
```

Omitted fields receive the defaults below. Both file and decoded-frame APIs use
this same object. Every observation includes the complete resolved object at
`observation.settings.algorithm`, including defaults, so `dataclasses.asdict`
and the example's JSON Lines output preserve it automatically. Keep the
definition and dependency versions alongside the settings when comparing runs;
the settings alone do not promise identical results across OpenCV versions or
hardware. The filter rejects a stream that changes algorithm settings midway.

| Python field | Default | Accepted values |
| --- | --- | --- |
| `target_grid_rows` | 20 | Positive integer; target divisions along the shorter image dimension. |
| `minimum_track_spacing_pixels` | 8 | Positive integer; floor on grid spacing. |
| `tracking_window_width_pixels` | 21 | Integer ≥ 3. |
| `tracking_window_height_pixels` | 21 | Integer ≥ 3. |
| `maximum_pyramid_level` | 3 | Integer 0–30; zero disables pyramid downsampling. Actual levels also depend on image/window size. |
| `tracking_maximum_iterations` | 30 | Integer 1–100 per pyramid level. |
| `tracking_convergence_epsilon_pixels` | 0.01 | Finite number 0–10. |
| `tracking_minimum_eigenvalue` | 0.0001 | Finite number ≥ 0; zero disables this texture threshold. |
| `forward_backward_tolerance_pixels` | 1 | Finite number ≥ 0; maximum Euclidean round-trip error. |
| `ransac_reprojection_tolerance_pixels` | 3 | Finite number > 0. |
| `ransac_maximum_iterations` | 2000 | Positive integer. |
| `ransac_confidence` | 0.99 | Finite number strictly between 0 and 1. |
| `refinement_maximum_iterations` | 10 | Integer ≥ 0; zero disables refinement. |

Integer fields also fit a signed 32-bit native integer. Booleans, nonfinite
numbers, and out-of-range settings raise `ValueError` at construction. Tracking
termination bounds prevent OpenCV silently clamping a requested value and
making the recorded setting misleading.
Window area must fit OpenCV's signed integer scratch-buffer calculation
(`3 * width * height <= 2**31 - 1`). Image dimensions plus twice the corresponding
window dimension must also fit a signed 32-bit integer; this image-dependent
check runs before estimation. These prevent arithmetic overflow, not memory
exhaustion: larger valid settings can still require substantial working memory.

Every field has a matching CLI flag with underscores replaced by hyphens:

```bash
uv run python examples/camera_motion.py recording.mp4 --fps 30 \
  --target-grid-rows 30 --tracking-window-width-pixels 25 \
  --tracking-window-height-pixels 25 --maximum-pyramid-level 4 \
  --forward-backward-tolerance-pixels 0.75 \
  --ransac-reprojection-tolerance-pixels 2
```

The algorithm definition still fixes the similarity model, regular grid,
Euclidean forward/backward check, RANSAC method, and two-correspondence minimum.
Alternative motion models or tracking methods would be separate algorithm
choices. No measurement-acceptance threshold is introduced by these settings.
Shake window and field of view remain separate `CameraShakeSettings` options;
the current temporal filter is a centered moving average.

## Filter shake independently

```python
from pathlib import Path

from hflow.camera_motion import (
    CameraMotionStreamSettings,
    CameraShakeSettings,
    MeasuredCameraShake,
    filter_camera_shake,
    stream_camera_motion,
)

motion_settings = CameraMotionStreamSettings(frames_per_second=30)
shake_settings = CameraShakeSettings(
    half_window_pairs=15,
    horizontal_field_of_view_degrees=90,
)
with stream_camera_motion(Path("recording.mp4"), settings=motion_settings) as observations:
    for filtered in filter_camera_shake(observations, settings=shake_settings):
        if isinstance(filtered.shake, MeasuredCameraShake):
            print(
                filtered.motion.start_seconds, filtered.shake.residual.magnitude_degrees_per_second
            )
        else:
            print(filtered.motion.start_seconds, "unavailable", filtered.shake.reason)
```

The filter converts each measured transform into signed angular rates. Rotation
uses degrees per frame divided by the frame interval. Translation uses the
explicit approximation `horizontal_fov_degrees / frame_width_pixels`, then the
same interval conversion. It subtracts a centered mean from each of these three
rate components; the residual magnitude is their Euclidean norm. Scale is
available in the raw observation but excluded from this shake residual.

With the defaults, the window contains 31 pairs and needs 15 future pairs:
0.5 seconds of video lookahead at 30 fps. It is a moving-average residual, not
an ideal frequency cutoff. Fast intentional motion may contribute to it. The
smoothed component is named `smoothed_motion`, not "intentional motion".

Every input observation has exactly one output. The first/last 15 pairs have
`UnavailableCameraShake(reason="insufficient_context")`. Any complete window
containing an unmeasured pair has `reason="unmeasured_context"`. Missing rates
are never filled with zero, and edges are never padded with invented samples.
Recovery requires a complete measured window around the target pair. A clip
shorter than a complete window has no measured shake values.

Feed the complete ordered stream, including failed pairs: removing failures
before filtering changes time semantics. Missing indices, mixed settings, and
inconsistent timestamps raise. `filter_camera_shake` does not take ownership of
the supplied iterable; keep it inside the file stream's context as shown above.

## Memory and validation scope

Raw estimation retains the previous/current images and optical-flow working
memory. Filtering retains at most `2 * half_window_pairs + 1` observations, plus
bounded temporary rates. The window is limited to 1–4096 pairs on either side.
Memory does not grow with video length unless the caller accumulates outputs.
Exact whole-video percentiles need retained values or external storage; use
running sums or approximate quantiles when bounded summaries are required.

Synthetic tests cover known translation/rotation/scale, a small moving
foreground, weak large-motion fits, featureless gaps and recovery, signed filter
residuals, lookahead, long-stream memory retention, and early decoder closure.
They do not establish accuracy on arbitrary real footage, dominant moving
subjects, perspective changes, rolling shutter, or severe motion blur.

The existing `hflow.checks.camera_stability` aggregate retains its v1 policy.
This stream preserves finite estimates and uses a 31-pair gap-aware filter.
The check keeps its legacy 12-track policy, 30-pair filter at 30 fps, and
zero-filled gaps. The stream's numerical values are not interchangeable with the legacy
aggregate. No default quality gate or scheduled task is enabled by using this API.
