"""Print continuous motion or filtered shake as JSON Lines for a fixed-rate video.

Requires hflow[motion] and FFmpeg. Reads the source without modifying it and may
download HFlow's managed FFmpeg. No model, service, or credentials are used.
Run from the repository root:
    uv run python examples/camera_motion.py recording.mp4 --fps 30 --shake
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from hflow.camera_motion import (
    CameraMotionAlgorithmSettings,
    CameraMotionObservation,
    CameraMotionStreamSettings,
    CameraShakeObservation,
    CameraShakeSettings,
    MeasuredCameraShake,
    filter_camera_shake,
    stream_camera_motion,
)


def observation_json(observation: CameraMotionObservation | CameraShakeObservation) -> str:
    payload = asdict(observation)
    if isinstance(observation, CameraShakeObservation) and isinstance(
        observation.shake, MeasuredCameraShake
    ):
        payload["shake"]["magnitude_degrees_per_second"] = (
            observation.shake.residual.magnitude_degrees_per_second
        )
    return json.dumps(payload, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--fps", type=float, required=True, help="Actual constant source frame rate"
    )
    parser.add_argument("--shake", action="store_true", help="Apply centered shake filtering")
    parser.add_argument("--half-window-pairs", type=int, default=15)
    parser.add_argument("--horizontal-fov-degrees", type=float, default=90)
    algorithm_defaults = CameraMotionAlgorithmSettings()
    algorithm_arguments = parser.add_argument_group("motion estimator parameters")
    algorithm_arguments.add_argument(
        "--target-grid-rows", type=int, default=algorithm_defaults.target_grid_rows
    )
    algorithm_arguments.add_argument(
        "--minimum-track-spacing-pixels",
        type=int,
        default=algorithm_defaults.minimum_track_spacing_pixels,
    )
    algorithm_arguments.add_argument(
        "--tracking-window-width-pixels",
        type=int,
        default=algorithm_defaults.tracking_window_width_pixels,
    )
    algorithm_arguments.add_argument(
        "--tracking-window-height-pixels",
        type=int,
        default=algorithm_defaults.tracking_window_height_pixels,
    )
    algorithm_arguments.add_argument(
        "--maximum-pyramid-level", type=int, default=algorithm_defaults.maximum_pyramid_level
    )
    algorithm_arguments.add_argument(
        "--tracking-maximum-iterations",
        type=int,
        default=algorithm_defaults.tracking_maximum_iterations,
    )
    algorithm_arguments.add_argument(
        "--tracking-convergence-epsilon-pixels",
        type=float,
        default=algorithm_defaults.tracking_convergence_epsilon_pixels,
    )
    algorithm_arguments.add_argument(
        "--tracking-minimum-eigenvalue",
        type=float,
        default=algorithm_defaults.tracking_minimum_eigenvalue,
    )
    algorithm_arguments.add_argument(
        "--forward-backward-tolerance-pixels",
        type=float,
        default=algorithm_defaults.forward_backward_tolerance_pixels,
    )
    algorithm_arguments.add_argument(
        "--ransac-reprojection-tolerance-pixels",
        type=float,
        default=algorithm_defaults.ransac_reprojection_tolerance_pixels,
    )
    algorithm_arguments.add_argument(
        "--ransac-maximum-iterations",
        type=int,
        default=algorithm_defaults.ransac_maximum_iterations,
    )
    algorithm_arguments.add_argument(
        "--ransac-confidence", type=float, default=algorithm_defaults.ransac_confidence
    )
    algorithm_arguments.add_argument(
        "--refinement-maximum-iterations",
        type=int,
        default=algorithm_defaults.refinement_maximum_iterations,
    )
    arguments = parser.parse_args()
    try:
        motion_settings = CameraMotionStreamSettings(
            frames_per_second=arguments.fps,
            algorithm=CameraMotionAlgorithmSettings(
                target_grid_rows=arguments.target_grid_rows,
                minimum_track_spacing_pixels=arguments.minimum_track_spacing_pixels,
                tracking_window_width_pixels=arguments.tracking_window_width_pixels,
                tracking_window_height_pixels=arguments.tracking_window_height_pixels,
                maximum_pyramid_level=arguments.maximum_pyramid_level,
                tracking_maximum_iterations=arguments.tracking_maximum_iterations,
                tracking_convergence_epsilon_pixels=arguments.tracking_convergence_epsilon_pixels,
                tracking_minimum_eigenvalue=arguments.tracking_minimum_eigenvalue,
                forward_backward_tolerance_pixels=arguments.forward_backward_tolerance_pixels,
                ransac_reprojection_tolerance_pixels=arguments.ransac_reprojection_tolerance_pixels,
                ransac_maximum_iterations=arguments.ransac_maximum_iterations,
                ransac_confidence=arguments.ransac_confidence,
                refinement_maximum_iterations=arguments.refinement_maximum_iterations,
            ),
        )
        shake_settings = CameraShakeSettings(
            half_window_pairs=arguments.half_window_pairs,
            horizontal_field_of_view_degrees=arguments.horizontal_fov_degrees,
        )
    except ValueError as error:
        parser.error(str(error))
    with stream_camera_motion(arguments.video, settings=motion_settings) as observations:
        if arguments.shake:
            for filtered in filter_camera_shake(observations, settings=shake_settings):
                print(observation_json(filtered), flush=True)
        else:
            for observation in observations:
                print(observation_json(observation), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # The stream context has reaped FFmpeg. Redirect the descriptor so
        # Python's final stdout flush cannot raise again on the closed pipe.
        with Path(os.devnull).open("w") as discarded_output:
            os.dup2(discarded_output.fileno(), sys.stdout.fileno())
