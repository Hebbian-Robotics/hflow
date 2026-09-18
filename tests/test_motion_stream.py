"""Continuous measurement outcomes, missingness, and bounded consumption."""

import gc
import json
import math
import subprocess
import sys
import tracemalloc
from collections.abc import Iterator
from dataclasses import asdict, replace
from itertools import repeat
from pathlib import Path

import numpy as np
import pytest

from hflow.camera_motion import (
    CameraMotionAlgorithmSettings,
    CameraMotionObservation,
    CameraMotionStreamSettings,
    CameraMotionTransform,
    CameraShakeSettings,
    MeasuredCameraMotion,
    MeasuredCameraShake,
    MotionFitEvidence,
    UnavailableCameraShake,
    UnmeasuredCameraMotion,
    filter_camera_shake,
    iter_frame_motion,
    stream_camera_motion,
)

cv2 = pytest.importorskip("cv2", reason="camera motion requires the motion extra")


@pytest.fixture(scope="module", autouse=True)
def single_opencv_thread() -> Iterator[None]:
    previous_thread_count = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(previous_thread_count)


def textured_image(height: int = 240, width: int = 320) -> np.ndarray:
    random_generator = np.random.default_rng(20260912)
    texture = random_generator.integers(0, 256, size=(height, width), dtype=np.uint8)
    return cv2.GaussianBlur(texture, (5, 5), 0)


def motion_observation(
    pair_index: int, translation_pixels: float, *, measured: bool = True
) -> CameraMotionObservation:
    evidence = MotionFitEvidence(100, 100, 100, 0.0)
    measurement = (
        MeasuredCameraMotion(CameraMotionTransform(translation_pixels, 0.0, 0.0, 1.0), evidence)
        if measured
        else UnmeasuredCameraMotion("insufficient_tracks", MotionFitEvidence(100, 0, 0, None))
    )
    return CameraMotionObservation(
        pair_index=pair_index,
        start_seconds=float(pair_index),
        end_seconds=float(pair_index + 1),
        frame_width_pixels=90,
        measurement=measurement,
        settings=CameraMotionStreamSettings(frames_per_second=1.0),
    )


@pytest.mark.parametrize(
    "translation,rotation,scale", [(3.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.015)]
)
def test_estimates_known_translation_rotation_and_scale(
    translation: float, rotation: float, scale: float
) -> None:
    earlier_frame = textured_image()
    transformation_matrix = cv2.getRotationMatrix2D((160, 120), rotation, scale)
    transformation_matrix[0, 2] += translation
    later_frame = cv2.warpAffine(earlier_frame, transformation_matrix, (320, 240))
    observations = list(
        iter_frame_motion(
            [earlier_frame, later_frame], settings=CameraMotionStreamSettings(frames_per_second=30)
        )
    )
    assert len(observations) == 1
    observation = observations[0]
    assert (observation.start_seconds, observation.end_seconds) == (0.0, 1 / 30)
    assert isinstance(observation.measurement, MeasuredCameraMotion)
    measured = observation.measurement
    assert measured.transform.horizontal_translation_pixels == pytest.approx(
        transformation_matrix[0, 2], abs=0.15
    )
    assert measured.transform.vertical_translation_pixels == pytest.approx(
        transformation_matrix[1, 2], abs=0.15
    )
    assert measured.transform.rotation_degrees == pytest.approx(-rotation, abs=0.1)
    assert measured.transform.scale == pytest.approx(scale, abs=0.002)
    assert measured.evidence.inlier_ratio > 0.9
    assert measured.evidence.median_inlier_reprojection_error_pixels is not None
    assert measured.evidence.median_inlier_reprojection_error_pixels < 0.3


def test_reused_input_frame_buffer_does_not_erase_real_movement() -> None:
    texture = textured_image(width=400)
    reusable_buffer = np.empty((240, 320), dtype=np.uint8)

    def frames() -> Iterator[np.ndarray]:
        for frame_index in range(12):
            reusable_buffer[:] = texture[:, frame_index * 3 : frame_index * 3 + 320]
            yield reusable_buffer

    observations = list(iter_frame_motion(frames(), settings=CameraMotionStreamSettings(30)))
    assert len(observations) == 11
    for observation in observations:
        assert isinstance(observation.measurement, MeasuredCameraMotion)
        assert observation.measurement.transform.horizontal_translation_pixels == pytest.approx(
            -3, abs=0.1
        )


def test_featureless_gap_is_unmeasured_and_tracking_recovers() -> None:
    texture = textured_image()
    flat_frame = np.full_like(texture, 128)
    frames = [texture, texture, flat_frame, flat_frame, texture, texture]
    observations = list(iter_frame_motion(frames, settings=CameraMotionStreamSettings(30)))
    assert [observation.measurement.status for observation in observations] == [
        "measured",
        "unmeasured",
        "unmeasured",
        "unmeasured",
        "measured",
    ]
    for observation in observations[1:4]:
        assert isinstance(observation.measurement, UnmeasuredCameraMotion)
        assert observation.measurement.reason == "insufficient_tracks"
        assert observation.measurement.evidence.retained_track_count < 2


def test_large_shake_preserves_weak_estimates_and_their_evidence() -> None:
    texture = textured_image(height=512, width=1024)
    frames = []
    for frame_index in range(60):
        phase = 2 * math.pi * 8 * frame_index / 30
        horizontal_offset = 64 + round(24 * math.sin(phase))
        vertical_offset = 64 + round(24 * math.cos(phase))
        frames.append(
            texture[
                vertical_offset : vertical_offset + 240, horizontal_offset : horizontal_offset + 320
            ]
        )
    observations = list(iter_frame_motion(frames, settings=CameraMotionStreamSettings(30)))
    assert all(
        isinstance(observation.measurement, MeasuredCameraMotion) for observation in observations
    )
    weak_estimates = [
        observation
        for observation in observations
        if observation.measurement.evidence.track_retention_ratio < 0.25
        or observation.measurement.evidence.inlier_ratio < 0.5
    ]
    assert len(weak_estimates) > len(observations) // 2
    assert len(observations) == len(frames) - 1


def test_sparse_texture_preserves_a_solvable_transform_with_its_small_track_count() -> None:
    frame = textured_image(height=8, width=16)
    observation = next(iter_frame_motion([frame, frame], settings=CameraMotionStreamSettings(30)))
    assert isinstance(observation.measurement, MeasuredCameraMotion)
    assert observation.measurement.evidence.retained_track_count == 2
    assert observation.measurement.transform.horizontal_translation_pixels == pytest.approx(0)
    assert observation.measurement.transform.vertical_translation_pixels == pytest.approx(0)
    assert observation.measurement.transform.rotation_degrees == pytest.approx(0)
    assert observation.measurement.transform.scale == pytest.approx(1)


def test_custom_estimator_settings_change_sampling_and_preserve_motion_and_provenance() -> None:
    algorithm = CameraMotionAlgorithmSettings(
        target_grid_rows=10,
        minimum_track_spacing_pixels=10,
        tracking_window_width_pixels=25,
        tracking_window_height_pixels=19,
        maximum_pyramid_level=2,
        tracking_maximum_iterations=40,
        tracking_convergence_epsilon_pixels=0.005,
        tracking_minimum_eigenvalue=0.0002,
        forward_backward_tolerance_pixels=0.5,
        ransac_reprojection_tolerance_pixels=1.5,
        ransac_maximum_iterations=1000,
        ransac_confidence=0.95,
        refinement_maximum_iterations=5,
    )
    earlier_frame = textured_image()
    later_frame = cv2.warpAffine(
        earlier_frame, np.array([[1, 0, 3], [0, 1, 0]], dtype=np.float32), (320, 240)
    )
    frames = [earlier_frame, later_frame]
    default_observation = next(iter_frame_motion(frames, settings=CameraMotionStreamSettings(30)))
    custom_observation = next(
        iter_frame_motion(frames, settings=CameraMotionStreamSettings(30, algorithm=algorithm))
    )
    assert isinstance(custom_observation.measurement, MeasuredCameraMotion)
    assert custom_observation.measurement.transform.horizontal_translation_pixels == pytest.approx(
        3, abs=0.15
    )
    assert (
        custom_observation.measurement.evidence.attempted_track_count
        < default_observation.measurement.evidence.attempted_track_count
    )
    payload = json.loads(json.dumps(asdict(custom_observation), allow_nan=False))
    assert payload["settings"]["algorithm"] == asdict(algorithm)


def test_tracking_eigenvalue_setting_controls_which_motion_can_be_estimated() -> None:
    frame = textured_image()
    permissive = next(iter_frame_motion([frame, frame], settings=CameraMotionStreamSettings(30)))
    restrictive = next(
        iter_frame_motion(
            [frame, frame],
            settings=CameraMotionStreamSettings(
                30, algorithm=CameraMotionAlgorithmSettings(tracking_minimum_eigenvalue=1e6)
            ),
        )
    )
    assert isinstance(permissive.measurement, MeasuredCameraMotion)
    assert isinstance(restrictive.measurement, UnmeasuredCameraMotion)
    assert restrictive.measurement.evidence.retained_track_count == 0


def test_moving_foreground_does_not_become_camera_motion() -> None:
    background = textured_image()
    foreground = textured_image(100, 100)
    frames = []
    for frame_index in range(20):
        frame = background.copy()
        left = 10 + 4 * frame_index
        frame[70:170, left : left + 100] = foreground
        frames.append(frame)
    for observation in iter_frame_motion(frames, settings=CameraMotionStreamSettings(30)):
        assert isinstance(observation.measurement, MeasuredCameraMotion)
        transform = observation.measurement.transform
        assert (
            math.hypot(
                transform.horizontal_translation_pixels, transform.vertical_translation_pixels
            )
            < 1.0  # Subpixel camera error despite the foreground moving four pixels.
        )


def test_filter_returns_continuous_signed_residuals_and_marks_edges() -> None:
    observations = [motion_observation(index, value) for index, value in enumerate((1, 1, 4, 1, 1))]
    results = list(
        filter_camera_shake(observations, settings=CameraShakeSettings(half_window_pairs=1))
    )
    assert len(results) == 5
    assert results[0].shake == results[-1].shake == UnavailableCameraShake("insufficient_context")
    for result, expected_residual in zip(results[1:4], (-1, 2, -1), strict=True):
        assert isinstance(result.shake, MeasuredCameraShake)
        assert result.shake.residual.horizontal_degrees_per_second == expected_residual
        assert result.shake.residual.magnitude_degrees_per_second == abs(expected_residual)
        assert result.shake.smoothed_motion.horizontal_degrees_per_second == 2


def test_filter_never_fills_a_gap_with_zero_and_recovers_after_its_window() -> None:
    observations = [motion_observation(index, 3, measured=index != 7) for index in range(15)]
    results = list(
        filter_camera_shake(observations, settings=CameraShakeSettings(half_window_pairs=2))
    )
    assert [result.motion.pair_index for result in results] == list(range(15))
    for index, result in enumerate(results):
        if index < 2 or index > 12:
            assert result.shake == UnavailableCameraShake("insufficient_context")
        elif 5 <= index <= 9:
            assert result.shake == UnavailableCameraShake("unmeasured_context")
        else:
            assert isinstance(result.shake, MeasuredCameraShake)
            assert result.shake.residual.magnitude_degrees_per_second == 0
            assert result.shake.smoothed_motion.horizontal_degrees_per_second == 3


@pytest.mark.parametrize("length", [0, 1, 2, 3, 10])
def test_short_stream_preserves_every_pair_without_inventing_filter_context(length: int) -> None:
    observations = (motion_observation(index, 1) for index in range(length))
    results = list(
        filter_camera_shake(observations, settings=CameraShakeSettings(half_window_pairs=15))
    )
    assert len(results) == length
    assert all(result.shake == UnavailableCameraShake("insufficient_context") for result in results)


def test_filter_consumes_only_bounded_lookahead() -> None:
    consumed = 0

    def infinite_motion() -> Iterator[CameraMotionObservation]:
        nonlocal consumed
        while True:
            observation = motion_observation(consumed, 2)
            consumed += 1
            yield observation

    filtered = filter_camera_shake(
        infinite_motion(), settings=CameraShakeSettings(half_window_pairs=2)
    )
    for pair_index in range(100):
        result = next(filtered)
        assert result.motion.pair_index == pair_index
        assert consumed == pair_index + 3


def test_discarding_results_keeps_filter_memory_bounded_over_a_long_stream() -> None:
    def observations() -> Iterator[CameraMotionObservation]:
        for pair_index in range(6000):
            yield motion_observation(pair_index, 1)

    filtered = filter_camera_shake(
        observations(), settings=CameraShakeSettings(half_window_pairs=2)
    )
    tracemalloc.start()
    try:
        for _ in range(500):
            next(filtered)
        gc.collect()
        initial_retained_bytes = tracemalloc.get_traced_memory()[0]
        for _ in range(5000):
            next(filtered)
        gc.collect()
        later_retained_bytes = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert later_retained_bytes - initial_retained_bytes < 64 * 1024


def test_discarding_raw_motion_keeps_frame_and_observation_memory_bounded() -> None:
    observations = iter_frame_motion(
        repeat(textured_image(120, 160)), settings=CameraMotionStreamSettings(30)
    )
    tracemalloc.start()
    try:
        for _ in range(50):
            next(observations)
        gc.collect()
        initial_retained_bytes = tracemalloc.get_traced_memory()[0]
        for _ in range(400):
            next(observations)
        gc.collect()
        later_retained_bytes = tracemalloc.get_traced_memory()[0]
    finally:
        observations.close()
        tracemalloc.stop()
    assert later_retained_bytes - initial_retained_bytes < 64 * 1024


def test_skipping_observations_is_an_error_instead_of_smoothing_across_unknown_time() -> None:
    observations = [motion_observation(0, 1), motion_observation(2, 1)]
    with pytest.raises(ValueError, match="consecutive"):
        list(filter_camera_shake(observations, settings=CameraShakeSettings(half_window_pairs=1)))


def test_filter_refuses_to_combine_different_estimator_definitions() -> None:
    first = motion_observation(0, 1)
    second = replace(
        motion_observation(1, 1),
        settings=CameraMotionStreamSettings(
            1, algorithm=CameraMotionAlgorithmSettings(ransac_reprojection_tolerance_pixels=1)
        ),
    )
    with pytest.raises(ValueError, match="consistent settings"):
        list(
            filter_camera_shake([first, second], settings=CameraShakeSettings(half_window_pairs=1))
        )


def test_observation_refuses_timestamps_that_disagree_with_declared_cadence() -> None:
    with pytest.raises(ValueError, match="cadence"):
        replace(motion_observation(0, 1), end_seconds=0.5)


def test_overflowing_rate_cannot_be_reported_as_a_measured_nan() -> None:
    settings = CameraMotionStreamSettings(frames_per_second=1e308)
    observations = [
        replace(
            motion_observation(pair_index, 3),
            start_seconds=pair_index / settings.frames_per_second,
            end_seconds=(pair_index + 1) / settings.frames_per_second,
            settings=settings,
        )
        for pair_index in range(3)
    ]
    with pytest.raises(ValueError, match="finite"):
        list(filter_camera_shake(observations, settings=CameraShakeSettings(half_window_pairs=1)))


def test_large_finite_rates_do_not_overflow_the_centered_average() -> None:
    observations = [motion_observation(pair_index, 1e308) for pair_index in range(3)]
    results = list(
        filter_camera_shake(observations, settings=CameraShakeSettings(half_window_pairs=1))
    )
    assert isinstance(results[1].shake, MeasuredCameraShake)
    assert results[1].shake.residual.magnitude_degrees_per_second == 0
    assert results[1].shake.smoothed_motion.horizontal_degrees_per_second == 1e308


@pytest.mark.parametrize("invalid_frame_rate", [0, -1, math.nan, math.inf, True, 5e-324])
def test_frame_cadence_requires_a_finite_positive_interval(invalid_frame_rate: float) -> None:
    with pytest.raises(ValueError, match="frames_per_second"):
        CameraMotionStreamSettings(invalid_frame_rate)


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("target_grid_rows", 0),
        ("minimum_track_spacing_pixels", True),
        ("tracking_window_width_pixels", 2),
        ("tracking_window_height_pixels", 2**31),
        ("tracking_window_width_pixels", 2**31 - 1),
        ("maximum_pyramid_level", -1),
        ("maximum_pyramid_level", 31),
        ("tracking_maximum_iterations", 101),
        ("tracking_convergence_epsilon_pixels", 10.1),
        ("tracking_minimum_eigenvalue", -1),
        ("forward_backward_tolerance_pixels", math.inf),
        ("ransac_reprojection_tolerance_pixels", 0),
        ("ransac_maximum_iterations", 0),
        ("ransac_confidence", math.nan),
        ("ransac_confidence", 1),
        ("refinement_maximum_iterations", -1),
    ],
)
def test_estimator_settings_reject_invalid_or_silently_clamped_values(
    field_name: str, invalid_value: int | float
) -> None:
    with pytest.raises(ValueError, match=field_name):
        replace(CameraMotionAlgorithmSettings(), **{field_name: invalid_value})


@pytest.mark.parametrize("invalid_window", [0, -1, 4097, True])
def test_filter_requires_a_bounded_positive_integer_window(invalid_window: int) -> None:
    with pytest.raises(ValueError, match="half_window_pairs"):
        CameraShakeSettings(half_window_pairs=invalid_window)


def test_frame_shape_change_is_an_input_error() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        list(
            iter_frame_motion(
                [textured_image(), textured_image(120, 160)],
                settings=CameraMotionStreamSettings(30),
            )
        )


@pytest.mark.parametrize("frame_count", [0, 1])
def test_no_adjacent_frames_yields_no_observations(frame_count: int) -> None:
    frames = [textured_image()] * frame_count
    assert list(iter_frame_motion(frames, settings=CameraMotionStreamSettings(30))) == []


def test_file_stream_decodes_all_pairs_and_closes_on_early_exit(tmp_path: Path) -> None:
    video_path = tmp_path / "static.mkv"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter.fourcc(*"FFV1"), 30, (320, 240), False
    )
    assert writer.isOpened()
    try:
        for _ in range(12):
            writer.write(textured_image())
    finally:
        writer.release()
    settings = CameraMotionStreamSettings(30)
    with stream_camera_motion(video_path, settings=settings) as observations:
        first = next(observations)
        assert first.pair_index == 0
    assert next(observations, None) is None
    with stream_camera_motion(video_path, settings=settings) as observations:
        completed = list(observations)
    assert len(completed) == 11
    assert all(
        isinstance(observation.measurement, MeasuredCameraMotion) for observation in completed
    )
    assert completed[-1].end_seconds == 11 / 30

    # Exercise the public CLI boundary against real video, including recorded
    # settings that explain how each measurement was produced.
    command_result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "examples" / "camera_motion.py"),
            str(video_path),
            "--fps",
            "30",
            "--target-grid-rows",
            "10",
            "--ransac-confidence",
            "0.95",
            "--refinement-maximum-iterations",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [json.loads(line) for line in command_result.stdout.splitlines()]
    assert len(records) == 11
    expected_algorithm = CameraMotionAlgorithmSettings(
        target_grid_rows=10, ransac_confidence=0.95, refinement_maximum_iterations=0
    )
    for record in records:
        assert record["settings"]["algorithm"] == asdict(expected_algorithm)
        assert record["measurement"]["status"] == "measured"
        assert record["measurement"]["transform"]["horizontal_translation_pixels"] == pytest.approx(
            0
        )
