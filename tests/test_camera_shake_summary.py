import math

import pytest

from hflow.camera_motion import (
    AngularMotionRate,
    CameraMotionObservation,
    CameraMotionStreamSettings,
    CameraMotionTransform,
    CameraShakeObservation,
    CameraShakeRateBin,
    CameraShakeSettings,
    MeasuredCameraMotion,
    MeasuredCameraShake,
    MotionFitEvidence,
    UnavailableCameraShake,
    UnmeasuredCameraMotion,
    camera_shake_rate_percentile,
    summarize_camera_shake,
)


def shake_observation(
    pair_index: int,
    *,
    duration_seconds: float,
    shake: MeasuredCameraShake | UnavailableCameraShake,
    motion_measured: bool = True,
) -> CameraShakeObservation:
    motion_settings = CameraMotionStreamSettings(frames_per_second=1 / duration_seconds)
    motion = (
        MeasuredCameraMotion(
            CameraMotionTransform(0.0, 0.0, 0.0, 1.0),
            MotionFitEvidence(20, 20, 20, 0.0),
        )
        if motion_measured
        else UnmeasuredCameraMotion("insufficient_tracks", MotionFitEvidence(20, 0, 0, None))
    )
    return CameraShakeObservation(
        motion=CameraMotionObservation(
            pair_index=pair_index,
            start_seconds=pair_index / motion_settings.frames_per_second,
            end_seconds=(pair_index + 1) / motion_settings.frames_per_second,
            frame_width_pixels=320,
            measurement=motion,
            settings=motion_settings,
        ),
        shake=shake,
        settings=CameraShakeSettings(),
    )


def measured_shake(rate: float) -> MeasuredCameraShake:
    return MeasuredCameraShake(
        residual=AngularMotionRate(0.0, rate, 0.0),
        smoothed_motion=AngularMotionRate(0.0, 0.0, 0.0),
    )


def test_summary_weights_shake_and_coverage_by_observed_duration() -> None:
    # The reducer can consume multiple clips with different fixed frame cadences.
    observations = (
        shake_observation(0, duration_seconds=0.25, shake=measured_shake(2.0)),
        shake_observation(
            1, duration_seconds=0.25, shake=UnavailableCameraShake("insufficient_context")
        ),
        shake_observation(0, duration_seconds=0.5, shake=measured_shake(4.0)),
        shake_observation(
            1,
            duration_seconds=0.5,
            shake=UnavailableCameraShake("unmeasured_context"),
            motion_measured=False,
        ),
    )

    summary = summarize_camera_shake(iter(observations))

    assert summary.pair_count == 4
    assert summary.measured_motion_pair_count == 3
    assert summary.measured_shake_pair_count == 2
    assert summary.insufficient_context_pair_count == 1
    assert summary.unmeasured_context_pair_count == 1
    assert summary.observed_seconds == pytest.approx(1.5)
    assert summary.assessed_seconds == pytest.approx(0.75)
    assert summary.unassessed_seconds == pytest.approx(0.75)
    assert summary.assessed_fraction == pytest.approx(0.5)
    assert summary.mean_shake_degrees_per_second == pytest.approx(10 / 3)
    assert summary.rms_shake_degrees_per_second == pytest.approx(math.sqrt(12))
    assert summary.maximum_shake_degrees_per_second == pytest.approx(4.0)
    assert [
        (rate_bin.rate_floor_degrees_per_second, rate_bin.assessed_seconds)
        for rate_bin in summary.rate_bins
    ] == [
        (2, 0.25),
        (4, 0.5),
    ]
    assert summary.p99_shake_degrees_per_second == 4.0


def test_merged_p99_ignores_one_extreme_pair_without_losing_its_duration() -> None:
    normal_observations = tuple(
        shake_observation(index, duration_seconds=0.0625, shake=measured_shake(11.25))
        for index in range(99)
    )
    extreme_observation = shake_observation(
        99, duration_seconds=0.0625, shake=measured_shake(2261.0)
    )
    first_window = summarize_camera_shake(iter(normal_observations[:49]))
    second_window = summarize_camera_shake(iter((*normal_observations[49:], extreme_observation)))

    assert second_window.maximum_shake_degrees_per_second == 2261.0
    assert (
        camera_shake_rate_percentile(
            (*first_window.rate_bins, *second_window.rate_bins),
            percentile=99,
            maximum_shake_degrees_per_second=2261.0,
        )
        == 12.0
    )
    assert (
        camera_shake_rate_percentile(
            (CameraShakeRateBin(11, 0.98), CameraShakeRateBin(2261, 0.02)),
            percentile=99,
            maximum_shake_degrees_per_second=2261.0,
        )
        == 2261.0
    )
    assert (
        camera_shake_rate_percentile(
            (CameraShakeRateBin(4992, 0.99), CameraShakeRateBin(6016, 0.01)),
            percentile=99,
            maximum_shake_degrees_per_second=6100.0,
        )
        == 5056.0
    )
    large_rate_summary = summarize_camera_shake(
        iter((shake_observation(0, duration_seconds=0.5, shake=measured_shake(5000.0)),))
    )
    assert large_rate_summary.rate_bins == (CameraShakeRateBin(4992, 0.5),)


def test_p99_stays_in_lower_bin_at_a_30_fps_duration_boundary() -> None:
    observations = (
        shake_observation(
            pair_index,
            duration_seconds=1 / 30,
            shake=measured_shake(
                11.25
                if pair_index < 1
                else 12.25
                if pair_index < 23
                else 13.25
                if pair_index < 297
                else 2261.0
            ),
        )
        for pair_index in range(300)
    )

    summary = summarize_camera_shake(observations)

    assert summary.maximum_shake_degrees_per_second == 2261.0
    assert summary.p99_shake_degrees_per_second == 14.0
    assert (
        camera_shake_rate_percentile(
            (
                CameraShakeRateBin(11, 1 / 30),
                CameraShakeRateBin(12, 22 / 30),
                CameraShakeRateBin(13, 274 / 30),
                CameraShakeRateBin(2261, 3 / 30),
            ),
            percentile=99,
            maximum_shake_degrees_per_second=2261.0,
        )
        == 14.0
    )
    assert (
        camera_shake_rate_percentile(
            (
                CameraShakeRateBin(11, 0.9899999999999995),
                CameraShakeRateBin(2261, 0.0100000000000005),
            ),
            percentile=99,
            maximum_shake_degrees_per_second=2261.0,
        )
        == 2261.0
    )


@pytest.mark.parametrize("empty", [True, False])
def test_no_assessed_motion_is_missing_instead_of_zero(empty: bool) -> None:
    observations = (
        ()
        if empty
        else (
            shake_observation(
                0, duration_seconds=0.5, shake=UnavailableCameraShake("insufficient_context")
            ),
            shake_observation(
                1,
                duration_seconds=0.5,
                shake=UnavailableCameraShake("unmeasured_context"),
                motion_measured=False,
            ),
        )
    )

    summary = summarize_camera_shake(iter(observations))

    assert summary.pair_count == (0 if empty else 2)
    assert summary.observed_seconds == (0 if empty else 1)
    assert summary.assessed_seconds == 0
    assert summary.unassessed_seconds == summary.observed_seconds
    assert summary.assessed_fraction == (None if empty else 0)
    assert summary.mean_shake_degrees_per_second is None
    assert summary.rms_shake_degrees_per_second is None
    assert summary.maximum_shake_degrees_per_second is None
    assert summary.rate_bins == ()
    assert summary.p99_shake_degrees_per_second is None
