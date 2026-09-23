import math

import pytest

from hflow.camera_motion import (
    AngularMotionRate,
    CameraMotionObservation,
    CameraMotionStreamSettings,
    CameraMotionTransform,
    CameraShakeObservation,
    CameraShakeSettings,
    MeasuredCameraMotion,
    MeasuredCameraShake,
    MotionFitEvidence,
    UnavailableCameraShake,
    UnmeasuredCameraMotion,
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
