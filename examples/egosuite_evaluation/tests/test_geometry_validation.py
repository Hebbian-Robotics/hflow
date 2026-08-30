from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.egosuite_evaluation.geometry import (
    CameraPoseInWorld,
    PinholeCameraCalibration,
    Point3D,
    Quaternion,
    project_world_joints,
)

IDENTITY_CAMERA_POSE = CameraPoseInWorld(
    translation=Point3D(0.0, 0.0, 0.0),
    rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
)
VALID_CALIBRATION = PinholeCameraCalibration(
    width=200,
    height=100,
    focal_length_x=100.0,
    focal_length_y=100.0,
    principal_point_x=100.0,
    principal_point_y=50.0,
)


def test_projection_rejects_zero_camera_quaternion() -> None:
    camera_pose = CameraPoseInWorld(
        translation=Point3D(0.0, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="camera quaternion must have a finite, nonzero norm"):
        project_world_joints([Point3D(0.0, 0.0, 1.0)], camera_pose, VALID_CALIBRATION)


def test_projection_rejects_nonfinite_camera_quaternion() -> None:
    camera_pose = CameraPoseInWorld(
        translation=Point3D(0.0, 0.0, 0.0),
        rotation=Quaternion(math.nan, 0.0, 0.0, 1.0),
    )

    with pytest.raises(ValueError, match="camera quaternion must have a finite, nonzero norm"):
        project_world_joints([Point3D(0.0, 0.0, 1.0)], camera_pose, VALID_CALIBRATION)


def test_projection_rejects_nonpositive_camera_dimensions() -> None:
    calibration = PinholeCameraCalibration(
        width=0,
        height=100,
        focal_length_x=100.0,
        focal_length_y=100.0,
        principal_point_x=100.0,
        principal_point_y=50.0,
    )

    with pytest.raises(ValueError, match="camera width and height must be positive"):
        project_world_joints([Point3D(0.0, 0.0, 1.0)], IDENTITY_CAMERA_POSE, calibration)


def test_projection_rejects_nonpositive_focal_lengths() -> None:
    calibration = PinholeCameraCalibration(
        width=200,
        height=100,
        focal_length_x=0.0,
        focal_length_y=100.0,
        principal_point_x=100.0,
        principal_point_y=50.0,
    )

    with pytest.raises(ValueError, match="camera focal lengths must be positive"):
        project_world_joints([Point3D(0.0, 0.0, 1.0)], IDENTITY_CAMERA_POSE, calibration)


def test_projection_rejects_nonfinite_calibration_values() -> None:
    calibration = PinholeCameraCalibration(
        width=200,
        height=100,
        focal_length_x=100.0,
        focal_length_y=100.0,
        principal_point_x=math.nan,
        principal_point_y=50.0,
    )

    with pytest.raises(ValueError, match="camera calibration values must be finite"):
        project_world_joints([Point3D(0.0, 0.0, 1.0)], IDENTITY_CAMERA_POSE, calibration)


def test_projection_rejects_empty_joint_sequence() -> None:
    with pytest.raises(ValueError, match="at least one hand joint is required"):
        project_world_joints([], IDENTITY_CAMERA_POSE, VALID_CALIBRATION)


def test_projection_rejects_nonfinite_joint_coordinates() -> None:
    with pytest.raises(ValueError, match="hand joint coordinates must be finite"):
        project_world_joints(
            [Point3D(math.inf, 0.0, 1.0)], IDENTITY_CAMERA_POSE, VALID_CALIBRATION
        )
