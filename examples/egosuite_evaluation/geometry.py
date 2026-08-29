"""Project EgoSuite world-space hand joints into a labeled camera image."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class CameraPoseInWorld:
    """The camera child frame expressed in its world parent frame."""

    translation: Point3D
    rotation: Quaternion


@dataclass(frozen=True)
class PinholeCameraCalibration:
    width: int
    height: int
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float


@dataclass(frozen=True)
class ProjectedJoint:
    image_x: float
    image_y: float
    depth: float
    is_in_frame: bool


@dataclass(frozen=True)
class ProjectedHand:
    joints: tuple[ProjectedJoint, ...]

    @property
    def in_frame_joint_count(self) -> int:
        return sum(joint.is_in_frame for joint in self.joints)

    @property
    def is_in_frame(self) -> bool:
        return self.in_frame_joint_count > 0


def _world_from_camera_rotation_matrix(
    quaternion: Quaternion,
) -> tuple[tuple[float, float, float], ...]:
    quaternion_norm = math.sqrt(
        quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
        + quaternion.w * quaternion.w
    )
    if not math.isfinite(quaternion_norm) or quaternion_norm <= 0.0:
        raise ValueError("camera quaternion must have a finite, nonzero norm")
    x = quaternion.x / quaternion_norm
    y = quaternion.y / quaternion_norm
    z = quaternion.z / quaternion_norm
    w = quaternion.w / quaternion_norm
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _validate_calibration(calibration: PinholeCameraCalibration) -> None:
    if calibration.width <= 0 or calibration.height <= 0:
        raise ValueError("camera width and height must be positive")
    if calibration.focal_length_x <= 0.0 or calibration.focal_length_y <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    calibration_values = (
        calibration.focal_length_x,
        calibration.focal_length_y,
        calibration.principal_point_x,
        calibration.principal_point_y,
    )
    if not all(math.isfinite(value) for value in calibration_values):
        raise ValueError("camera calibration values must be finite")


def project_world_joints(
    world_joints: Sequence[Point3D],
    camera_pose_in_world: CameraPoseInWorld,
    calibration: PinholeCameraCalibration,
) -> ProjectedHand:
    """Apply world-to-camera inversion and pinhole projection to hand joints.

    EgoSuite stores the camera child-frame pose in the world parent frame, so
    each world point is translated and multiplied by the transpose of the
    camera-to-world rotation before applying the intrinsic matrix.
    """

    if not world_joints:
        raise ValueError("at least one hand joint is required")
    _validate_calibration(calibration)
    world_from_camera = _world_from_camera_rotation_matrix(camera_pose_in_world.rotation)
    projected_joints: list[ProjectedJoint] = []
    for world_joint in world_joints:
        translated_x = world_joint.x - camera_pose_in_world.translation.x
        translated_y = world_joint.y - camera_pose_in_world.translation.y
        translated_z = world_joint.z - camera_pose_in_world.translation.z
        camera_x = (
            world_from_camera[0][0] * translated_x
            + world_from_camera[1][0] * translated_y
            + world_from_camera[2][0] * translated_z
        )
        camera_y = (
            world_from_camera[0][1] * translated_x
            + world_from_camera[1][1] * translated_y
            + world_from_camera[2][1] * translated_z
        )
        camera_depth = (
            world_from_camera[0][2] * translated_x
            + world_from_camera[1][2] * translated_y
            + world_from_camera[2][2] * translated_z
        )
        if not all(math.isfinite(value) for value in (camera_x, camera_y, camera_depth)):
            raise ValueError("hand joint coordinates must be finite")
        if camera_depth <= 0.0:
            projected_joints.append(
                ProjectedJoint(
                    image_x=math.nan,
                    image_y=math.nan,
                    depth=camera_depth,
                    is_in_frame=False,
                )
            )
            continue
        image_x = (
            calibration.focal_length_x * camera_x / camera_depth + calibration.principal_point_x
        )
        image_y = (
            calibration.focal_length_y * camera_y / camera_depth + calibration.principal_point_y
        )
        projected_joints.append(
            ProjectedJoint(
                image_x=image_x,
                image_y=image_y,
                depth=camera_depth,
                is_in_frame=(
                    0.0 <= image_x < calibration.width and 0.0 <= image_y < calibration.height
                ),
            )
        )
    return ProjectedHand(joints=tuple(projected_joints))
