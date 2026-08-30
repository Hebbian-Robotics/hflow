from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.egosuite_evaluation.evaluate import (
    CameraView,
    ProjectedHandFrameLabel,
    _evaluation_result_summary,
    _selected_frame_indices,
    parse_hand_count_response,
    select_episode_paths,
    select_stratified_labels,
)
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
CALIBRATION = PinholeCameraCalibration(
    width=200,
    height=100,
    focal_length_x=100.0,
    focal_length_y=100.0,
    principal_point_x=100.0,
    principal_point_y=50.0,
)


def test_projection_counts_a_hand_when_any_labeled_joint_is_in_frame() -> None:
    projected_hand = project_world_joints(
        [
            Point3D(0.0, 0.0, 1.0),
            Point3D(5.0, 0.0, 1.0),
            Point3D(0.0, 0.0, -1.0),
        ],
        IDENTITY_CAMERA_POSE,
        CALIBRATION,
    )

    assert projected_hand.is_in_frame
    assert projected_hand.in_frame_joint_count == 1
    assert projected_hand.joints[0].image_x == pytest.approx(100.0)
    assert projected_hand.joints[0].image_y == pytest.approx(50.0)
    assert not projected_hand.joints[1].is_in_frame
    assert not projected_hand.joints[2].is_in_frame


def test_projection_inverts_the_camera_pose_from_world_coordinates() -> None:
    half_turn_component = math.sqrt(0.5)
    camera_facing_world_positive_x = CameraPoseInWorld(
        translation=Point3D(10.0, 5.0, 2.0),
        rotation=Quaternion(0.0, half_turn_component, 0.0, half_turn_component),
    )

    projected_hand = project_world_joints(
        [Point3D(12.0, 5.0, 2.0)],
        camera_facing_world_positive_x,
        CALIBRATION,
    )

    assert projected_hand.is_in_frame
    assert projected_hand.joints[0].depth == pytest.approx(2.0)
    assert projected_hand.joints[0].image_x == pytest.approx(100.0)
    assert projected_hand.joints[0].image_y == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("response_text", "expected_hand_count"),
    [
        ("0", 0),
        ('{"hand_count": 1}', 1),
        ('```json\n{"hand_count": 2}\n```', 2),
    ],
)
def test_hand_count_parser_accepts_supported_endpoint_responses(
    response_text: str, expected_hand_count: int
) -> None:
    assert parse_hand_count_response(response_text) == expected_hand_count


def test_evaluation_summary_reports_accuracy_and_confusion_without_counting_failures() -> None:
    results: list[dict[str, object]] = [
        {"status": "ok", "expected_value": 2, "predicted_value": 2},
        {"status": "ok", "expected_value": 1, "predicted_value": 2},
        {"status": "invalid", "expected_value": 0},
    ]

    summary = _evaluation_result_summary(results)

    assert summary["attempted_count"] == 3
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 1
    assert summary["agreement_fraction"] == 0.5
    assert summary["attempted_agreement_fraction"] == pytest.approx(1 / 3)
    assert summary["macro_agreement_fraction"] == 0.5
    assert summary["macro_attempted_agreement_fraction"] == pytest.approx(1 / 3)
    assert summary["per_class_agreement"] == {
        "0": {
            "attempted_count": 1,
            "valid_count": 0,
            "agreement_count": 0,
            "agreement_fraction": None,
            "attempted_agreement_fraction": 0.0,
        },
        "1": {
            "attempted_count": 1,
            "valid_count": 1,
            "agreement_count": 0,
            "agreement_fraction": 0.0,
            "attempted_agreement_fraction": 0.0,
        },
        "2": {
            "attempted_count": 1,
            "valid_count": 1,
            "agreement_count": 1,
            "agreement_fraction": 1.0,
            "attempted_agreement_fraction": 1.0,
        },
    }
    assert summary["confusion_matrix"] == {
        "0": {"0": 0, "1": 0, "2": 0},
        "1": {"0": 0, "1": 0, "2": 1},
        "2": {"0": 0, "1": 0, "2": 1},
    }


def test_stratified_selection_caps_each_class_and_is_reproducible() -> None:
    source_path = Path("episode.mcap")
    labels = [
        ProjectedHandFrameLabel(
            source_path=source_path,
            source_episode="episode",
            camera_view=CameraView.HEAD_LEFT,
            frame_index=frame_index,
            left_in_frame_joint_count=21 if expected_hand_count >= 1 else 0,
            right_in_frame_joint_count=21 if expected_hand_count == 2 else 0,
            expected_hand_count=expected_hand_count,
            left_hand_issue_reasons=(),
            right_hand_issue_reasons=(),
        )
        for frame_index, expected_hand_count in enumerate([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ]

    first_selection = select_stratified_labels(
        {source_path: labels}, samples_per_hand_count=2, sample_seed=7
    )
    repeated_selection = select_stratified_labels(
        {source_path: labels}, samples_per_hand_count=2, sample_seed=7
    )

    selected_labels = first_selection[source_path]
    assert [label.frame_index for label in selected_labels] == sorted(
        label.frame_index for label in selected_labels
    )
    assert [label.expected_hand_count for label in selected_labels].count(0) == 2
    assert [label.expected_hand_count for label in selected_labels].count(1) == 2
    assert [label.expected_hand_count for label in selected_labels].count(2) == 2
    assert first_selection == repeated_selection


def test_natural_sampling_selects_reproducible_episodes_and_frames() -> None:
    source_paths = tuple(Path(f"episode-{episode_index}.mcap") for episode_index in range(5))

    first_episode_selection = select_episode_paths(
        source_paths,
        episode_count=3,
        sample_seed=7,
    )
    repeated_episode_selection = select_episode_paths(
        source_paths,
        episode_count=3,
        sample_seed=7,
    )
    selected_frame_indices = _selected_frame_indices(
        first_episode_selection[0],
        300,
        frame_stride=30,
        limit_per_episode=None,
        samples_per_episode=4,
        sample_seed=7,
    )

    assert first_episode_selection == repeated_episode_selection
    assert len(first_episode_selection) == 3
    assert set(first_episode_selection).issubset(source_paths)
    assert selected_frame_indices == sorted(selected_frame_indices)
    assert len(selected_frame_indices) == 4
    assert all(frame_index % 30 == 0 for frame_index in selected_frame_indices)
