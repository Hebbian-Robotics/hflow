from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import hflow
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.egosuite_evaluation.evaluate import (
    CameraView,
    ProjectedHandFrameLabel,
    _evaluation_result_summary,
    _selected_frame_indices,
    load_projected_hand_label_report,
    main,
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
from examples.egosuite_evaluation.judgment import (
    ModelResponseMetadata,
    ParsedHandCountOutcome,
    ResponseFormat,
    UnparsedHandCountOutcome,
    evaluate_image_with_model,
)
from examples.egosuite_evaluation.pipeline import (
    VISION_ENDPOINT_ALIAS,
    app,
    hand_visibility_check_result,
    labels_for_pipeline_episode,
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


def test_model_response_without_text_is_a_recoverable_invalid_judgment() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        model="routed-model",
        usage=SimpleNamespace(model_dump=lambda **_arguments: {"total_tokens": 17}),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_arguments: response),
        )
    )

    judgment = evaluate_image_with_model(
        client=client,
        model="requested-model",
        prompt="count hands",
        image_data_url="data:image/jpeg;base64,unused",
        response_format=ResponseFormat.JSON_SCHEMA,
        temperature=None,
        max_tokens=512,
    )

    assert judgment == UnparsedHandCountOutcome(
        raw_response="",
        response_metadata=ModelResponseMetadata(
            response_model="routed-model",
            usage={"total_tokens": 17},
        ),
        parse_error="endpoint returned no text completion content",
    )


def test_pipeline_registers_projected_hand_visibility_as_an_hflow_check() -> None:
    checks_by_name = {check.name: check for check in app.checks}

    assert set(checks_by_name) == {"egosuite_projected_hand_visibility"}
    assert checks_by_name["egosuite_projected_hand_visibility"].uses == VISION_ENDPOINT_ALIAS


def test_saved_label_report_selects_exact_frames_for_a_canonical_episode(tmp_path: Path) -> None:
    source_path = tmp_path / "episode-123.mcap"
    report_path = tmp_path / "labels.json"
    report_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "source_path": str(source_path),
                        "source_episode": "episode-123",
                        "camera_view": "head-left",
                        "frame_index": frame_index,
                        "left_in_frame_joint_count": 21,
                        "right_in_frame_joint_count": 21 if frame_index == 8 else 0,
                        "expected_hand_count": 2 if frame_index == 8 else 1,
                        "left_hand_issue_reasons": [],
                        "right_hand_issue_reasons": ["occlusion"] if frame_index == 8 else [],
                    }
                    for frame_index in (8, 3)
                ]
            }
        )
    )

    labels_by_source_episode = load_projected_hand_label_report(report_path)
    selected_labels = labels_for_pipeline_episode(
        tmp_path / "episode-123.canonical.mcap", labels_by_source_episode
    )

    assert [label.frame_index for label in selected_labels] == [3, 8]
    assert [label.expected_hand_count for label in selected_labels] == [1, 2]
    assert selected_labels[1].right_hand_issue_reasons == ("occlusion",)


def test_pipeline_records_agreement_output_validity_and_frame_intervals() -> None:
    source_path = Path("episode.mcap")
    labels = [
        ProjectedHandFrameLabel(
            source_path=source_path,
            source_episode="episode",
            camera_view=CameraView.HEAD_LEFT,
            frame_index=frame_index,
            left_in_frame_joint_count=21,
            right_in_frame_joint_count=21 if expected_hand_count == 2 else 0,
            expected_hand_count=expected_hand_count,
            left_hand_issue_reasons=(),
            right_hand_issue_reasons=("occlusion",) if frame_index == 2 else (),
        )
        for frame_index, expected_hand_count in enumerate([1, 2, 2])
    ]
    extracted_frames = [
        hflow.ExtractedFrame(path=Path(f"frame-{frame_index}.jpg"), log_time_ns=frame_index * 10)
        for frame_index in range(3)
    ]
    judgments = [
        ParsedHandCountOutcome(
            raw_response='{"hand_count": 1}',
            response_metadata=ModelResponseMetadata(
                response_model="routed-model",
                usage={"prompt_tokens": 10, "completion_tokens": 2},
            ),
            predicted_hand_count=1,
        ),
        ParsedHandCountOutcome(
            raw_response='{"hand_count": 1}',
            response_metadata=ModelResponseMetadata(
                response_model="routed-model",
                usage={"prompt_tokens": 11, "completion_tokens": 2},
            ),
            predicted_hand_count=1,
        ),
        UnparsedHandCountOutcome(
            raw_response="not a count",
            response_metadata=ModelResponseMetadata(
                response_model="routed-model",
                usage={"prompt_tokens": 12, "completion_tokens": 2},
            ),
            parse_error="hand count must be 0, 1, or 2",
        ),
    ]

    result = hand_visibility_check_result(
        labels,
        extracted_frames,
        judgments,
        requested_model="requested-model",
    )

    assert result.measurements == {
        "egosuite/hand_count/attempted_count": 3,
        "egosuite/hand_count/valid_count": 2,
        "egosuite/hand_count/invalid_count": 1,
        "egosuite/hand_count/agreement_count": 1,
        "egosuite/hand_count/attempted_agreement_fraction": pytest.approx(1 / 3),
        "egosuite/hand_count/valid_agreement_fraction": 0.5,
        "egosuite/hand_count/requested_model": "requested-model",
        "egosuite/hand_count/pose_issue_frame_count": 1,
        "egosuite/hand_count/reference/0": 0,
        "egosuite/hand_count/reference/1": 1,
        "egosuite/hand_count/reference/2": 2,
        "egosuite/hand_count/predicted/0": 0,
        "egosuite/hand_count/predicted/1": 2,
        "egosuite/hand_count/predicted/2": 0,
        "egosuite/hand_count/response_models": "routed-model",
        "egosuite/hand_count/usage/prompt_tokens": 33,
        "egosuite/hand_count/usage/completion_tokens": 6,
    }
    assert result.intervals == [
        hflow.Interval(
            start_ns=10,
            end_ns=10,
            label="egosuite/hand_count/reference_2_predicted_1",
        ),
        hflow.Interval(
            start_ns=20,
            end_ns=20,
            label="egosuite/hand_count/unparsed",
        ),
    ]
    assert result.tags == ["egosuite/hand_count/has_unparsed_output"]
    assert result.observations == [
        hflow.Observation(
            observation_id="frame:0",
            timestamp_ns=0,
            values={
                "frame_index": 0,
                "reference_hand_count": 1,
                "predicted_hand_count": 1,
                "valid": True,
                "agreement": True,
                "raw_response": '{"hand_count": 1}',
                "requested_model": "requested-model",
                "response_model": "routed-model",
                "left_in_frame_joint_count": 21,
                "right_in_frame_joint_count": 0,
                "pose_issue": False,
                "usage/prompt_tokens": 10,
                "usage/completion_tokens": 2,
            },
        ),
        hflow.Observation(
            observation_id="frame:1",
            timestamp_ns=10,
            values={
                "frame_index": 1,
                "reference_hand_count": 2,
                "predicted_hand_count": 1,
                "valid": True,
                "agreement": False,
                "raw_response": '{"hand_count": 1}',
                "requested_model": "requested-model",
                "response_model": "routed-model",
                "left_in_frame_joint_count": 21,
                "right_in_frame_joint_count": 21,
                "pose_issue": False,
                "usage/prompt_tokens": 11,
                "usage/completion_tokens": 2,
            },
        ),
        hflow.Observation(
            observation_id="frame:2",
            timestamp_ns=20,
            values={
                "frame_index": 2,
                "reference_hand_count": 2,
                "valid": False,
                "agreement": False,
                "raw_response": "not a count",
                "requested_model": "requested-model",
                "response_model": "routed-model",
                "parse_error": "hand count must be 0, 1, or 2",
                "left_in_frame_joint_count": 21,
                "right_in_frame_joint_count": 21,
                "pose_issue": True,
                "usage/prompt_tokens": 12,
                "usage/completion_tokens": 2,
            },
        ),
    ]


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


def test_compare_refuses_a_missing_summary_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_summary_path = tmp_path / "missing-summary.json"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "compare", str(missing_summary_path)])

    with pytest.raises(SystemExit) as exit_info:
        main()

    streams = capsys.readouterr()
    assert exit_info.value.code == 2
    assert str(missing_summary_path) in streams.err
    assert "Traceback" not in streams.err


def test_compare_refuses_malformed_json_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    malformed_summary_path = tmp_path / "malformed-summary.json"
    malformed_summary_path.write_text("not json")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "compare", str(malformed_summary_path)])

    with pytest.raises(SystemExit) as exit_info:
        main()

    streams = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "Expecting value" in streams.err
    assert "Traceback" not in streams.err


def test_compare_preserves_successful_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "label": "candidate",
                "model": "vision-model",
                "camera_view": "head-left",
                "overall": {
                    "predicted_value_counts": {"0": 1, "1": 2, "2": 3},
                    "attempted_count": 6,
                    "agreement_count": 4,
                    "valid_count": 5,
                    "agreement_fraction": 0.8,
                    "attempted_agreement_fraction": 2 / 3,
                    "macro_attempted_agreement_fraction": 0.5,
                },
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "compare", str(summary_path)])

    main()

    assert capsys.readouterr().out.splitlines() == [
        "| run | model | camera | valid / attempted | valid accuracy | end-to-end accuracy "
        "| macro end-to-end accuracy | predicted 0 | predicted 1 | predicted 2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        "| candidate | vision-model | head-left | 5 / 6 | 80.00% | 66.67% | 50.00% | 1 | 2 | 3 |",
    ]
