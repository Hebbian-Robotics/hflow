from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import hflow
import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec
from inspect_ai.model import ModelOutput, ModelUsage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.egosuite_evaluation.evaluate import (
    CameraView,
    EvaluatedSample,
    EvaluationConfiguration,
    ExecutionErrorSampleOutcome,
    InvalidResponseSampleOutcome,
    ProjectedHandFrameLabel,
    RunMetadata,
    SampleOutcome,
    SampleResponseMetadata,
    SuccessfulSampleOutcome,
    _evaluation_result_summary,
    _prepare_output_directory,
    _sample_result,
    _selected_frame_indices,
    load_projected_hand_label_report,
    main,
    parse_hand_count_response,
    select_episode_paths,
    select_stratified_labels,
    summarize_evaluation_results,
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


def _outcome_response() -> SampleResponseMetadata:
    return SampleResponseMetadata(response_model=None, latency_seconds=None, usage=None)


def _evaluated_sample(
    frame_id: str, expected_value: int, outcome: SampleOutcome
) -> EvaluatedSample:
    return EvaluatedSample(
        frame_id=frame_id,
        source_episode="episode-0",
        expected_value=expected_value,
        outcome=outcome,
    )


def test_evaluation_summary_reports_accuracy_and_confusion_without_counting_failures() -> None:
    results = [
        _evaluated_sample(
            "0",
            2,
            SuccessfulSampleOutcome(
                raw_response="2", response_metadata=_outcome_response(), predicted_value=2
            ),
        ),
        _evaluated_sample(
            "1",
            1,
            SuccessfulSampleOutcome(
                raw_response="2", response_metadata=_outcome_response(), predicted_value=2
            ),
        ),
        _evaluated_sample(
            "2",
            0,
            InvalidResponseSampleOutcome(
                raw_response="no count",
                response_metadata=_outcome_response(),
                parse_error="no count",
            ),
        ),
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


def _eval_log(samples: list[EvalSample]) -> EvalLog:
    return EvalLog(
        eval=EvalSpec(
            created="2026-08-31T00:00:00Z",
            task="t",
            dataset=EvalDataset(),
            model="m",
            config=EvalConfig(),
        ),
        samples=samples,
    )


def _eval_sample(frame_id: str, output: ModelOutput, target: str = "2") -> EvalSample:
    return EvalSample(
        id=frame_id,
        input="image",
        target=target,
        epoch=1,
        metadata={"source_episode": "episode-0"},
        output=output,
    )


def test_sample_result_outcome_variants_are_exclusive() -> None:
    log = _eval_log(
        [
            _eval_sample("ok", ModelOutput(completion="2", model="m")),
            _eval_sample("invalid", ModelOutput(completion="not a count")),
            _eval_sample("error", ModelOutput(completion="garbage", error="boom")),
        ]
    )

    samples = log.samples
    assert samples is not None
    results = [_sample_result(log, sample) for sample in samples]

    success = results[0].outcome
    assert isinstance(success, SuccessfulSampleOutcome)
    assert success.predicted_value == 2
    assert success.raw_response == "2"
    assert success.response_metadata.response_model == "m"
    assert not hasattr(success, "parse_error")
    assert not hasattr(success, "error")
    invalid = results[1].outcome
    assert isinstance(invalid, InvalidResponseSampleOutcome)
    assert invalid.parse_error == "hand count must be 0, 1, or 2"
    assert invalid.raw_response == "not a count"
    assert not hasattr(invalid, "predicted_value")
    error = results[2].outcome
    assert isinstance(error, ExecutionErrorSampleOutcome)
    assert error.error == "boom"
    assert error.raw_response == "garbage"
    assert not hasattr(error, "predicted_value")
    assert results[0].expected_value == 2
    assert results[0].source_episode == "episode-0"


def test_evaluation_summary_counts_error_samples_latency_and_usage() -> None:
    results = [
        _evaluated_sample(
            "0",
            2,
            SuccessfulSampleOutcome(
                raw_response="2",
                response_metadata=SampleResponseMetadata(
                    response_model="m", latency_seconds=1.0, usage={"total_tokens": 3}
                ),
                predicted_value=2,
            ),
        ),
        _evaluated_sample(
            "1",
            2,
            ExecutionErrorSampleOutcome(
                raw_response="garbage",
                response_metadata=SampleResponseMetadata(
                    response_model=None, latency_seconds=3.0, usage=None
                ),
                error="boom",
            ),
        ),
    ]

    summary = _evaluation_result_summary(results)

    assert summary["attempted_count"] == 2
    assert summary["valid_count"] == 1
    assert summary["invalid_count"] == 0
    assert summary["error_count"] == 1
    assert summary["agreement_count"] == 1
    assert summary["average_latency_seconds"] == 2.0
    assert summary["usage_totals"] == {"total_tokens": 3.0}
    assert summary["confusion_matrix"] == {
        "0": {"0": 0, "1": 0, "2": 0},
        "1": {"0": 0, "1": 0, "2": 0},
        "2": {"0": 0, "1": 0, "2": 1},
    }


def test_sample_result_captures_usage_and_latency_from_the_inspect_log() -> None:
    log = _eval_log(
        [
            _eval_sample(
                "ok",
                ModelOutput(
                    completion="1",
                    model="m",
                    time=1.5,
                    usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
                ),
            )
        ]
    )

    samples = log.samples
    assert samples is not None
    result = _sample_result(log, samples[0])

    assert isinstance(result.outcome, SuccessfulSampleOutcome)
    assert result.outcome.response_metadata.latency_seconds == 1.5
    assert result.outcome.response_metadata.usage is not None
    assert result.outcome.response_metadata.usage["total_tokens"] == 12


def _evaluation_configuration(output_directory: Path) -> EvaluationConfiguration:
    source_path = output_directory.parent / "sources" / "episode-0.mcap"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"")
    return EvaluationConfiguration(
        source_paths=(source_path,),
        camera_view=CameraView.HEAD_LEFT,
        frame_stride=30,
        limit_per_episode=None,
        episode_count=None,
        samples_per_episode=None,
        samples_per_hand_count=None,
        sample_seed=42,
        output_directory=output_directory,
        model="vision-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key_environment_variable="HFLOW_TEST_API_KEY",
        allow_missing_api_key=True,
        response_format=ResponseFormat.JSON_SCHEMA,
        temperature=None,
        max_tokens=512,
        max_retries=2,
        worker_count=2,
        prompt="count hands",
        prompt_path=Path("prompts/hand-count.txt"),
        label="run-label",
    )


def test_prepare_output_directory_writes_and_resumes_the_same_fingerprint(
    tmp_path: Path,
) -> None:
    configuration = _evaluation_configuration(tmp_path / "run")
    metadata_path = tmp_path / "run" / "run.json"

    first = _prepare_output_directory(configuration)
    second = _prepare_output_directory(configuration)

    assert metadata_path.is_file()
    assert second == first
    assert second.fingerprint == first.fingerprint
    assert second.label == "run-label"
    assert second.model == "vision-model"
    assert second.frame_stride == 30
    assert second.sample_seed == 42


def test_prepare_output_directory_refuses_a_different_experiment(tmp_path: Path) -> None:
    configuration = _evaluation_configuration(tmp_path / "run")
    _prepare_output_directory(configuration)

    different = replace(configuration, model="other-model")

    with pytest.raises(ValueError, match="describes a different experiment"):
        _prepare_output_directory(different)


def test_run_metadata_refuses_a_non_object_run_json(tmp_path: Path) -> None:
    configuration = _evaluation_configuration(tmp_path / "run")
    metadata_path = tmp_path / "run" / "run.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("[]")

    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)

    message = str(error.value)
    assert str(metadata_path) in message
    assert "must contain a JSON object" in message


def test_run_metadata_refuses_invalid_json(tmp_path: Path) -> None:
    configuration = _evaluation_configuration(tmp_path / "run")
    metadata_path = tmp_path / "run" / "run.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("not json")

    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)

    message = str(error.value)
    assert str(metadata_path) in message
    assert "could not read run metadata" in message


def test_run_metadata_names_the_file_and_the_bad_field(tmp_path: Path) -> None:
    configuration = _evaluation_configuration(tmp_path / "run")
    metadata_path = tmp_path / "run" / "run.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path.write_text(json.dumps({"fingerprint": "x"}))
    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)
    message = str(error.value)
    assert str(metadata_path) in message
    assert "'label'" in message

    metadata_path.write_text(json.dumps({"label": 3}))
    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)
    message = str(error.value)
    assert str(metadata_path) in message
    assert "'label'" in message

    metadata_path.write_text(
        json.dumps(
            {
                "label": "run-label",
                "fingerprint": "x",
                "model": "vision-model",
                "camera_view": "head-left",
                "frame_stride": "30",
            }
        )
    )
    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)
    message = str(error.value)
    assert str(metadata_path) in message
    assert "'frame_stride'" in message

    metadata_path.write_text(
        json.dumps(
            {
                "label": "run-label",
                "fingerprint": "x",
                "model": "vision-model",
                "camera_view": "head-left",
                "frame_stride": 30,
                "sample_seed": True,
            }
        )
    )
    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)
    message = str(error.value)
    assert str(metadata_path) in message
    assert "'sample_seed'" in message

    metadata_path.write_text(
        json.dumps(
            {
                "label": "run-label",
                "fingerprint": "x",
                "model": "vision-model",
                "camera_view": "head-left",
                "frame_stride": 30,
                "sample_seed": 42,
                "episode_count": "none",
            }
        )
    )
    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)
    message = str(error.value)
    assert str(metadata_path) in message
    assert "'episode_count'" in message


def test_run_metadata_document_persists_the_existing_schema(tmp_path: Path) -> None:
    configuration = _evaluation_configuration(tmp_path / "run")
    metadata_path = tmp_path / "run" / "run.json"

    metadata = _prepare_output_directory(configuration)

    document = json.loads(metadata_path.read_text())
    assert document == metadata.to_json_dict()
    assert set(document) == {
        "adapter_schema_version",
        "api_key_environment_variable",
        "base_url",
        "camera_view",
        "episode_count",
        "fingerprint",
        "frame_stride",
        "inspect_ai_version",
        "label",
        "limit_per_episode",
        "max_retries",
        "max_tokens",
        "model",
        "projection_contract",
        "prompt",
        "response_format",
        "sample_seed",
        "samples_per_episode",
        "samples_per_hand_count",
        "schema_version",
        "sources",
        "temperature",
        "worker_count",
    }
    assert document["label"] == "run-label"
    assert document["episode_count"] is None
    assert document["samples_per_hand_count"] is None
    assert document["schema_version"] == 1


def test_summarize_evaluation_results_persists_the_existing_schema() -> None:
    run_metadata = RunMetadata(
        label="run-label",
        fingerprint="0" * 64,
        model="vision-model",
        camera_view="head-left",
        frame_stride=30,
        episode_count=None,
        samples_per_episode=None,
        samples_per_hand_count=2,
        sample_seed=42,
        document={},
    )
    results = [
        _evaluated_sample(
            "0",
            2,
            SuccessfulSampleOutcome(
                raw_response="2", response_metadata=_outcome_response(), predicted_value=2
            ),
        ),
        _evaluated_sample(
            "0",
            2,
            SuccessfulSampleOutcome(
                raw_response="2", response_metadata=_outcome_response(), predicted_value=1
            ),
        ),
    ]

    summary = summarize_evaluation_results(run_metadata, results)
    overall = summary["overall"]
    assert isinstance(overall, dict)

    assert set(summary) == {
        "schema_version",
        "label",
        "fingerprint",
        "model",
        "camera_view",
        "frame_stride",
        "episode_count",
        "samples_per_episode",
        "samples_per_hand_count",
        "sample_seed",
        "overall",
        "episodes",
    }
    assert set(overall) == {
        "attempted_count",
        "valid_count",
        "invalid_count",
        "error_count",
        "expected_value_counts",
        "predicted_value_counts",
        "agreement_count",
        "agreement_fraction",
        "attempted_agreement_fraction",
        "macro_agreement_fraction",
        "macro_attempted_agreement_fraction",
        "per_class_agreement",
        "confusion_matrix",
        "average_latency_seconds",
        "usage_totals",
    }
    assert summary["episodes"] == {"episode-0": overall}
    assert overall["attempted_count"] == 1
    assert overall["predicted_value_counts"] == {1: 1}


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
