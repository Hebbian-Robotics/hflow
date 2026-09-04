from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import duckdb
import hflow
import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec
from inspect_ai.model import ModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.build_ai_evaluation.evaluate import (
    BuildAIRunMetadata,
    DatasetVariant,
    EvaluatedSample,
    EvaluationConfiguration,
    ExecutionErrorSampleOutcome,
    InvalidResponseSampleOutcome,
    SampleResponseMetadata,
    SourceSelection,
    SuccessfulSampleOutcome,
    _argument_parser,
    _prepare_output_directory,
    _run_configuration_from_arguments,
    _sample_result,
    _sanitize_base_url,
    iter_evaluation_frames,
    main,
    summarize_results,
)
from examples.build_ai_evaluation.pipeline import (
    _argument_parser as pipeline_argument_parser,
)
from examples.build_ai_evaluation.pipeline import (
    _execution_from_environment,
    app,
)
from hflow.build_ai_vlm_checks import (
    ACTIVE_MANIPULATION_RESPONSE_SCHEMA,
    HAND_COUNT_RESPONSE_SCHEMA,
    EvaluationTask,
    ParsedVisionModelOutcome,
    ResponseFormat,
    TaskDefinition,
    UnparsedVisionModelOutcome,
    evaluate_image_with_model,
    model_output_check_result,
    parse_active_manipulation_response,
    parse_hand_count_response,
)


class _FixtureUsage:
    def model_dump(self, *, exclude_none: bool) -> dict[str, int]:
        assert exclude_none is True
        return {"prompt_tokens": 10, "completion_tokens": 3}


class _FixtureCompletions:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def create(self, **_request_parameters: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))],
            model="routed-vision-model",
            usage=_FixtureUsage(),
        )


class _FixtureOpenAICompatibleClient:
    def __init__(self, response_text: str) -> None:
        self.chat = SimpleNamespace(completions=_FixtureCompletions(response_text))


_TASK_VALUE_CONTRACTS: dict[
    EvaluationTask,
    tuple[dict[str, object], str, Callable[[str], int | str], tuple[object, ...]],
] = {
    EvaluationTask.HAND_COUNT: (
        HAND_COUNT_RESPONSE_SCHEMA,
        "hand_count",
        parse_hand_count_response,
        (-1, 3),
    ),
    EvaluationTask.ACTIVE_MANIPULATION: (
        ACTIVE_MANIPULATION_RESPONSE_SCHEMA,
        "answer",
        parse_active_manipulation_response,
        ("maybe", ""),
    ),
}


def test_task_schemas_and_parsers_match_the_published_contract() -> None:
    """Preserve Build AI's schema shapes and enforce its prose vocabulary."""
    executable_tasks = set(EvaluationTask) - {EvaluationTask.BOTH}
    assert set(_TASK_VALUE_CONTRACTS) == executable_tasks

    for task, (schema, property_name, parser, rejected_values) in _TASK_VALUE_CONTRACTS.items():
        properties = cast(dict[str, dict[str, object]], schema["properties"])
        property_schema = properties[property_name]
        permitted_values = (
            [0, 1, 2]
            if task is EvaluationTask.HAND_COUNT
            else cast(list[object], property_schema["enum"])
        )
        if task is EvaluationTask.HAND_COUNT:
            # The published schema says INTEGER without an enum. Keeping that
            # exact shape also avoids Gemini/OpenRouter emitting an empty object.
            assert property_schema == {"type": "integer"}
        for value in permitted_values:
            assert parser(json.dumps({property_name: value})) == value, (
                f"{task.value}: the schema permits {value!r} but the parser does not accept it"
            )
        for value in rejected_values:
            with pytest.raises(ValueError):
                parser(json.dumps({property_name: value}))


@pytest.mark.parametrize(
    ("response_text", "expected_hand_count"),
    [
        ('{"hand_count": 0}', 0),
        ("1", 1),
        ('```json\n{"hand_count": 2}\n```', 2),
    ],
)
def test_hand_count_parser_accepts_published_and_compatible_shapes(
    response_text: str, expected_hand_count: int
) -> None:
    assert parse_hand_count_response(response_text) == expected_hand_count


@pytest.mark.parametrize("response_text", ["3", '{"hand_count": true}', "two", ""])
def test_hand_count_parser_rejects_values_outside_the_evaluation_contract(
    response_text: str,
) -> None:
    with pytest.raises(ValueError, match="0, 1, or 2"):
        parse_hand_count_response(response_text)


@pytest.mark.parametrize(
    ("response_text", "expected_answer"),
    [
        ('{"answer": "yes"}', "yes"),
        ("NO", "no"),
        ('```json\n{"answer": "yes"}\n```', "yes"),
    ],
)
def test_active_manipulation_parser_accepts_published_and_compatible_shapes(
    response_text: str, expected_answer: str
) -> None:
    assert parse_active_manipulation_response(response_text) == expected_answer


def test_model_judgment_returns_hflow_measurements_for_the_replay_and_pipeline() -> None:
    outcome = evaluate_image_with_model(
        client=_FixtureOpenAICompatibleClient('{"hand_count": 2}'),
        model="requested-vision-model",
        task_definition=TaskDefinition(
            task=EvaluationTask.HAND_COUNT,
            prompt="Count hands.",
            response_schema={"type": "object"},
        ),
        image_data_url="data:image/png;base64,fixture",
        response_format=ResponseFormat.JSON_SCHEMA,
        temperature=None,
        max_tokens=32,
    )

    assert isinstance(outcome, ParsedVisionModelOutcome)
    assert outcome.predicted_value == 2
    check_result = model_output_check_result(
        task=EvaluationTask.HAND_COUNT,
        requested_model="requested-vision-model",
        outcome=outcome,
        observation_id="frame:123",
        timestamp_ns=123,
    )

    assert check_result.measurements == {
        "build_ai/hand_count/raw_response": '{"hand_count": 2}',
        "build_ai/hand_count/requested_model": "requested-vision-model",
        "build_ai/hand_count/prediction": 2,
        "build_ai/hand_count/response_model": "routed-vision-model",
        "build_ai/hand_count/usage/prompt_tokens": 10,
        "build_ai/hand_count/usage/completion_tokens": 3,
    }
    assert check_result.observations == [
        hflow.Observation(
            observation_id="frame:123",
            timestamp_ns=123,
            values={
                "task": "hand-count",
                "raw_response": '{"hand_count": 2}',
                "requested_model": "requested-vision-model",
                "valid": True,
                "prediction": 2,
                "response_model": "routed-vision-model",
                "usage/prompt_tokens": 10,
                "usage/completion_tokens": 3,
            },
        )
    ]


def test_unparsed_model_judgment_is_an_explicit_recoverable_outcome() -> None:
    outcome = evaluate_image_with_model(
        client=_FixtureOpenAICompatibleClient("unclear"),
        model="requested-vision-model",
        task_definition=TaskDefinition(
            task=EvaluationTask.HAND_COUNT,
            prompt="Count hands.",
            response_schema={"type": "object"},
        ),
        image_data_url="data:image/png;base64,fixture",
        response_format=ResponseFormat.JSON_SCHEMA,
        temperature=None,
        max_tokens=32,
    )

    assert isinstance(outcome, UnparsedVisionModelOutcome)
    check_result = model_output_check_result(
        task=EvaluationTask.HAND_COUNT,
        requested_model="requested-vision-model",
        outcome=outcome,
        observation_id="frame:123",
        timestamp_ns=123,
    )
    assert check_result.measurements["build_ai/hand_count/parse_error"] == (
        "hand count must be 0, 1, or 2"
    )
    assert check_result.observations[0].values["valid"] is False
    assert check_result.tags == ["build_ai/hand_count/unparsed"]


def test_build_ai_pipeline_registers_both_judgments_as_hflow_checks() -> None:
    checks_by_name = {check.name: check for check in app.checks}

    assert set(checks_by_name) == {
        "build_ai_active_manipulation",
        "build_ai_hand_visibility",
    }
    assert all(check.requires == frozenset({"vision-model"}) for check in checks_by_name.values())


def test_build_ai_pipeline_defaults_to_hosted_and_can_select_openai_compatible_execution() -> None:
    hosted_execution = _execution_from_environment("BUILD_AI_HAND_VISIBILITY", {})
    openai_compatible_execution = _execution_from_environment(
        "BUILD_AI_HAND_VISIBILITY",
        {
            "BUILD_AI_EXECUTION": "hflow-hosted",
            "BUILD_AI_HAND_VISIBILITY_EXECUTION": "openai-compatible",
            "BUILD_AI_HAND_VISIBILITY_BASE_URL": "http://localhost:8000/v1",
            "BUILD_AI_HAND_VISIBILITY_MODEL": "local-vision-model",
        },
    )

    assert hosted_execution == hflow.build_ai_vlm_checks.HFlowHostedExecution()
    assert openai_compatible_execution == hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
        endpoint="http://localhost:8000/v1",
        model="local-vision-model",
    )


def test_build_ai_pipeline_requires_an_episode_with_meaningful_footage() -> None:
    with pytest.raises(SystemExit):
        pipeline_argument_parser().parse_args([])

    arguments = pipeline_argument_parser().parse_args(["recording.mcap"])

    assert arguments.episode == Path("recording.mcap")


def test_summary_reports_prevalence_agreement_and_failures_without_counting_failures_negative() -> (
    None
):
    run_metadata = BuildAIRunMetadata(
        label="candidate",
        fingerprint="fingerprint",
        dataset_variant="10k",
        model="candidate-model",
        prompt_sha256s={
            EvaluationTask.HAND_COUNT.value: "hand-hash",
            EvaluationTask.ACTIVE_MANIPULATION.value: "active-hash",
        },
        document={},
    )
    results = [
        EvaluatedSample(
            source_dataset="egocentric10k",
            frame_id="frame-1",
            task=EvaluationTask.HAND_COUNT,
            expected_value=2,
            outcome=SuccessfulSampleOutcome(
                raw_response="2",
                response_metadata=SampleResponseMetadata(
                    response_model=None,
                    latency_seconds=1.0,
                    usage={"prompt_tokens": 10, "completion_tokens": 2},
                ),
                predicted_value=2,
            ),
        ),
        EvaluatedSample(
            source_dataset="egocentric10k",
            frame_id="frame-2",
            task=EvaluationTask.HAND_COUNT,
            expected_value=1,
            outcome=SuccessfulSampleOutcome(
                raw_response="0",
                response_metadata=SampleResponseMetadata(
                    response_model=None,
                    latency_seconds=3.0,
                    usage={"prompt_tokens": 12, "completion_tokens": 2},
                ),
                predicted_value=0,
            ),
        ),
        EvaluatedSample(
            source_dataset="egocentric10k",
            frame_id="frame-1",
            task=EvaluationTask.ACTIVE_MANIPULATION,
            expected_value="yes",
            outcome=SuccessfulSampleOutcome(
                raw_response="yes",
                response_metadata=SampleResponseMetadata(
                    response_model=None, latency_seconds=2.0, usage=None
                ),
                predicted_value="yes",
            ),
        ),
        EvaluatedSample(
            source_dataset="egocentric10k",
            frame_id="frame-2",
            task=EvaluationTask.ACTIVE_MANIPULATION,
            expected_value="no",
            outcome=ExecutionErrorSampleOutcome(
                raw_response="",
                response_metadata=SampleResponseMetadata(
                    response_model=None, latency_seconds=4.0, usage=None
                ),
                error="boom",
            ),
        ),
    ]

    summary = summarize_results(run_metadata, results)
    summaries_by_source = cast(dict[str, dict[str, dict[str, object]]], summary["sources"])
    source_summary = summaries_by_source["egocentric10k"]
    hand_summary = source_summary[EvaluationTask.HAND_COUNT.value]
    active_summary = source_summary[EvaluationTask.ACTIVE_MANIPULATION.value]

    assert hand_summary["predicted_value_fractions"] == {"0": 0.5, "2": 0.5}
    assert hand_summary["reference_value_fractions"] == {"1": 0.5, "2": 0.5}
    assert hand_summary["agreement_fraction"] == 0.5
    assert hand_summary["average_latency_seconds"] == 2.0
    assert hand_summary["usage_totals"] == {"completion_tokens": 4.0, "prompt_tokens": 22.0}
    assert active_summary["valid_count"] == 1
    assert active_summary["error_count"] == 1
    assert active_summary["predicted_value_fractions"] == {"yes": 1.0}


def _eval_log(metadata: dict[str, str], samples: list[EvalSample]) -> EvalLog:
    return EvalLog(
        eval=EvalSpec(
            created="2026-08-31T00:00:00Z",
            task="t",
            dataset=EvalDataset(),
            model="m",
            config=EvalConfig(),
            metadata=metadata,
        ),
        samples=samples,
    )


def _eval_sample(frame_id: str, output: ModelOutput, target: str = "2") -> EvalSample:
    return EvalSample(
        id=frame_id,
        input="image",
        target=target,
        epoch=1,
        output=output,
    )


def test_sample_result_outcome_variants_are_exclusive() -> None:
    log = _eval_log(
        {"evaluation_task": "hand-count", "source_dataset": "egocentric10k"},
        [
            _eval_sample("ok", ModelOutput(completion='{"hand_count": 2}', model="m")),
            _eval_sample("invalid", ModelOutput(completion="not a count")),
            _eval_sample("error", ModelOutput(completion="garbage", error="boom")),
        ],
    )
    samples = log.samples
    assert samples is not None

    results = [_sample_result(log, sample) for sample in samples]

    success = results[0].outcome
    assert isinstance(success, SuccessfulSampleOutcome)
    assert success.predicted_value == 2
    assert success.raw_response == '{"hand_count": 2}'
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
    assert results[0].task == EvaluationTask.HAND_COUNT
    assert results[0].source_dataset == "egocentric10k"


def _evaluation_configuration(output_directory: Path) -> EvaluationConfiguration:
    data_directory = output_directory.parent / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    return EvaluationConfiguration(
        dataset_variant=DatasetVariant.EGOCENTRIC_10K,
        selected_sources=(SourceSelection.BUILD_AI,),
        selected_tasks=(EvaluationTask.HAND_COUNT,),
        data_directory=data_directory,
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
        row_limit_per_source=None,
        label="run-label",
        task_definitions={
            EvaluationTask.HAND_COUNT: TaskDefinition(
                task=EvaluationTask.HAND_COUNT,
                prompt="count hands",
                response_schema=HAND_COUNT_RESPONSE_SCHEMA,
            )
        },
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
    assert second.prompt_sha256s == first.prompt_sha256s


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
                "dataset_variant": "10k",
            }
        )
    )
    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)
    message = str(error.value)
    assert str(metadata_path) in message
    assert "'prompts'" in message

    metadata_path.write_text(
        json.dumps(
            {
                "label": "run-label",
                "fingerprint": "x",
                "model": "vision-model",
                "dataset_variant": "10k",
                "prompts": {"hand-count": {"text": "no digest"}},
            }
        )
    )
    with pytest.raises(ValueError) as error:
        _prepare_output_directory(configuration)
    message = str(error.value)
    assert str(metadata_path) in message
    assert "'hand-count'" in message
    assert "'sha256'" in message


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
        "dataset_repository",
        "dataset_revision",
        "dataset_variant",
        "fingerprint",
        "inspect_ai_version",
        "label",
        "max_retries",
        "max_tokens",
        "model",
        "prompts",
        "response_format",
        "row_limit_per_source",
        "schema_version",
        "sources",
        "tasks",
        "temperature",
        "worker_count",
    }
    assert document["label"] == "run-label"
    assert document["row_limit_per_source"] is None
    assert document["schema_version"] == 1


def test_summarize_results_persists_the_existing_schema() -> None:
    run_metadata = BuildAIRunMetadata(
        label="run-label",
        fingerprint="0" * 64,
        dataset_variant="10k",
        model="vision-model",
        prompt_sha256s={EvaluationTask.HAND_COUNT.value: "hand-hash"},
        document={},
    )
    results = [
        EvaluatedSample(
            source_dataset="egocentric10k",
            frame_id="frame-1",
            task=EvaluationTask.HAND_COUNT,
            expected_value=2,
            outcome=SuccessfulSampleOutcome(
                raw_response="2",
                response_metadata=SampleResponseMetadata(
                    response_model=None, latency_seconds=None, usage=None
                ),
                predicted_value=2,
            ),
        )
    ]

    summary = summarize_results(run_metadata, results)
    source_summaries = summary["sources"]
    assert isinstance(source_summaries, dict)

    assert set(summary) == {
        "schema_version",
        "label",
        "fingerprint",
        "dataset_variant",
        "model",
        "prompts",
        "sources",
    }
    assert summary["prompts"] == {"hand-count": {"sha256": "hand-hash"}}
    task_summaries = source_summaries["egocentric10k"]
    assert isinstance(task_summaries, dict)
    hand_summary = task_summaries[EvaluationTask.HAND_COUNT.value]
    assert isinstance(hand_summary, dict)
    assert set(hand_summary) == {
        "attempted_count",
        "valid_count",
        "invalid_count",
        "error_count",
        "predicted_value_counts",
        "predicted_value_fractions",
        "reference_value_counts",
        "reference_value_fractions",
        "agreement_count",
        "agreement_fraction",
        "average_latency_seconds",
        "usage_totals",
    }
    assert hand_summary["attempted_count"] == 1
    assert hand_summary["agreement_fraction"] == 1.0


def test_base_url_metadata_drops_embedded_credentials_and_query_values() -> None:
    assert (
        _sanitize_base_url("https://user:secret@example.com/v1?api_key=secret")
        == "https://example.com/v1"
    )


def test_cli_preserves_typed_source_and_task_selections() -> None:
    arguments = _argument_parser().parse_args(
        [
            "run",
            "--dataset",
            "100k",
            "--source",
            "build",
            "--task",
            "hand-count",
            "--model",
            "vision-model",
            "--base-url",
            "http://localhost:8000/v1",
        ]
    )

    configuration = _run_configuration_from_arguments(arguments)

    assert configuration.selected_sources == (SourceSelection.BUILD_AI,)
    assert configuration.selected_tasks == (EvaluationTask.HAND_COUNT,)


@pytest.mark.parametrize(
    ("include_frame_id", "active_labor", "expected_active_manipulation"),
    [
        (True, "yes", "yes"),
        (False, "true", "yes"),
        (False, "false", "no"),
    ],
)
def test_evaluation_reader_supports_both_published_parquet_schemas(
    tmp_path: Path,
    include_frame_id: bool,
    active_labor: str,
    expected_active_manipulation: str,
) -> None:
    parquet_path = tmp_path / "evaluation.parquet"
    connection = duckdb.connect()
    try:
        frame_id_selection = "'upstream-frame-id' AS frame_id, " if include_frame_id else ""
        connection.execute(
            f"CREATE TABLE evaluation AS SELECT {frame_id_selection}"
            "{'bytes': ?::BLOB, 'path': NULL::VARCHAR} AS image, "
            "'egocentric' AS source_dataset, 2::INTEGER AS hand_count, "
            "?::VARCHAR AS active_labor",
            [b"\x89PNG\r\n\x1a\nfixture", active_labor],
        )
        escaped_parquet_path = str(parquet_path).replace("'", "''")
        connection.execute(f"COPY evaluation TO '{escaped_parquet_path}' (FORMAT PARQUET)")
    finally:
        connection.close()

    frames = list(iter_evaluation_frames(parquet_path, source_dataset_name="published-corpus"))

    assert len(frames) == 1
    assert frames[0].frame_id == ("upstream-frame-id" if include_frame_id else "row-00000")
    assert frames[0].source_dataset == "published-corpus"
    assert frames[0].expected_hand_count == 2
    assert frames[0].expected_active_manipulation == expected_active_manipulation
    assert frames[0].image_data_url.startswith("data:image/png;base64,")


def test_compare_missing_file_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_file = "/tmp/nope_file_does_not_exist.json"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "compare", missing_file])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "No such file or directory" in captured.err
    assert missing_file in captured.err
    assert "Traceback" not in captured.err


def test_compare_malformed_json_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not json content")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "compare", str(bad_json)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid JSON in" in captured.err
    assert str(bad_json) in captured.err
    assert "Traceback" not in captured.err


def test_compare_success_exits_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_data = {
        "label": "run1",
        "dataset_variant": "10k",
        "sources": {
            "egocentric": {
                EvaluationTask.HAND_COUNT.value: {
                    "valid_count": 10,
                    "predicted_value_fractions": {"0": 0.1, "2": 0.8},
                    "reference_value_fractions": {"2": 0.9},
                    "agreement_fraction": 0.85,
                },
                EvaluationTask.ACTIVE_MANIPULATION.value: {
                    "valid_count": 10,
                    "predicted_value_fractions": {"yes": 0.9},
                    "reference_value_fractions": {"yes": 0.95},
                    "agreement_fraction": 0.9,
                },
            }
        },
    }
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary_data))

    monkeypatch.setattr(sys, "argv", ["evaluate.py", "compare", str(summary_file)])
    main()

    captured = capsys.readouterr()
    assert "| run | dataset | source |" in captured.out
    assert "| run1 | 10k | egocentric |" in captured.out


def test_run_missing_file_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_file = "/tmp/nope.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "run",
            "--dataset",
            "10k",
            "--model",
            "x",
            "--base-url",
            "http://127.0.0.1:1",
            "--hand-count-prompt",
            missing_file,
            "--allow-missing-api-key",
            "--limit",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "No such file or directory" in captured.err
    assert missing_file in captured.err
    assert "Traceback" not in captured.err
