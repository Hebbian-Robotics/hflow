from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.build_ai_evaluation.evaluate import (
    SourceSelection,
    _argument_parser,
    _run_configuration_from_arguments,
    _sanitize_base_url,
    iter_evaluation_frames,
    summarize_results,
)
from examples.build_ai_evaluation.judgment import (
    EvaluationTask,
    ResponseFormat,
    TaskDefinition,
    evaluate_image_with_model,
    parse_active_manipulation_response,
    parse_hand_count_response,
)
from examples.build_ai_evaluation.pipeline import VISION_ENDPOINT_ALIAS, app


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

    assert outcome.predicted_value == 2
    assert outcome.parse_error is None
    assert outcome.check_result.measurements == {
        "build_ai/hand_count/raw_response": '{"hand_count": 2}',
        "build_ai/hand_count/requested_model": "requested-vision-model",
        "build_ai/hand_count/prediction": 2,
        "build_ai/hand_count/response_model": "routed-vision-model",
        "build_ai/hand_count/usage/prompt_tokens": 10,
        "build_ai/hand_count/usage/completion_tokens": 3,
    }


def test_build_ai_pipeline_registers_both_judgments_as_hflow_checks() -> None:
    checks_by_name = {check.name: check for check in app.checks}

    assert set(checks_by_name) == {
        "build_ai_active_manipulation",
        "build_ai_hand_count",
    }
    assert all(check.uses == VISION_ENDPOINT_ALIAS for check in checks_by_name.values())


def test_summary_reports_prevalence_agreement_and_failures_without_counting_failures_negative() -> (
    None
):
    run_metadata: dict[str, object] = {
        "label": "candidate",
        "fingerprint": "fingerprint",
        "dataset_variant": "10k",
        "model": "candidate-model",
        "prompts": {
            EvaluationTask.HAND_COUNT.value: {"sha256": "hand-hash"},
            EvaluationTask.ACTIVE_MANIPULATION.value: {"sha256": "active-hash"},
        },
    }
    results: list[dict[str, object]] = [
        {
            "source_dataset": "egocentric10k",
            "frame_id": "frame-1",
            "task": EvaluationTask.HAND_COUNT.value,
            "status": "ok",
            "predicted_value": 2,
            "expected_value": 2,
            "latency_seconds": 1.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "source_dataset": "egocentric10k",
            "frame_id": "frame-2",
            "task": EvaluationTask.HAND_COUNT.value,
            "status": "ok",
            "predicted_value": 0,
            "expected_value": 1,
            "latency_seconds": 3.0,
            "usage": {"prompt_tokens": 12, "completion_tokens": 2},
        },
        {
            "source_dataset": "egocentric10k",
            "frame_id": "frame-1",
            "task": EvaluationTask.ACTIVE_MANIPULATION.value,
            "status": "ok",
            "predicted_value": "yes",
            "expected_value": "yes",
            "latency_seconds": 2.0,
        },
        {
            "source_dataset": "egocentric10k",
            "frame_id": "frame-2",
            "task": EvaluationTask.ACTIVE_MANIPULATION.value,
            "status": "error",
            "latency_seconds": 4.0,
        },
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
