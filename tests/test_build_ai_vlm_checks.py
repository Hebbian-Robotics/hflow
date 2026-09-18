from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx2
import pytest

import hflow
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode


class _StubHostedResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    async def __aenter__(self) -> _StubHostedResponse:
        return self

    async def __aexit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        yield self._body


def test_build_ai_vlm_checks_register_independent_execution_contracts(tmp_path: Path) -> None:
    application = hflow.App("model-checks", data_root=tmp_path, default_checks=())

    hand_check = hflow.build_ai_vlm_checks.register_hand_visibility(
        application,
        execution=hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="http://hand-model.internal/v1",
            model="hand-model",
        ),
    )
    active_manipulation_check = hflow.build_ai_vlm_checks.register_active_manipulation(
        application,
        execution=hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="https://hosted.example/v1",
            model="manipulation-model",
            api_key_environment_variable="HOSTED_MODEL_API_KEY",
        ),
    )

    assert hand_check is application.checks[0].function
    assert active_manipulation_check is application.checks[1].function
    assert [registered_check.name for registered_check in application.checks] == [
        "build_ai_hand_visibility",
        "build_ai_active_manipulation",
    ]
    assert all(
        registered_check.requires == frozenset({"vision-model"})
        for registered_check in application.checks
    )
    assert application.checks[0].version != application.checks[1].version


def test_build_ai_check_version_changes_with_model_configuration(tmp_path: Path) -> None:
    first_application = hflow.App("first", data_root=tmp_path / "first", default_checks=())
    second_application = hflow.App("second", data_root=tmp_path / "second", default_checks=())

    hflow.build_ai_vlm_checks.register_hand_visibility(
        first_application,
        execution=hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="http://localhost:8000/v1",
            model="model-a",
        ),
    )
    hflow.build_ai_vlm_checks.register_hand_visibility(
        second_application,
        execution=hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="http://localhost:8000/v1",
            model="model-b",
        ),
    )

    assert first_application.checks[0].version != second_application.checks[0].version


def test_build_ai_check_version_changes_with_hosted_check_version(tmp_path: Path) -> None:
    first_application = hflow.App("first", data_root=tmp_path / "first", default_checks=())
    second_application = hflow.App("second", data_root=tmp_path / "second", default_checks=())

    hflow.build_ai_vlm_checks.register_hand_visibility(
        first_application,
        execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(check_version=1),
    )
    hflow.build_ai_vlm_checks.register_hand_visibility(
        second_application,
        execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(check_version=2),
    )

    assert first_application.checks[0].version != second_application.checks[0].version


@pytest.mark.parametrize(
    (
        "endpoint",
        "model",
        "response_format",
        "api_key_environment_variable",
        "temperature",
        "max_tokens",
        "max_retries",
        "expected_message",
    ),
    [
        (
            "localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            32,
            5,
            "absolute http",
        ),
        (
            "http://localhost:8000/v1",
            " ",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            32,
            5,
            "model must not be empty",
        ),
        (
            "http://localhost:8000/v1",
            " model ",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            32,
            5,
            "model must not have leading or trailing whitespace",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            "json",
            None,
            None,
            32,
            5,
            "response_format must be",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            "INVALID-NAME",
            None,
            32,
            5,
            "valid environment variable name",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            "32",
            5,
            "max_tokens must be an int",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            True,
            5,
            "max_tokens must be an int",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            0,
            5,
            "max_tokens must be > 0",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            32,
            "5",
            "max_retries must be an int",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            32,
            True,
            "max_retries must be an int",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            None,
            32,
            -1,
            "max_retries must be >= 0",
        ),
        (
            "http://localhost:8000/v1",
            "model",
            hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
            None,
            float("nan"),
            32,
            5,
            "temperature must be finite",
        ),
    ],
)
def test_openai_compatible_execution_refuses_invalid_configuration(
    endpoint: str,
    model: str,
    response_format: object,
    api_key_environment_variable: str | None,
    temperature: float | None,
    max_tokens: object,
    max_retries: object,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint=endpoint,
            model=model,
            response_format=response_format,  # ty: ignore
            api_key_environment_variable=api_key_environment_variable,
            temperature=temperature,
            max_tokens=max_tokens,  # ty: ignore
            max_retries=max_retries,  # ty: ignore
        )


def test_openai_compatible_execution_accepts_valid_configuration() -> None:
    execution = hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
        endpoint="http://localhost:8000/v1",
        model="model",
        api_key_environment_variable="TEST_MODEL_API_KEY",
        response_format=hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA,
        temperature=0.2,
        max_tokens=32,
        max_retries=5,
    )

    assert execution.endpoint == "http://localhost:8000/v1"
    assert execution.model == "model"
    assert execution.api_key_environment_variable == "TEST_MODEL_API_KEY"
    assert execution.response_format is hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA
    assert execution.temperature == 0.2
    assert execution.max_tokens == 32
    assert execution.max_retries == 5


@pytest.mark.parametrize("rejected_max_retries", [True, 2.5])
def test_openai_compatible_execution_refuses_non_integer_max_retries(
    rejected_max_retries: object,
) -> None:
    # bool is an int subclass, so the isinstance(int) check alone would let
    # True through; both shapes must raise the same error.
    with pytest.raises(ValueError, match="max_retries must be an int"):
        hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="https://example.com/v1",
            model="model",
            max_retries=rejected_max_retries,  # ty: ignore
        )


def test_openai_compatible_execution_accepts_an_integer_max_retries() -> None:
    # The control: without it the refusals above could pass on a constructor
    # that rejects every max_retries.
    assert hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
        endpoint="https://example.com/v1",
        model="model",
        max_retries=3,
    )


def test_hosted_execution_refuses_custom_prompt(tmp_path: Path) -> None:
    application = hflow.App("invalid-hosted-prompt", data_root=tmp_path, default_checks=())

    with pytest.raises(ValueError, match="does not support prompt overrides"):
        hflow.build_ai_vlm_checks.register_hand_visibility(
            application,
            execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(),
            prompt="Use a different definition of visibility.",
        )


def test_hosted_execution_refuses_a_base_url_that_cannot_accept_check_paths() -> None:
    with pytest.raises(ValueError, match="query string or fragment"):
        hflow.build_ai_vlm_checks.HFlowHostedExecution(
            base_url="https://checks.example?model=arbitrary"
        )


def test_hosted_execution_sends_the_selected_frame_and_returns_standard_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_episode = synthesize_episode(
        tmp_path / "episode.mcap",
        SyntheticEpisodeSpec(
            duration_s=2.0,
            cameras=("head_camera",),
            black_segment=None,
            joint_jump_at_s=None,
            timestamp_offset_segment=None,
        ),
    )
    application = hflow.App("hosted-check", data_root=tmp_path / "data", default_checks=())
    hflow.build_ai_vlm_checks.register_hand_visibility(
        application,
        execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(
            base_url="https://checks.example",
            check_version=3,
            request_timeout_seconds=12.0,
        ),
    )
    captured_requests: list[
        tuple[
            str,
            str,
            dict[str, str],
            dict[str, tuple[str, bytes, str]],
            float,
            bool,
        ]
    ] = []

    def hosted_response(
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        timeout: float,
        follow_redirects: bool,
    ) -> _StubHostedResponse:
        captured_requests.append((method, url, headers, files, timeout, follow_redirects))
        return _StubHostedResponse(
            {
                "outcome": "parsed",
                "prediction": 2,
                "raw_response": "2",
            }
        )

    monkeypatch.setattr(httpx2.AsyncClient, "stream", staticmethod(hosted_response))

    report = asyncio.run(application.test(source_episode, verbose=False))
    check_run = report.check("build_ai_hand_visibility")
    assert check_run.result is not None
    result = check_run.result

    assert len(captured_requests) == 1
    method, url, headers, files, timeout, follow_redirects = captured_requests[0]
    assert method == "POST"
    assert url == ("https://checks.example/v1/checks/build_ai_hand_visibility/versions/3/evaluate")
    assert timeout == 12.0
    assert follow_redirects is False
    assert headers == {
        "Accept": "application/json",
        "User-Agent": f"hflow/{hflow.__version__} (+https://hflow.dev)",
    }
    filename, uploaded_image_bytes, uploaded_image_mime_type = files["observation"]
    assert filename == "observation.jpg"
    assert uploaded_image_bytes.startswith(b"\xff\xd8\xff")
    assert uploaded_image_mime_type == "image/jpeg"
    assert result.measurements == {
        "build_ai/hand_count/raw_response": "2",
        "build_ai/hand_count/requested_model": "hflow-hosted/build_ai_hand_visibility@3",
        "build_ai/hand_count/prediction": 2,
    }
    assert result.tags == []
    assert len(result.observations) == 1
    assert result.observations[0].values["valid"] is True
    assert result.observations[0].values["prediction"] == 2


def test_hosted_unparsed_response_remains_an_evaluation_outcome() -> None:
    outcome = hflow.build_ai_vlm_checks._parse_hosted_check_response(
        hflow.build_ai_vlm_checks.EvaluationTask.ACTIVE_MANIPULATION,
        {
            "outcome": "unparsed",
            "raw_response": "probably",
            "parse_error": 'active manipulation must be "yes" or "no"',
        },
    )

    assert isinstance(outcome, hflow.build_ai_vlm_checks.UnparsedVisionModelOutcome)
    assert outcome.raw_response == "probably"
    assert outcome.parse_error == 'active manipulation must be "yes" or "no"'


@pytest.mark.parametrize("prediction", [True, False, -1, 3, 1.0, "1", None])
def test_hosted_response_refuses_a_prediction_outside_the_check_contract(
    prediction: object,
) -> None:
    with pytest.raises(RuntimeError, match="invalid response"):
        hflow.build_ai_vlm_checks._parse_hosted_check_response(
            hflow.build_ai_vlm_checks.EvaluationTask.HAND_COUNT,
            {
                "outcome": "parsed",
                "prediction": prediction,
                "raw_response": "3",
            },
        )


@pytest.mark.parametrize(
    ("factory", "field", "boundary", "valid"),
    [
        (
            partial(
                hflow.build_ai_vlm_checks.OpenAICompatibleExecution,
                endpoint="https://example.com/v1",
                model="model",
            ),
            "max_tokens",
            0,
            1,
        ),
        (
            partial(
                hflow.build_ai_vlm_checks.OpenAICompatibleExecution,
                endpoint="https://example.com/v1",
                model="model",
            ),
            "max_retries",
            -1,
            0,
        ),
        (hflow.build_ai_vlm_checks.HFlowHostedExecution, "max_retries", -1, 0),
        (hflow.build_ai_vlm_checks.HFlowHostedExecution, "check_version", 0, 1),
        (hflow.build_ai_vlm_checks.HFlowHostedExecution, "request_timeout_seconds", 0, 0.5),
        (hflow.build_ai_vlm_checks.HFlowHostedExecution, "total_timeout_seconds", 0, 0.5),
        (hflow.build_ai_vlm_checks.FrameSampling, "fps", 0, 0.5),
        (hflow.build_ai_vlm_checks.FrameSampling, "start_s", -1, 0),
        (partial(hflow.build_ai_vlm_checks.FrameSampling, start_s=2), "end_s", 2, 3),
    ],
)
def test_numeric_configuration_boundaries(
    factory: Callable[..., Any], field: str, boundary: int, valid: int | float
) -> None:
    for value in [True, False, "1", None, float("nan"), float("inf"), float("-inf"), boundary]:
        if field == "end_s" and value is None:
            continue
        with pytest.raises(ValueError, match=field):
            factory(**{field: value})
    assert getattr(factory(**{field: valid}), field) == valid


@pytest.mark.parametrize("value", [True, False, "1", float("nan"), float("inf"), float("-inf")])
def test_temperature_requires_a_finite_number(value: Any) -> None:
    with pytest.raises(ValueError, match="temperature"):
        hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="https://example.com/v1", model="model", temperature=value
        )


@pytest.mark.parametrize("value", [1, 0, "true", None, 1.0])
def test_skip_black_frames_requires_an_actual_bool(value: Any) -> None:
    # The one guard in this file that _field_guards cannot express: the field
    # wants a bool, so the bool exclusion every other guard makes is inverted
    # here. Deleting it left the whole suite green, hence this case. 1 and 0
    # are the interesting rows, since they compare equal to True and False.
    with pytest.raises(ValueError, match="skip_black_frames must be a bool"):
        hflow.build_ai_vlm_checks.FrameSampling(fps=1.0, skip_black_frames=value)


def test_skip_black_frames_accepts_both_bools() -> None:
    # False is the row that matters: a truthiness check instead of a type
    # check would let it through and a falsy-value guard would refuse it.
    assert (
        hflow.build_ai_vlm_checks.FrameSampling(fps=1.0, skip_black_frames=False).skip_black_frames
        is False
    )
    assert (
        hflow.build_ai_vlm_checks.FrameSampling(fps=1.0, skip_black_frames=True).skip_black_frames
        is True
    )


@pytest.mark.parametrize("value", [None, -1, 0, 0.5])
def test_temperature_accepts_optional_finite_numbers(value: float | None) -> None:
    execution = hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
        endpoint="https://example.com/v1", model="model", temperature=value
    )
    assert execution.temperature == value


@pytest.mark.parametrize(
    "value", [True, False, "1", None, -1, float("nan"), float("inf"), float("-inf")]
)
def test_registration_refuses_invalid_frame_times(tmp_path: Path, value: Any) -> None:
    application = hflow.App("frame-times", data_root=tmp_path, default_checks=())
    with pytest.raises(ValueError, match="frame_time_seconds"):
        hflow.build_ai_vlm_checks.register_hand_visibility(
            application,
            execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(),
            frame_time_seconds=value,
        )


@pytest.mark.parametrize(
    "value", [True, False, -1, 3, 1.0, None, "0", "1", "2", "01", "3", float("nan"), float("inf")]
)
def test_hand_count_response_refuses_invalid_numbers(value: object) -> None:
    with pytest.raises(ValueError, match=r"^hand count must be 0, 1, or 2$"):
        hflow.build_ai_vlm_checks.parse_hand_count_response(json.dumps({"hand_count": value}))


@pytest.mark.parametrize("value", [0, 1, 2])
def test_hand_count_response_accepts_integer_and_text_counts(value: int | str) -> None:
    assert hflow.build_ai_vlm_checks.parse_hand_count_response(
        json.dumps({"hand_count": value})
    ) == int(value)


def test_usage_booleans_are_observations_but_not_numeric_measurements() -> None:
    checks = hflow.build_ai_vlm_checks
    result = checks.model_output_check_result(
        task=checks.EvaluationTask.HAND_COUNT,
        requested_model="model",
        outcome=checks.ParsedVisionModelOutcome(
            raw_response="1",
            response_metadata=checks.ModelResponseMetadata(
                response_model=None,
                usage={
                    "tokens": 2,
                    "cost": 0.5,
                    "cached": True,
                    "empty": False,
                    "label": "text",
                    "missing": None,
                },
            ),
            predicted_value=1,
        ),
        observation_id="frame:0",
        timestamp_ns=0,
    )
    assert {key: value for key, value in result.measurements.items() if "/usage/" in key} == {
        "build_ai/hand_count/usage/tokens": 2,
        "build_ai/hand_count/usage/cost": 0.5,
    }
    assert result.observations[0].values["usage/cached"] is True
    assert result.observations[0].values["usage/empty"] is False


# --- version contract covers every knob that changes result completeness (#404)


def _versions_for(executions: list) -> list:
    versions = []
    for index, execution in enumerate(executions):
        application = hflow.App(
            f"app-{index}-{abs(hash(execution))}",
            data_root=Path(f"/tmp/unused-{index}-{abs(hash(execution))}"),
            default_checks=(),
        )
        hflow.build_ai_vlm_checks.register_hand_visibility(application, execution=execution)
        versions.append(application.checks[0].version)
    return versions


def test_check_version_stable_when_every_covered_field_is_identical(tmp_path: Path) -> None:
    """DoD 3: a configuration identical in every covered field keeps its
    current version, so unchanged methodology never silently invalidates."""
    first_application = hflow.App("first", data_root=tmp_path / "first", default_checks=())
    second_application = hflow.App("second", data_root=tmp_path / "second", default_checks=())
    execution = hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
        endpoint="http://localhost:8000/v1",
        model="model-a",
        temperature=0.5,
        max_tokens=512,
        max_retries=3,
    )
    hflow.build_ai_vlm_checks.register_hand_visibility(first_application, execution=execution)
    hflow.build_ai_vlm_checks.register_hand_visibility(second_application, execution=execution)

    assert first_application.checks[0].version == second_application.checks[0].version


# Golden versions for two fixed configurations, one per execution branch.
#
# The equality test above only proves _check_version is a function: it cannot
# fail unless the same input starts producing two answers. The property DoD 3
# actually claims is that unchanged methodology keeps its identity across
# changes to this module, and only a recorded value can hold that. Adding a
# field to the contract, renaming a key, or reordering nothing at all silently
# re-mints every stored check version; here it fails instead.
#
# Editing these strings is the signal, not the chore. Change them only
# together with a deliberate contract change, and say in the PR why every
# existing Build AI result is being invalidated.
_GOLDEN_OPENAI_CHECK_VERSION = "build-ai-single-frame-v2-d9a739c8f3f88364"
# Re-minted when HFlowHostedExecution gained max_retries: like the OpenAI
# branch's max_retries (#404), it decides which frames produce a result at all.
_GOLDEN_HOSTED_CHECK_VERSION = "build-ai-single-frame-v2-5c2e7b10be83c45a"


def test_check_version_is_pinned_for_a_fixed_openai_configuration(tmp_path: Path) -> None:
    application = hflow.App("golden-openai", data_root=tmp_path, default_checks=())
    hflow.build_ai_vlm_checks.register_hand_visibility(
        application,
        execution=hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="http://localhost:8000/v1",
            model="model-a",
            temperature=0.5,
            max_tokens=512,
            max_retries=3,
        ),
    )

    assert str(application.checks[0].version) == _GOLDEN_OPENAI_CHECK_VERSION


def test_check_version_is_pinned_for_a_fixed_hosted_configuration(tmp_path: Path) -> None:
    application = hflow.App("golden-hosted", data_root=tmp_path, default_checks=())
    hflow.build_ai_vlm_checks.register_hand_visibility(
        application,
        execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(
            check_version=1,
            request_timeout_seconds=30.0,
        ),
    )

    assert str(application.checks[0].version) == _GOLDEN_HOSTED_CHECK_VERSION


def test_check_version_changes_with_max_retries(tmp_path: Path) -> None:
    """max_retries decides whether a transient error becomes a prediction or
    a failed run: retries change which items produce answers at all, so two
    executions differing only in retries must not share a version."""
    first_application = hflow.App("first", data_root=tmp_path / "first", default_checks=())
    second_application = hflow.App("second", data_root=tmp_path / "second", default_checks=())
    hflow.build_ai_vlm_checks.register_hand_visibility(
        first_application,
        execution=hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="http://localhost:8000/v1", model="model-a", max_retries=0
        ),
    )
    hflow.build_ai_vlm_checks.register_hand_visibility(
        second_application,
        execution=hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint="http://localhost:8000/v1", model="model-a", max_retries=5
        ),
    )

    assert first_application.checks[0].version != second_application.checks[0].version


@pytest.mark.parametrize("timeout_field", ["request_timeout_seconds", "total_timeout_seconds"])
def test_check_version_changes_with_timeout(tmp_path: Path, timeout_field: str) -> None:
    """The hosted branch applies the same rule: a timeout decides whether a
    slow-but-valid response is included, so the field belongs in identity."""
    first_application = hflow.App("first", data_root=tmp_path / "first", default_checks=())
    second_application = hflow.App("second", data_root=tmp_path / "second", default_checks=())
    hflow.build_ai_vlm_checks.register_hand_visibility(
        first_application,
        execution=replace(
            hflow.build_ai_vlm_checks.HFlowHostedExecution(check_version=1), **{timeout_field: 1.0}
        ),
    )
    hflow.build_ai_vlm_checks.register_hand_visibility(
        second_application,
        execution=replace(
            hflow.build_ai_vlm_checks.HFlowHostedExecution(check_version=1), **{timeout_field: 60.0}
        ),
    )

    assert first_application.checks[0].version != second_application.checks[0].version


def test_check_version_applies_the_rule_symmetrically_across_branches(
    tmp_path: Path,
) -> None:
    """DoD 4: both branches treat the rule the same way. Each branch must
    change its version when its own completeness knob changes, by the same
    mechanism (the contract), not by an asymmetric special case."""
    openai_versions = _versions_for(
        [
            hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
                endpoint="http://localhost:8000/v1", model="model-a", max_retries=0
            ),
            hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
                endpoint="http://localhost:8000/v1", model="model-a", max_retries=5
            ),
        ]
    )
    hosted_versions = _versions_for(
        [
            hflow.build_ai_vlm_checks.HFlowHostedExecution(
                check_version=1, request_timeout_seconds=1.0
            ),
            hflow.build_ai_vlm_checks.HFlowHostedExecution(
                check_version=1, request_timeout_seconds=60.0
            ),
        ]
    )
    assert openai_versions[0] != openai_versions[1]
    assert hosted_versions[0] != hosted_versions[1]


@pytest.mark.parametrize(
    ("task", "response_text"),
    [
        ("hand-count", '{"hand_count":0,"hand_count":2}'),
        ("active-manipulation", '{"answer":"no","answer":"yes"}'),
        ("hand-count", '{"hand_count":2,"extra":NaN}'),
        ("hand-count", '{"hand_count":2,"extra":1e999}'),
        ("hand-count", '{"hand_count":2,"extra":1}'),
        ("hand-count", " " * (64 * 1024) + "2"),
        ("hand-count", "\ud800"),
        ("active-manipulation", '{"answer":"YES"}'),
    ],
)
def test_model_answers_reject_ambiguous_or_out_of_contract_json(
    task: str, response_text: str
) -> None:
    checks = hflow.build_ai_vlm_checks
    with pytest.raises(ValueError) as captured_error:
        checks.parse_task_response(checks.EvaluationTask(task), response_text)
    assert captured_error.value.__cause__ is None
    assert captured_error.value.__suppress_context__


@pytest.mark.parametrize(
    ("task", "response_text", "expected"),
    [("hand-count", " 2 ", 2), ("active-manipulation", "NO.", "no")],
)
def test_plain_text_response_mode_remains_explicitly_supported(
    task: str, response_text: str, expected: int | str
) -> None:
    checks = hflow.build_ai_vlm_checks
    assert checks.parse_task_response(checks.EvaluationTask(task), response_text) == expected


@pytest.mark.parametrize(
    ("finish_reason", "refusal", "tool_calls", "choice_count", "accepted"),
    [
        ("stop", None, None, 1, True),
        ("length", None, None, 1, False),
        ("content_filter", None, None, 1, False),
        (None, None, None, 1, False),
        ("stop", "cannot answer", None, 1, False),
        ("stop", None, [{"id": "tool"}], 1, False),
        ("stop", None, None, 2, False),
        ("stop", None, None, 0, False),
    ],
)
def test_only_one_complete_nonrefused_completion_produces_a_prediction(
    finish_reason: str | None,
    refusal: str | None,
    tool_calls: list[object] | None,
    choice_count: int,
    accepted: bool,
) -> None:
    from types import SimpleNamespace

    checks = hflow.build_ai_vlm_checks
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content='{"hand_count":2}', refusal=refusal, tool_calls=tool_calls
                ),
            )
        ]
        * choice_count
    )

    async def complete(**_arguments: object) -> object:
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    outcome = asyncio.run(
        checks.evaluate_image_with_model(
            client=client,
            model="model",
            task_definition=checks.load_task_definitions()[checks.EvaluationTask.HAND_COUNT],
            image_data_url="data:image/png;base64,fixture",
            response_format=checks.ResponseFormat.JSON_SCHEMA,
            temperature=None,
            max_tokens=32,
        )
    )
    assert isinstance(outcome, checks.ParsedVisionModelOutcome) is accepted
    if isinstance(outcome, checks.ParsedVisionModelOutcome):
        assert outcome.predicted_value == 2
    else:
        assert "ValidationError" not in outcome.parse_error


@pytest.fixture
def hosted_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    elapsed_seconds = [0.0]
    from types import SimpleNamespace

    from tenacity import AsyncRetrying

    monkeypatch.setattr(
        hflow.build_ai_vlm_checks, "time", SimpleNamespace(monotonic=lambda: elapsed_seconds[0])
    )

    async def advance_clock(seconds: float) -> None:
        elapsed_seconds[0] += seconds

    monkeypatch.setattr(
        hflow.build_ai_vlm_checks, "AsyncRetrying", partial(AsyncRetrying, sleep=advance_clock)
    )
    return elapsed_seconds


def _hosted_success_response() -> httpx2.Response:
    return httpx2.Response(
        200,
        json={"outcome": "parsed", "prediction": 2, "raw_response": "2"},
        request=httpx2.Request("POST", "https://checks.example/evaluate"),
    )


def _evaluate_hosted_hand_count(
    **configuration: Any,
) -> hflow.build_ai_vlm_checks.VisionModelOutcome:
    checks = hflow.build_ai_vlm_checks
    return asyncio.run(
        checks._evaluate_image_with_hflow_hosted_service(
            execution=checks.HFlowHostedExecution(**configuration),
            task=checks.EvaluationTask.HAND_COUNT,
            image_bytes=b"\xff\xd8\xffsynthetic-image",
        )
    )


@pytest.mark.parametrize("status_code", [429, 502, 503, 504, None])
def test_hosted_retry_recovers_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
    hosted_clock: list[float],
    status_code: int | None,
) -> None:
    from contextlib import asynccontextmanager

    responses = iter((status_code, 200))

    @asynccontextmanager
    async def respond(
        *_arguments: object, **_keyword_arguments: object
    ) -> AsyncIterator[httpx2.Response]:
        next_status = next(responses)
        if next_status is None:
            raise httpx2.ConnectError("connection interrupted")
        response = (
            _hosted_success_response()
            if next_status == 200
            else httpx2.Response(
                next_status,
                headers={"Retry-After": "2"},
                request=httpx2.Request("POST", "https://checks.example/evaluate"),
            )
        )
        try:
            yield response
        finally:
            await response.aclose()

    monkeypatch.setattr(httpx2.AsyncClient, "stream", staticmethod(respond))
    outcome = _evaluate_hosted_hand_count()
    assert isinstance(outcome, hflow.build_ai_vlm_checks.ParsedVisionModelOutcome)
    assert outcome.predicted_value == 2
    assert hosted_clock[0] == (1 if status_code is None else 2)


@pytest.mark.parametrize("failure", ["authorization", "malformed", "exhausted", "retry-budget"])
def test_hosted_request_does_not_turn_terminal_failures_into_success(
    monkeypatch: pytest.MonkeyPatch,
    hosted_clock: list[float],
    failure: str,
) -> None:
    from contextlib import asynccontextmanager

    failure_responses = {
        "authorization": [httpx2.Response(401)],
        "malformed": [
            httpx2.Response(
                200,
                content=b'{"outcome":"parsed","prediction":0,"prediction":2,"raw_response":"2"}',
            )
        ],
        "exhausted": [httpx2.Response(503), httpx2.Response(503)],
        "retry-budget": [httpx2.Response(503, headers={"Retry-After": "10"})],
    }
    responses = iter([*failure_responses[failure], _hosted_success_response()])

    @asynccontextmanager
    async def respond(
        *_arguments: object, **_keyword_arguments: object
    ) -> AsyncIterator[httpx2.Response]:
        response = next(responses)
        response.request = httpx2.Request("POST", "https://checks.example/private-input")
        try:
            yield response
        finally:
            await response.aclose()

    monkeypatch.setattr(httpx2.AsyncClient, "stream", staticmethod(respond))
    with pytest.raises(RuntimeError) as captured_error:
        _evaluate_hosted_hand_count(max_retries=1, total_timeout_seconds=5)
    assert "private-input" not in str(captured_error.value)
    assert captured_error.value.__cause__ is None
    assert hosted_clock[0] == (1 if failure == "exhausted" else 0)


def test_hosted_response_that_crosses_the_total_budget_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
    hosted_clock: list[float],
) -> None:
    class SlowHostedResponse(_StubHostedResponse):
        async def aiter_bytes(self) -> AsyncIterator[bytes]:
            yield self._body[:1]
            hosted_clock[0] += 6
            yield self._body[1:]

    monkeypatch.setattr(
        httpx2.AsyncClient,
        "stream",
        lambda *_arguments, **_keyword_arguments: SlowHostedResponse(
            {
                "outcome": "parsed",
                "prediction": 2,
                "raw_response": "2",
            }
        ),
    )
    with pytest.raises(RuntimeError, match="total timeout"):
        _evaluate_hosted_hand_count(total_timeout_seconds=5)


def test_hosted_total_deadline_interrupts_a_stalled_response_body() -> None:
    """A responsive header cannot let an indefinitely stalled body hold an episode."""
    checks = hflow.build_ai_vlm_checks

    async def scenario() -> None:
        connection_closed = asyncio.Event()
        body_started = asyncio.Event()

        async def stall_response(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                content_length = next(
                    int(line.partition(b":")[2])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                await reader.readexactly(content_length)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
                await writer.drain()
                body_started.set()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
                connection_closed.set()

        server = await asyncio.start_server(stall_response, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server, asyncio.timeout(3):
            with pytest.raises(RuntimeError, match="total timeout"):
                await checks._evaluate_image_with_hflow_hosted_service(
                    execution=checks.HFlowHostedExecution(
                        base_url=f"http://127.0.0.1:{port}",
                        total_timeout_seconds=0.3,
                        request_timeout_seconds=2,
                    ),
                    task=checks.EvaluationTask.HAND_COUNT,
                    image_bytes=b"\xff\xd8\xffsynthetic-image",
                )
            assert body_started.is_set()
            await connection_closed.wait()

    asyncio.run(scenario())
