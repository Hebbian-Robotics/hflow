from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import httpx2
import pytest

import hflow
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode


class _StubHostedResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _StubHostedResponse:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self) -> Iterator[bytes]:
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
    ("endpoint", "model", "expected_message"),
    [
        ("localhost:8000/v1", "model", "absolute http"),
        ("http://localhost:8000/v1", " ", "model must not be empty"),
    ],
)
def test_openai_compatible_execution_refuses_invalid_configuration(
    endpoint: str,
    model: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint=endpoint,
            model=model,
        )


@pytest.mark.parametrize("rejected_max_retries", [True, 2.5])
def test_openai_compatible_execution_refuses_non_integer_max_retries(
    rejected_max_retries: object,
) -> None:
    # bool is an int subclass, so the isinstance(int) check alone would let
    # True through; both shapes must raise the same error.
    with pytest.raises(ValueError, match="max_retries must be an integer"):
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

    monkeypatch.setattr(httpx2, "stream", hosted_response)

    report = application.test(source_episode, verbose=False)
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


def test_hosted_response_refuses_a_prediction_outside_the_check_contract() -> None:
    with pytest.raises(RuntimeError, match="outside 0, 1, or 2"):
        hflow.build_ai_vlm_checks._parse_hosted_check_response(
            hflow.build_ai_vlm_checks.EvaluationTask.HAND_COUNT,
            {
                "outcome": "parsed",
                "prediction": 3,
                "raw_response": "3",
            },
        )
