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
_GOLDEN_OPENAI_CHECK_VERSION = "build-ai-single-frame-v1-2aed30388241d554"
_GOLDEN_HOSTED_CHECK_VERSION = "build-ai-single-frame-v1-400d7f82abd83534"


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


def test_check_version_changes_with_request_timeout_seconds(tmp_path: Path) -> None:
    """The hosted branch applies the same rule: a timeout decides whether a
    slow-but-valid response is included, so the field belongs in identity."""
    first_application = hflow.App("first", data_root=tmp_path / "first", default_checks=())
    second_application = hflow.App("second", data_root=tmp_path / "second", default_checks=())
    hflow.build_ai_vlm_checks.register_hand_visibility(
        first_application,
        execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(
            check_version=1, request_timeout_seconds=1.0
        ),
    )
    hflow.build_ai_vlm_checks.register_hand_visibility(
        second_application,
        execution=hflow.build_ai_vlm_checks.HFlowHostedExecution(
            check_version=1, request_timeout_seconds=60.0
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
