from __future__ import annotations

from pathlib import Path

import pytest

import hflow


def test_build_ai_vlm_checks_register_independent_model_contracts(tmp_path: Path) -> None:
    application = hflow.App("model-checks", data_root=tmp_path, default_checks=())

    hand_check = hflow.build_ai_vlm_checks.register_hand_visibility(
        application,
        endpoint="http://hand-model.internal/v1",
        model="hand-model",
    )
    active_manipulation_check = hflow.build_ai_vlm_checks.register_active_manipulation(
        application,
        endpoint="https://hosted.example/v1",
        model="manipulation-model",
        api_key_environment_variable="HOSTED_MODEL_API_KEY",
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
        endpoint="http://localhost:8000/v1",
        model="model-a",
    )
    hflow.build_ai_vlm_checks.register_hand_visibility(
        second_application,
        endpoint="http://localhost:8000/v1",
        model="model-b",
    )

    assert first_application.checks[0].version != second_application.checks[0].version


@pytest.mark.parametrize(
    ("endpoint", "model", "expected_message"),
    [
        ("localhost:8000/v1", "model", "absolute http"),
        ("http://localhost:8000/v1", " ", "model must not be empty"),
    ],
)
def test_build_ai_registration_refuses_invalid_model_configuration(
    tmp_path: Path,
    endpoint: str,
    model: str,
    expected_message: str,
) -> None:
    application = hflow.App("invalid", data_root=tmp_path, default_checks=())

    with pytest.raises(ValueError, match=expected_message):
        hflow.build_ai_vlm_checks.register_hand_visibility(
            application,
            endpoint=endpoint,
            model=model,
        )
