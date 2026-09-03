"""Run Build AI's published single-frame checks on HFlow episodes.

The example uses HFlow's fixed hosted checks by default, without an API key.
Set ``BUILD_AI_EXECUTION=openai-compatible`` to use a local or third-party
OpenAI-compatible vision endpoint instead.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

import hflow

DEFAULT_HFLOW_DATA_ROOT = Path("data/build-ai-evaluation/hflow")
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "model-not-configured"
DEFAULT_EXECUTION_NAME = "hflow-hosted"
OPENAI_COMPATIBLE_EXECUTION_NAME = "openai-compatible"


def _optional_float_environment_variable(name: str, environment: Mapping[str, str]) -> float | None:
    raw_value = environment.get(name)
    return float(raw_value) if raw_value is not None else None


def _optional_prompt(environment_variable_name: str, default_prompt: str) -> str:
    prompt_path = os.environ.get(environment_variable_name)
    return Path(prompt_path).read_text() if prompt_path is not None else default_prompt


def _execution_from_environment(
    check_environment_variable_prefix: str,
    environment: Mapping[str, str],
) -> hflow.build_ai_vlm_checks.BuildAIExecution:
    check_execution_environment_variable = f"{check_environment_variable_prefix}_EXECUTION"
    execution_name = environment.get(
        check_execution_environment_variable,
        environment.get("BUILD_AI_EXECUTION", DEFAULT_EXECUTION_NAME),
    )
    if execution_name == DEFAULT_EXECUTION_NAME:
        return hflow.build_ai_vlm_checks.HFlowHostedExecution()
    if execution_name == OPENAI_COMPATIBLE_EXECUTION_NAME:
        return hflow.build_ai_vlm_checks.OpenAICompatibleExecution(
            endpoint=environment.get(
                f"{check_environment_variable_prefix}_BASE_URL",
                environment.get("OPENAI_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL),
            ),
            model=environment.get(
                f"{check_environment_variable_prefix}_MODEL",
                environment.get("OPENAI_MODEL", DEFAULT_MODEL_NAME),
            ),
            api_key_environment_variable=environment.get(
                f"{check_environment_variable_prefix}_API_KEY_ENV",
                environment.get("BUILD_AI_API_KEY_ENV"),
            ),
            response_format=hflow.build_ai_vlm_checks.ResponseFormat(
                environment.get(
                    "BUILD_AI_RESPONSE_FORMAT",
                    hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA.value,
                )
            ),
            temperature=_optional_float_environment_variable("BUILD_AI_TEMPERATURE", environment),
            max_tokens=int(environment.get("BUILD_AI_MAX_TOKENS", "32")),
            max_retries=int(environment.get("BUILD_AI_MAX_RETRIES", "5")),
        )
    raise ValueError(
        f"{check_execution_environment_variable} (or BUILD_AI_EXECUTION) must be "
        f"{DEFAULT_EXECUTION_NAME!r} or {OPENAI_COMPATIBLE_EXECUTION_NAME!r}, "
        f"got {execution_name!r}"
    )


camera = os.environ.get("BUILD_AI_CAMERA")
frame_time_seconds = float(os.environ.get("BUILD_AI_FRAME_TIME_SECONDS", "0"))

app = hflow.App(
    "build-ai-single-frame-example",
    data_root=os.environ.get("HFLOW_DATA_ROOT", str(DEFAULT_HFLOW_DATA_ROOT)),
    default_checks=(),
)

hand_visibility_execution = _execution_from_environment("BUILD_AI_HAND_VISIBILITY", os.environ)
hflow.build_ai_vlm_checks.register_hand_visibility(
    app,
    execution=hand_visibility_execution,
    camera=camera,
    frame_time_seconds=frame_time_seconds,
    prompt=_optional_prompt(
        "BUILD_AI_HAND_VISIBILITY_PROMPT",
        hflow.build_ai_vlm_checks.BUILD_AI_HAND_VISIBILITY_PROMPT,
    ),
)

active_manipulation_execution = _execution_from_environment(
    "BUILD_AI_ACTIVE_MANIPULATION", os.environ
)
hflow.build_ai_vlm_checks.register_active_manipulation(
    app,
    execution=active_manipulation_execution,
    camera=camera,
    frame_time_seconds=frame_time_seconds,
    prompt=_optional_prompt(
        "BUILD_AI_ACTIVE_MANIPULATION_PROMPT",
        hflow.build_ai_vlm_checks.BUILD_AI_ACTIVE_MANIPULATION_PROMPT,
    ),
)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        type=Path,
        help="MCAP episode containing meaningful egocentric footage to evaluate",
    )
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    report = app.test(arguments.episode)
    if report.has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
