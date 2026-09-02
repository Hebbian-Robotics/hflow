"""Run Build AI's published single-frame checks on HFlow episodes.

Each check may target a different OpenAI-compatible endpoint and model. Omit
its API-key environment-variable setting for an unauthenticated local server.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import hflow

DEFAULT_HFLOW_DATA_ROOT = Path("data/build-ai-evaluation/hflow")
DEFAULT_SAMPLE_EPISODE_PATH = DEFAULT_HFLOW_DATA_ROOT / "single-camera-sample.mcap"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "model-not-configured"


def _optional_float_environment_variable(name: str) -> float | None:
    raw_value = os.environ.get(name)
    return float(raw_value) if raw_value is not None else None


def _optional_prompt(environment_variable_name: str, default_prompt: str) -> str:
    prompt_path = os.environ.get(environment_variable_name)
    return Path(prompt_path).read_text() if prompt_path is not None else default_prompt


shared_endpoint = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL)
shared_model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_NAME)
shared_api_key_environment_variable = os.environ.get("BUILD_AI_API_KEY_ENV")
response_format = hflow.build_ai_vlm_checks.ResponseFormat(
    os.environ.get(
        "BUILD_AI_RESPONSE_FORMAT", hflow.build_ai_vlm_checks.ResponseFormat.JSON_SCHEMA.value
    )
)
temperature = _optional_float_environment_variable("BUILD_AI_TEMPERATURE")
max_tokens = int(os.environ.get("BUILD_AI_MAX_TOKENS", "32"))
max_retries = int(os.environ.get("BUILD_AI_MAX_RETRIES", "5"))
camera = os.environ.get("BUILD_AI_CAMERA")
frame_time_seconds = float(os.environ.get("BUILD_AI_FRAME_TIME_SECONDS", "0"))

app = hflow.App(
    "build-ai-single-frame-example",
    data_root=os.environ.get("HFLOW_DATA_ROOT", str(DEFAULT_HFLOW_DATA_ROOT)),
    default_checks=(),
)

hflow.build_ai_vlm_checks.register_hand_visibility(
    app,
    endpoint=os.environ.get("BUILD_AI_HAND_VISIBILITY_BASE_URL", shared_endpoint),
    model=os.environ.get("BUILD_AI_HAND_VISIBILITY_MODEL", shared_model),
    api_key_environment_variable=os.environ.get(
        "BUILD_AI_HAND_VISIBILITY_API_KEY_ENV",
        shared_api_key_environment_variable,
    ),
    response_format=response_format,
    temperature=temperature,
    max_tokens=max_tokens,
    max_retries=max_retries,
    camera=camera,
    frame_time_seconds=frame_time_seconds,
    prompt=_optional_prompt(
        "BUILD_AI_HAND_VISIBILITY_PROMPT",
        hflow.build_ai_vlm_checks.BUILD_AI_HAND_VISIBILITY_PROMPT,
    ),
)

hflow.build_ai_vlm_checks.register_active_manipulation(
    app,
    endpoint=os.environ.get("BUILD_AI_ACTIVE_MANIPULATION_BASE_URL", shared_endpoint),
    model=os.environ.get("BUILD_AI_ACTIVE_MANIPULATION_MODEL", shared_model),
    api_key_environment_variable=os.environ.get(
        "BUILD_AI_ACTIVE_MANIPULATION_API_KEY_ENV",
        shared_api_key_environment_variable,
    ),
    response_format=response_format,
    temperature=temperature,
    max_tokens=max_tokens,
    max_retries=max_retries,
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
        nargs="?",
        type=Path,
        help="MCAP episode to evaluate; omitted: synthesize a small sample",
    )
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    episode_path = arguments.episode or DEFAULT_SAMPLE_EPISODE_PATH
    if arguments.episode is None and not episode_path.is_file():
        print(f"synthesizing a sample episode at {episode_path} ...")
        hflow.testing.synthesize_episode(
            episode_path,
            hflow.testing.SyntheticEpisodeSpec(
                duration_s=2.0,
                cameras=("wrist_cam",),
                black_segment=None,
                joint_jump_at_s=None,
                timestamp_offset_segment=None,
            ),
        )
    report = app.test(episode_path)
    if report.has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
