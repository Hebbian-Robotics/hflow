"""Run the Build AI single-frame judgments on HFlow episodes.

Run from the repository root with an MCAP episode path, or omit the path to
create a small synthetic episode under ``data/build-ai-evaluation/hflow``.

The companion ``evaluate.py`` adapter applies the same prompts, response
schemas, and parsers to Build AI's pinned Parquet frames through Inspect AI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, NamedTuple

import hflow

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from examples.build_ai_evaluation.judgment import (  # noqa: E402
    DEFAULT_PROMPTS_DIRECTORY,
    EvaluationTask,
    ResponseFormat,
    TaskDefinition,
    evaluate_image_with_model,
    image_file_data_url,
    load_task_definitions,
)

VISION_ENDPOINT_ALIAS = "vision"
DEFAULT_HFLOW_DATA_ROOT = Path("data/build-ai-evaluation/hflow")
DEFAULT_SAMPLE_EPISODE_PATH = DEFAULT_HFLOW_DATA_ROOT / "single-camera-sample.mcap"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "model-not-configured"


class PipelineConfiguration(NamedTuple):
    model: str
    api_key_environment_variable: str
    allow_missing_api_key: bool
    response_format: ResponseFormat
    temperature: float | None
    max_tokens: int
    max_retries: int
    camera: str | None
    frame_time_seconds: float
    task_definitions: dict[EvaluationTask, TaskDefinition]


def _optional_float_environment_variable(name: str) -> float | None:
    raw_value = os.environ.get(name)
    return float(raw_value) if raw_value is not None else None


def _boolean_environment_variable(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _pipeline_configuration() -> PipelineConfiguration:
    hand_count_prompt_path = Path(
        os.environ.get(
            "BUILD_AI_HAND_COUNT_PROMPT",
            str(DEFAULT_PROMPTS_DIRECTORY / "hand_count.txt"),
        )
    )
    active_manipulation_prompt_path = Path(
        os.environ.get(
            "BUILD_AI_ACTIVE_MANIPULATION_PROMPT",
            str(DEFAULT_PROMPTS_DIRECTORY / "active_manipulation.txt"),
        )
    )
    return PipelineConfiguration(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL_NAME),
        api_key_environment_variable=os.environ.get("BUILD_AI_API_KEY_ENV", "OPENAI_API_KEY"),
        allow_missing_api_key=_boolean_environment_variable("BUILD_AI_ALLOW_MISSING_API_KEY"),
        response_format=ResponseFormat(
            os.environ.get("BUILD_AI_RESPONSE_FORMAT", ResponseFormat.JSON_SCHEMA.value)
        ),
        temperature=_optional_float_environment_variable("BUILD_AI_TEMPERATURE"),
        max_tokens=int(os.environ.get("BUILD_AI_MAX_TOKENS", "32")),
        max_retries=int(os.environ.get("BUILD_AI_MAX_RETRIES", "5")),
        camera=os.environ.get("BUILD_AI_CAMERA"),
        frame_time_seconds=float(os.environ.get("BUILD_AI_FRAME_TIME_SECONDS", "0")),
        task_definitions=load_task_definitions(
            hand_count_prompt_path, active_manipulation_prompt_path
        ),
    )


pipeline_configuration = _pipeline_configuration()
app = hflow.App(
    "build-ai-single-frame-example",
    data_root=os.environ.get("HFLOW_DATA_ROOT", str(DEFAULT_HFLOW_DATA_ROOT)),
    endpoints={
        VISION_ENDPOINT_ALIAS: os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL)
    },
    default_checks=(),
)


def _check_version(task_definition: TaskDefinition) -> str:
    version_contract = {
        "contract": "build-ai-single-frame-v1",
        "task": task_definition.task.value,
        "prompt": task_definition.prompt,
        "response_schema": task_definition.response_schema,
        "model": pipeline_configuration.model,
        "response_format": pipeline_configuration.response_format.value,
        "temperature": pipeline_configuration.temperature,
        "max_tokens": pipeline_configuration.max_tokens,
        "camera": pipeline_configuration.camera,
        "frame_time_seconds": pipeline_configuration.frame_time_seconds,
    }
    serialized_contract = json.dumps(version_contract, sort_keys=True, separators=(",", ":"))
    contract_digest = hashlib.sha256(serialized_contract.encode()).hexdigest()[:16]
    return f"build-ai-single-frame-v1-{contract_digest}"


_client_by_thread = threading.local()


def _client_for_current_thread() -> Any:
    if pipeline_configuration.model == DEFAULT_MODEL_NAME:
        raise ValueError("OPENAI_MODEL is required")
    api_key = os.environ.get(pipeline_configuration.api_key_environment_variable)
    if not api_key and not pipeline_configuration.allow_missing_api_key:
        raise ValueError(
            f"{pipeline_configuration.api_key_environment_variable} is not set; "
            "set BUILD_AI_ALLOW_MISSING_API_KEY=1 only for an unauthenticated endpoint"
        )
    endpoint_url = app.endpoints[VISION_ENDPOINT_ALIAS]
    client_cache_key = (endpoint_url, api_key)
    if getattr(_client_by_thread, "cache_key", None) != client_cache_key:
        try:
            openai_module = importlib.import_module("openai")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "the OpenAI-compatible client is missing; run with "
                "`uv run --project examples/build_ai_evaluation ...`"
            ) from error
        _client_by_thread.client = openai_module.OpenAI(
            api_key=api_key or "not-needed",
            base_url=endpoint_url,
            max_retries=pipeline_configuration.max_retries,
        )
        _client_by_thread.cache_key = client_cache_key
    return _client_by_thread.client


def _selected_frame_data_url(episode: hflow.Episode) -> str:
    frame_time_seconds = pipeline_configuration.frame_time_seconds
    extracted_frames = episode.frames(
        pipeline_configuration.camera,
        fps=1.0,
        start_s=frame_time_seconds,
        end_s=frame_time_seconds + 1.0,
    )
    if not extracted_frames:
        raise ValueError(
            f"episode has no frame at {frame_time_seconds:g} seconds for camera "
            f"{pipeline_configuration.camera!r}"
        )
    return image_file_data_url(extracted_frames[0].path)


def _evaluate_episode_task(episode: hflow.Episode, task: EvaluationTask) -> hflow.CheckResult:
    outcome = evaluate_image_with_model(
        client=_client_for_current_thread(),
        model=pipeline_configuration.model,
        task_definition=pipeline_configuration.task_definitions[task],
        image_data_url=_selected_frame_data_url(episode),
        response_format=pipeline_configuration.response_format,
        temperature=pipeline_configuration.temperature,
        max_tokens=pipeline_configuration.max_tokens,
    )
    return outcome.check_result


@app.check(
    name="build_ai_hand_count",
    uses=VISION_ENDPOINT_ALIAS,
    version=_check_version(pipeline_configuration.task_definitions[EvaluationTask.HAND_COUNT]),
)
def build_ai_hand_count(episode: hflow.Episode) -> hflow.CheckResult:
    """Record Build AI's single-frame hand-count judgment as HFlow evidence."""
    return _evaluate_episode_task(episode, EvaluationTask.HAND_COUNT)


@app.check(
    name="build_ai_active_manipulation",
    uses=VISION_ENDPOINT_ALIAS,
    version=_check_version(
        pipeline_configuration.task_definitions[EvaluationTask.ACTIVE_MANIPULATION]
    ),
)
def build_ai_active_manipulation(episode: hflow.Episode) -> hflow.CheckResult:
    """Record Build AI's active-manipulation judgment as HFlow evidence."""
    return _evaluate_episode_task(episode, EvaluationTask.ACTIVE_MANIPULATION)


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
