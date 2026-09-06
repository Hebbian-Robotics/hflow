"""Two egocentric vision checks, hand visibility and active manipulation.

Each check is a contract, not a particular model: one egocentric frame in,
one fixed answer out (a hand count of 0, 1, or 2; a yes or no on active
manipulation), recorded as the same measurements, observations, and, when
sampled, intervals whichever way the answer was produced. Two executions
implement the contract and are chosen per registered check:

- :class:`OpenAICompatibleExecution` runs Build AI's published methodology:
  their exact prompts and response schemas (copied below from the evaluation
  release) through an OpenAI-compatible vision model you name. The module's
  name and the ``build_ai_`` check names record that this is where the
  contract and the reference prompts come from.
- :class:`HFlowHostedExecution` runs HFlow's hosted checks: fixed, versioned
  services that answer the same contract. Their prompt, model, and generation
  settings belong to the service and are pinned per hosted check version;
  they are not required to match Build AI's, and a version may diverge from
  the published methodology while keeping the input and output shapes.

Results from the two are comparable in shape, not guaranteed to agree in
value. Every result records which execution answered it: the
``requested_model`` measurement holds the model name for an OpenAI-compatible
run and ``hflow-hosted/<check>@<version>`` for a hosted run, and the
execution's settings enter the check version, so a catalog never mixes the
two under one version.

Build AI's methodology and released evaluation inputs:
https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation
https://huggingface.co/datasets/builddotai/Egocentric-100K-Evaluation
"""

from __future__ import annotations

import base64
import importlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never
from urllib.parse import urlsplit

import httpx2

from hflow._version import __version__
from hflow._video_measurement_toolchain import measure_video_frame_statistics_for_hflow
from hflow._video_measurements import FrameStatisticsSettings
from hflow.episode import Episode
from hflow.fingerprints import step_version_from_contract
from hflow.steps import (
    CheckFunction,
    CheckResult,
    Interval,
    MeasurementValue,
    Observation,
    StepVersion,
)

if TYPE_CHECKING:
    from hflow.app import App

# Copied from Build AI's prompt file at the immutable revision used by the
# reproduction runner:
# https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation/blob/d74b7883c998dd360e3f051830fcc792a83985e6/prompts/hand_count.txt
BUILD_AI_HAND_VISIBILITY_PROMPT = """You are labeling an egocentric first-person image.
Your task is to count how many camera-wearer\N{RIGHT SINGLE QUOTATION MARK}s hands are visually present in the image: 0, 1, or 2.

Rules:
• Only count hands that are directly visible.
• Do not infer hands that are outside the frame or potentially behind objects.
• Ignore hands belonging to other people.
• Any amount of visibility counts (even fingertips).
• Return only one of: 0, 1, 2. No extra words.
"""

# Copied from Build AI's prompt file at the immutable revision used by the
# reproduction runner:
# https://huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation/blob/d74b7883c998dd360e3f051830fcc792a83985e6/prompts/active_manipulation.txt
BUILD_AI_ACTIVE_MANIPULATION_PROMPT = """You are labeling an egocentric first-person image.

Your task is to determine whether the camera-wearer is actively doing active manipulation at this exact moment.

Definition:
"Active Manipulation" means the wearer is visibly using their hands to work on, modify, assemble, process, or handle physical objects, materials, components, or workpieces in pursuit of a specific goal

Rules:
• Do not infer actions that are not visible in the frame.
• If the action is ambiguous or not clearly happening, respond "no."
• Ignore actions performed by other people.
• Respond only with: "yes" or "no."
"""

BUILD_AI_HAND_VISIBILITY_CHECK_NAME = "build_ai_hand_visibility"
BUILD_AI_ACTIVE_MANIPULATION_CHECK_NAME = "build_ai_active_manipulation"
DEFAULT_HFLOW_HOSTED_BASE_URL = "https://api.hflow.dev"

_HFLOW_HOSTED_TRANSPORT_VERSION = 1
_DEFAULT_HFLOW_HOSTED_CHECK_VERSION = 1
_MAX_HFLOW_HOSTED_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_HFLOW_HOSTED_RESPONSE_BYTES = 64 * 1024
_HFLOW_HOSTED_USER_AGENT = f"hflow/{__version__} (+https://hflow.dev)"
# Statuses the hosted service answers for a request that may succeed if simply
# repeated: its own admission control (429) and its upstream's transient
# failures (502/503/504). Anything else describes the request itself.
_RETRYABLE_HOSTED_STATUS_CODES = frozenset({429, 502, 503, 504})
_MAX_HOSTED_RETRY_DELAY_SECONDS = 120.0


def _hosted_retry_delay_seconds(retry_after_header: str | None, attempt: int) -> float:
    """The server's Retry-After when it gave one, else exponential from one second."""
    if retry_after_header is not None:
        try:
            requested_delay = float(retry_after_header)
        except ValueError:
            requested_delay = None
        if requested_delay is not None and math.isfinite(requested_delay) and requested_delay >= 0:
            return min(requested_delay, _MAX_HOSTED_RETRY_DELAY_SECONDS)
    return min(float(2**attempt), _MAX_HOSTED_RETRY_DELAY_SECONDS)


class EvaluationTask(StrEnum):
    HAND_COUNT = "hand-count"
    ACTIVE_MANIPULATION = "active-manipulation"
    BOTH = "both"


class ResponseFormat(StrEnum):
    JSON_SCHEMA = "json-schema"
    JSON_OBJECT = "json-object"
    TEXT = "text"


@dataclass(frozen=True)
class TaskDefinition:
    task: EvaluationTask
    prompt: str
    response_schema: dict[str, object]


@dataclass(frozen=True)
class ModelResponseMetadata:
    response_model: str | None
    usage: dict[str, object]


@dataclass(frozen=True)
class ParsedVisionModelOutcome:
    raw_response: str
    response_metadata: ModelResponseMetadata
    predicted_value: int | str


@dataclass(frozen=True)
class UnparsedVisionModelOutcome:
    raw_response: str
    response_metadata: ModelResponseMetadata
    parse_error: str


VisionModelOutcome = ParsedVisionModelOutcome | UnparsedVisionModelOutcome


@dataclass(frozen=True)
class SkippedBlackFrame:
    """A sampled frame the camera instrument read as black, so no model was asked."""


SampledFrameOutcome = VisionModelOutcome | SkippedBlackFrame

HAND_COUNT_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"hand_count": {"type": "integer"}},
    "required": ["hand_count"],
}
ACTIVE_MANIPULATION_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
}


@dataclass(frozen=True)
class OpenAICompatibleExecution:
    """Answer a check with Build AI's published prompts through a model you name.

    This is the reference methodology: the prompt and response schema are the
    ones Build AI released, and the model is whatever the OpenAI-compatible
    endpoint serves. Changing the model changes the answers but not the
    contract.
    """

    endpoint: str
    model: str
    api_key_environment_variable: str | None = None
    response_format: ResponseFormat = ResponseFormat.JSON_SCHEMA
    temperature: float | None = None
    max_tokens: int = 32
    max_retries: int = 5

    def __post_init__(self) -> None:
        _require_absolute_http_url(self.endpoint, name="endpoint")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.model != self.model.strip():
            raise ValueError("model must not have leading or trailing whitespace")
        if not isinstance(self.response_format, ResponseFormat):
            raise ValueError("response_format must be an hflow.build_ai_vlm_checks.ResponseFormat")
        if (
            self.api_key_environment_variable is not None
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_environment_variable) is None
        ):
            raise ValueError(
                "api_key_environment_variable must be a valid environment variable name"
            )
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.temperature is not None and not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")


@dataclass(frozen=True)
class HFlowHostedExecution:
    """Answer a check with HFlow's hosted service, a fixed and versioned implementation.

    The service owns the prompt, the model, and the generation settings for
    each hosted check version, so a caller configures only where it is, which
    version to ask, and how long to wait. It answers the same contract as the
    Build AI methodology (one frame in, the same answer shape out) but is a
    separate implementation: a hosted version is not required to reproduce
    Build AI's prompts or results, and the two should be compared, not
    assumed equal.
    """

    base_url: str = DEFAULT_HFLOW_HOSTED_BASE_URL
    check_version: int = _DEFAULT_HFLOW_HOSTED_CHECK_VERSION
    request_timeout_seconds: float = 60.0
    # Retries for transient failures only (429, 502, 503, 504, transport
    # errors), each after the server's Retry-After or an exponential delay. A
    # sampled check makes one request per frame, so one gateway timeout must
    # not discard the whole run.
    max_retries: int = 5

    def __post_init__(self) -> None:
        _require_absolute_http_url(self.base_url, name="base_url")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        parsed_base_url = urlsplit(self.base_url)
        if parsed_base_url.query or parsed_base_url.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        if not isinstance(self.check_version, int) or isinstance(self.check_version, bool):
            raise ValueError("check_version must be an integer")
        if self.check_version <= 0:
            raise ValueError("check_version must be greater than zero")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, int | float)
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be finite and greater than zero")


BuildAIExecution = OpenAICompatibleExecution | HFlowHostedExecution


@dataclass(frozen=True)
class FrameSampling:
    """Evaluate every frame at ``fps`` over a window instead of one frame.

    Build AI's methodology is single-frame; sampling applies it repeatedly so
    the per-frame answers become per-frame observations plus intervals where
    the answer was "no hands" or "not manipulating". ``start_s``/``end_s``
    are seconds from the start of the camera stream, and ``end_s=None`` runs
    to its end. Model calls scale with ``fps`` times the window length.

    ``skip_black_frames`` leaves out frames the camera instrument reads as
    black (the same definition ``camera_frame_stats`` records as
    ``black:<camera>`` intervals): a black frame says nothing about hands or
    work, so asking the model would record an absence that is really a
    blackout. Skipped frames are observations with ``skipped="black_frame"``
    and never open or extend an absence interval.
    """

    fps: float = 1.0
    start_s: float = 0.0
    end_s: float | None = None
    skip_black_frames: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.skip_black_frames, bool):
            raise ValueError("skip_black_frames must be a bool")
        if isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and greater than zero")
        if isinstance(self.start_s, bool) or not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("start_s must be finite and non-negative")
        if self.end_s is not None and (
            isinstance(self.end_s, bool)
            or not math.isfinite(self.end_s)
            or self.end_s <= self.start_s
        ):
            raise ValueError("end_s must be finite and greater than start_s")


@dataclass(frozen=True)
class _RegisteredBuildAICheckConfiguration:
    execution: BuildAIExecution
    task_definition: TaskDefinition
    published_prompt: str
    camera: str | None
    frame_time_seconds: float
    sampling: FrameSampling | None = None

    def __post_init__(self) -> None:
        if self.sampling is not None and not isinstance(self.sampling, FrameSampling):
            raise ValueError("sampling must be a FrameSampling or None")
        if not isinstance(self.execution, OpenAICompatibleExecution | HFlowHostedExecution):
            raise ValueError(
                "execution must be an OpenAICompatibleExecution or HFlowHostedExecution"
            )
        if not self.task_definition.prompt.strip():
            raise ValueError("prompt must not be empty")
        if (
            isinstance(self.execution, HFlowHostedExecution)
            and self.task_definition.prompt != self.published_prompt
        ):
            raise ValueError(
                "HFlowHostedExecution uses the hosted check's fixed prompt and does not support "
                "prompt overrides"
            )
        if isinstance(self.frame_time_seconds, bool) or not math.isfinite(self.frame_time_seconds):
            raise ValueError("frame_time_seconds must be finite and non-negative")
        if self.frame_time_seconds < 0:
            raise ValueError("frame_time_seconds must be finite and non-negative")
        if self.camera == "":
            raise ValueError("camera must be None or a non-empty topic name")


def _require_absolute_http_url(value: str, *, name: str) -> None:
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    parsed_url = urlsplit(value)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"{name} must be an absolute http(s) URL, got {value!r}")


def _strip_markdown_code_fence(response_text: str) -> str:
    stripped_response = response_text.strip()
    code_fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return code_fence_match.group(1).strip() if code_fence_match else stripped_response


def _parse_json_or_scalar(response_text: str) -> object:
    stripped_response = _strip_markdown_code_fence(response_text)
    try:
        return json.loads(stripped_response)
    except json.JSONDecodeError:
        return stripped_response


def parse_hand_count_response(response_text: str) -> int:
    """Parse the published structured shape and compatible plain-text answers."""
    parsed_response = _parse_json_or_scalar(response_text)
    if isinstance(parsed_response, dict):
        parsed_response = parsed_response.get("hand_count")
    if isinstance(parsed_response, bool):
        raise ValueError("hand count must be 0, 1, or 2")
    if isinstance(parsed_response, int):
        hand_count = parsed_response
    elif isinstance(parsed_response, str) and re.fullmatch(r"[012]", parsed_response.strip()):
        hand_count = int(parsed_response)
    else:
        raise ValueError("hand count must be 0, 1, or 2")
    if hand_count not in {0, 1, 2}:
        raise ValueError("hand count must be 0, 1, or 2")
    return hand_count


def parse_active_manipulation_response(response_text: str) -> str:
    """Parse the published structured shape and compatible plain-text answers."""
    parsed_response = _parse_json_or_scalar(response_text)
    if isinstance(parsed_response, dict):
        parsed_response = parsed_response.get("answer")
    if not isinstance(parsed_response, str):
        raise ValueError('active manipulation must be "yes" or "no"')
    normalized_answer = parsed_response.strip().lower().rstrip(".")
    if normalized_answer not in {"yes", "no"}:
        raise ValueError('active manipulation must be "yes" or "no"')
    return normalized_answer


def parse_task_response(task: EvaluationTask, response_text: str) -> int | str:
    match task:
        case EvaluationTask.HAND_COUNT:
            return parse_hand_count_response(response_text)
        case EvaluationTask.ACTIVE_MANIPULATION:
            return parse_active_manipulation_response(response_text)
        case EvaluationTask.BOTH:
            raise AssertionError("BOTH is a CLI selection, not an executable task")


def _mime_type_for_image(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("evaluation image is not JPEG, PNG, or WebP")


def image_bytes_data_url(image_bytes: bytes) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{_mime_type_for_image(image_bytes)};base64,{encoded_image}"


def image_file_data_url(image_path: Path) -> str:
    return image_bytes_data_url(image_path.read_bytes())


def load_task_definitions(
    hand_count_prompt_path: Path | None = None,
    active_manipulation_prompt_path: Path | None = None,
) -> dict[EvaluationTask, TaskDefinition]:
    """Return the defaults, optionally replacing either prompt from a file."""
    hand_visibility_prompt = (
        hand_count_prompt_path.read_text()
        if hand_count_prompt_path is not None
        else BUILD_AI_HAND_VISIBILITY_PROMPT
    )
    active_manipulation_prompt = (
        active_manipulation_prompt_path.read_text()
        if active_manipulation_prompt_path is not None
        else BUILD_AI_ACTIVE_MANIPULATION_PROMPT
    )
    return {
        EvaluationTask.HAND_COUNT: TaskDefinition(
            task=EvaluationTask.HAND_COUNT,
            prompt=hand_visibility_prompt,
            response_schema=HAND_COUNT_RESPONSE_SCHEMA,
        ),
        EvaluationTask.ACTIVE_MANIPULATION: TaskDefinition(
            task=EvaluationTask.ACTIVE_MANIPULATION,
            prompt=active_manipulation_prompt,
            response_schema=ACTIVE_MANIPULATION_RESPONSE_SCHEMA,
        ),
    }


def _response_format_payload(
    task_definition: TaskDefinition,
    response_format: ResponseFormat,
) -> object:
    match response_format:
        case ResponseFormat.JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": task_definition.task.value.replace("-", "_"),
                    "schema": task_definition.response_schema,
                },
            }
        case ResponseFormat.JSON_OBJECT:
            return {"type": "json_object"}
        case ResponseFormat.TEXT:
            return None


def _chat_completion_response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("endpoint returned no completion choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for content_part in content:
            if isinstance(content_part, dict) and isinstance(content_part.get("text"), str):
                text_parts.append(content_part["text"])
            elif isinstance(getattr(content_part, "text", None), str):
                text_parts.append(content_part.text)
        if text_parts:
            return "".join(text_parts)
    raise ValueError("endpoint returned no text completion content")


def _response_metadata(response: object) -> ModelResponseMetadata:
    response_model = getattr(response, "model", None)
    parsed_response_model = response_model if isinstance(response_model, str) else None
    parsed_usage: dict[str, object] = {}
    usage = getattr(response, "usage", None)
    if usage is not None and callable(getattr(usage, "model_dump", None)):
        dumped_usage = usage.model_dump(exclude_none=True)
        if isinstance(dumped_usage, dict):
            parsed_usage = dumped_usage
    return ModelResponseMetadata(response_model=parsed_response_model, usage=parsed_usage)


def _task_measurement_prefix(task: EvaluationTask) -> str:
    if task is EvaluationTask.BOTH:
        raise AssertionError("BOTH is a CLI selection, not an executable task")
    return f"build_ai/{task.value.replace('-', '_')}"


def model_output_check_result(
    *,
    task: EvaluationTask,
    requested_model: str,
    outcome: VisionModelOutcome,
    observation_id: str,
    timestamp_ns: int,
) -> CheckResult:
    """Adapt one model outcome to HFlow's complete evidence boundary."""
    measurement_prefix = _task_measurement_prefix(task)
    measurements: dict[str, MeasurementValue] = {
        f"{measurement_prefix}/raw_response": outcome.raw_response,
        f"{measurement_prefix}/requested_model": requested_model,
    }
    if outcome.response_metadata.response_model is not None:
        measurements[f"{measurement_prefix}/response_model"] = (
            outcome.response_metadata.response_model
        )
    for usage_name, usage_value in outcome.response_metadata.usage.items():
        if isinstance(usage_value, int | float) and not isinstance(usage_value, bool):
            measurements[f"{measurement_prefix}/usage/{usage_name}"] = usage_value

    observation_values: dict[str, MeasurementValue] = {
        "task": task.value,
        "raw_response": outcome.raw_response,
        "requested_model": requested_model,
    }
    if outcome.response_metadata.response_model is not None:
        observation_values["response_model"] = outcome.response_metadata.response_model
    for usage_name, usage_value in outcome.response_metadata.usage.items():
        if isinstance(usage_value, int | float | str | bool):
            observation_values[f"usage/{usage_name}"] = usage_value
    match outcome:
        case ParsedVisionModelOutcome(predicted_value=predicted_value):
            measurements[f"{measurement_prefix}/prediction"] = predicted_value
            observation_values["valid"] = True
            observation_values["prediction"] = predicted_value
            tags: list[str] = []
        case UnparsedVisionModelOutcome(parse_error=parse_error):
            measurements[f"{measurement_prefix}/parse_error"] = parse_error
            observation_values["valid"] = False
            observation_values["parse_error"] = parse_error
            tags = [f"{measurement_prefix}/unparsed"]
        case unexpected_outcome:
            assert_never(unexpected_outcome)
    return CheckResult(
        measurements=measurements,
        observations=[
            Observation(
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                values=observation_values,
            )
        ],
        tags=tags,
    )


NANOSECONDS_PER_SECOND = 1_000_000_000


def _absence_interval_kind(task: EvaluationTask) -> str:
    """The interval kind a run of negative answers to ``task`` records."""
    match task:
        case EvaluationTask.HAND_COUNT:
            return "hands_absent"
        case EvaluationTask.ACTIVE_MANIPULATION:
            return "no_manipulation"
        case EvaluationTask.BOTH:
            raise AssertionError("BOTH is a CLI selection, not an executable task")
        case unexpected_task:
            assert_never(unexpected_task)


def _prediction_is_absence(task: EvaluationTask, predicted_value: int | str) -> bool:
    match task:
        case EvaluationTask.HAND_COUNT:
            return predicted_value == 0
        case EvaluationTask.ACTIVE_MANIPULATION:
            return predicted_value == "no"
        case EvaluationTask.BOTH:
            raise AssertionError("BOTH is a CLI selection, not an executable task")
        case unexpected_task:
            assert_never(unexpected_task)


def sampled_model_output_check_result(
    *,
    task: EvaluationTask,
    camera_topic: str,
    sample_period_ns: int,
    frame_outcomes: Sequence[tuple[int, SampledFrameOutcome, str | None]],
) -> CheckResult:
    """Adapt one model outcome per sampled frame to HFlow's evidence boundary.

    Every frame becomes one observation shaped like the single-frame check's.
    Consecutive frames whose parsed answer is the task's negative (``0``
    hands, ``"no"`` manipulation) fold into ``hands_absent:<camera>`` or
    ``no_manipulation:<camera>`` intervals: a run opens at its first frame's
    log time and closes at the first frame after it that answered otherwise,
    or one sample period after the last frame when the run reaches the end.
    A frame whose answer did not parse ends any open run without opening
    one, so an unreadable answer is never counted as absence; a frame skipped
    as black does the same, because a blackout is not evidence about hands.
    """
    measurement_prefix = _task_measurement_prefix(task)
    interval_kind = _absence_interval_kind(task)
    observations: list[Observation] = []
    intervals: list[Interval] = []
    requested_models: set[str] = set()
    open_run_start_ns: int | None = None
    absent_frame_count = 0
    unparsed_frame_count = 0
    last_timestamp_ns: int | None = None

    def close_run(end_ns: int) -> None:
        nonlocal open_run_start_ns
        if open_run_start_ns is not None:
            intervals.append(
                Interval(
                    start_ns=open_run_start_ns,
                    end_ns=end_ns,
                    label=f"{interval_kind}:{camera_topic}",
                )
            )
            open_run_start_ns = None

    skipped_black_frame_count = 0
    for timestamp_ns, outcome, requested_model in frame_outcomes:
        last_timestamp_ns = timestamp_ns
        if isinstance(outcome, SkippedBlackFrame):
            skipped_black_frame_count += 1
            observations.append(
                Observation(
                    observation_id=f"frame:{timestamp_ns}",
                    timestamp_ns=timestamp_ns,
                    values={"task": task.value, "valid": False, "skipped": "black_frame"},
                )
            )
            close_run(timestamp_ns)
            continue
        if requested_model is None:
            raise AssertionError("an evaluated frame always names the model that answered it")
        requested_models.add(requested_model)
        single = model_output_check_result(
            task=task,
            requested_model=requested_model,
            outcome=outcome,
            observation_id=f"frame:{timestamp_ns}",
            timestamp_ns=timestamp_ns,
        )
        observations.extend(single.observations)
        match outcome:
            case ParsedVisionModelOutcome(predicted_value=predicted_value):
                if _prediction_is_absence(task, predicted_value):
                    absent_frame_count += 1
                    if open_run_start_ns is None:
                        open_run_start_ns = timestamp_ns
                else:
                    close_run(timestamp_ns)
            case UnparsedVisionModelOutcome():
                unparsed_frame_count += 1
                close_run(timestamp_ns)
            case unexpected_outcome:
                assert_never(unexpected_outcome)
    if last_timestamp_ns is not None:
        close_run(last_timestamp_ns + sample_period_ns)

    sampled_frame_count = len(frame_outcomes)
    evaluated_frame_count = sampled_frame_count - skipped_black_frame_count
    measurements: dict[str, MeasurementValue] = {
        f"{measurement_prefix}/requested_model": ", ".join(sorted(requested_models)),
        f"{measurement_prefix}/sampled_frame_count": sampled_frame_count,
        f"{measurement_prefix}/unparsed_frame_count": unparsed_frame_count,
        f"{measurement_prefix}/skipped_black_frame_count": skipped_black_frame_count,
        f"{measurement_prefix}/{interval_kind}_frame_count": absent_frame_count,
        f"{measurement_prefix}/{interval_kind}_frame_pct": (
            100.0 * absent_frame_count / evaluated_frame_count if evaluated_frame_count else 0.0
        ),
        f"{measurement_prefix}/{interval_kind}_total_s": sum(
            (interval.end_ns - interval.start_ns) / NANOSECONDS_PER_SECOND for interval in intervals
        ),
    }
    tags = [f"{measurement_prefix}/unparsed"] if unparsed_frame_count else []
    return CheckResult(
        measurements=measurements, observations=observations, intervals=intervals, tags=tags
    )


def evaluate_image_with_model(
    *,
    client: Any,
    model: str,
    task_definition: TaskDefinition,
    image_data_url: str,
    response_format: ResponseFormat,
    temperature: float | None,
    max_tokens: int,
) -> VisionModelOutcome:
    """Run one Build AI image judgment and return its parsed domain outcome."""
    request_parameters: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_definition.prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    response_format_payload = _response_format_payload(task_definition, response_format)
    if response_format_payload is not None:
        request_parameters["response_format"] = response_format_payload
    if temperature is not None:
        request_parameters["temperature"] = temperature

    response = client.chat.completions.create(**request_parameters)
    raw_response = _chat_completion_response_text(response)
    response_metadata = _response_metadata(response)
    try:
        predicted_value = parse_task_response(task_definition.task, raw_response)
    except ValueError as error:
        return UnparsedVisionModelOutcome(
            raw_response=raw_response,
            response_metadata=response_metadata,
            parse_error=str(error),
        )
    return ParsedVisionModelOutcome(
        raw_response=raw_response,
        response_metadata=response_metadata,
        predicted_value=predicted_value,
    )


def _check_name_for_task(task: EvaluationTask) -> str:
    match task:
        case EvaluationTask.HAND_COUNT:
            return BUILD_AI_HAND_VISIBILITY_CHECK_NAME
        case EvaluationTask.ACTIVE_MANIPULATION:
            return BUILD_AI_ACTIVE_MANIPULATION_CHECK_NAME
        case EvaluationTask.BOTH:
            raise AssertionError("BOTH is a CLI selection, not an executable task")


def _hosted_check_endpoint(execution: HFlowHostedExecution, task: EvaluationTask) -> str:
    check_name = _check_name_for_task(task)
    return (
        f"{execution.base_url.rstrip('/')}/v{_HFLOW_HOSTED_TRANSPORT_VERSION}/checks/"
        f"{check_name}/versions/{execution.check_version}/evaluate"
    )


def _hosted_execution_label(execution: HFlowHostedExecution, task: EvaluationTask) -> str:
    return f"hflow-hosted/{_check_name_for_task(task)}@{execution.check_version}"


def _hosted_observation_upload(image_bytes: bytes) -> tuple[str, bytes, str]:
    image_mime_type = _mime_type_for_image(image_bytes)
    image_extension_by_mime_type = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }
    filename = f"observation.{image_extension_by_mime_type[image_mime_type]}"
    return filename, image_bytes, image_mime_type


def _read_bounded_hosted_response(response: httpx2.Response) -> bytes:
    response_body = bytearray()
    for response_chunk in response.iter_bytes():
        if len(response_body) + len(response_chunk) > _MAX_HFLOW_HOSTED_RESPONSE_BYTES:
            raise RuntimeError("HFlow hosted check response exceeds the 64 KiB limit")
        response_body.extend(response_chunk)
    return bytes(response_body)


def _parse_hosted_prediction(task: EvaluationTask, value: object) -> int | str:
    match task:
        case EvaluationTask.HAND_COUNT:
            if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1, 2}:
                raise RuntimeError(
                    "HFlow hosted hand-visibility check returned a parsed prediction "
                    "outside 0, 1, or 2"
                )
            return value
        case EvaluationTask.ACTIVE_MANIPULATION:
            if not isinstance(value, str) or value not in {"yes", "no"}:
                raise RuntimeError(
                    "HFlow hosted active-manipulation check returned a parsed prediction other "
                    'than "yes" or "no"'
                )
            return value
        case EvaluationTask.BOTH:
            raise AssertionError("BOTH is a CLI selection, not an executable task")


def _parse_hosted_check_response(
    task: EvaluationTask,
    response_payload: object,
) -> VisionModelOutcome:
    if not isinstance(response_payload, dict):
        raise RuntimeError("HFlow hosted check returned JSON that is not an object")
    raw_response = response_payload.get("raw_response")
    if not isinstance(raw_response, str):
        raise RuntimeError("HFlow hosted check response is missing string field 'raw_response'")
    response_metadata = ModelResponseMetadata(response_model=None, usage={})
    outcome_kind = response_payload.get("outcome")
    match outcome_kind:
        case "parsed":
            predicted_value = _parse_hosted_prediction(task, response_payload.get("prediction"))
            return ParsedVisionModelOutcome(
                raw_response=raw_response,
                response_metadata=response_metadata,
                predicted_value=predicted_value,
            )
        case "unparsed":
            parse_error = response_payload.get("parse_error")
            if not isinstance(parse_error, str) or not parse_error:
                raise RuntimeError(
                    "HFlow hosted unparsed outcome is missing non-empty string field 'parse_error'"
                )
            return UnparsedVisionModelOutcome(
                raw_response=raw_response,
                response_metadata=response_metadata,
                parse_error=parse_error,
            )
        case _:
            raise RuntimeError(
                "HFlow hosted check response field 'outcome' must be 'parsed' or 'unparsed'"
            )


def _evaluate_image_with_hflow_hosted_service(
    *,
    execution: HFlowHostedExecution,
    task: EvaluationTask,
    image_bytes: bytes,
) -> VisionModelOutcome:
    if len(image_bytes) > _MAX_HFLOW_HOSTED_IMAGE_BYTES:
        raise ValueError("HFlow hosted check observation exceeds the 10 MiB image limit")
    endpoint = _hosted_check_endpoint(execution, task)
    response_bytes: bytes | None = None
    for attempt in range(execution.max_retries + 1):
        is_last_attempt = attempt == execution.max_retries
        try:
            with httpx2.stream(
                "POST",
                endpoint,
                headers={
                    "Accept": "application/json",
                    "User-Agent": _HFLOW_HOSTED_USER_AGENT,
                },
                files={"observation": _hosted_observation_upload(image_bytes)},
                timeout=execution.request_timeout_seconds,
                # An image-bearing API request must never follow a redirect to another origin.
                follow_redirects=False,
            ) as response:
                response.raise_for_status()
                response_bytes = _read_bounded_hosted_response(response)
            break
        except httpx2.HTTPStatusError as error:
            retry_after = error.response.headers.get("Retry-After")
            status_code = error.response.status_code
            if status_code in _RETRYABLE_HOSTED_STATUS_CODES and not is_last_attempt:
                time.sleep(_hosted_retry_delay_seconds(retry_after, attempt))
                continue
            retry_after_suffix = f"; retry after {retry_after}" if retry_after else ""
            raise RuntimeError(
                f"HFlow hosted check request failed with HTTP {status_code}{retry_after_suffix}"
            ) from error
        except httpx2.RequestError as error:
            if not is_last_attempt:
                time.sleep(_hosted_retry_delay_seconds(None, attempt))
                continue
            raise RuntimeError(f"HFlow hosted check endpoint is unreachable: {error}") from error
    if response_bytes is None:
        raise AssertionError("the hosted request loop exits by returning or raising")
    try:
        response_text = response_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("HFlow hosted check returned invalid UTF-8") from error
    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("HFlow hosted check returned malformed JSON") from error
    return _parse_hosted_check_response(task, response_payload)


def _check_version(configuration: _RegisteredBuildAICheckConfiguration) -> StepVersion:
    version_contract: dict[str, object] = {
        "task": configuration.task_definition.task.value,
        "prompt": configuration.task_definition.prompt,
        "response_schema": configuration.task_definition.response_schema,
        "camera": configuration.camera,
        "frame_time_seconds": configuration.frame_time_seconds,
    }
    # Sampling changes which frames produce results, so it is version-worthy;
    # the single-frame contract keeps its shape so existing versions hold.
    contract_name = "build-ai-single-frame-v1"
    if configuration.sampling is not None:
        contract_name = "build-ai-sampled-frames-v1"
        version_contract["sampling"] = {
            "fps": configuration.sampling.fps,
            "start_s": configuration.sampling.start_s,
            "end_s": configuration.sampling.end_s,
            "skip_black_frames": configuration.sampling.skip_black_frames,
        }
    match configuration.execution:
        case OpenAICompatibleExecution() as execution:
            # The contract shape was kept stable during the migration to the
            # explicit execution value so that migration alone did not
            # invalidate otherwise identical results. That constraint applied
            # to that migration only: the contract must include every knob
            # that changes which items produce results, so max_retries is
            # version-worthy even though it only affects request liveness
            # (#404). Adding a field re-mints the version by design.
            version_contract.update(
                {
                    "endpoint": execution.endpoint,
                    "model": execution.model,
                    "response_format": execution.response_format.value,
                    "temperature": execution.temperature,
                    "max_tokens": execution.max_tokens,
                    "max_retries": execution.max_retries,
                }
            )
        case HFlowHostedExecution() as execution:
            # Same rule, applied symmetrically: the timeout decides whether a
            # slow-but-valid response is included in the corpus at all.
            version_contract.update(
                {
                    "execution": "hflow-hosted",
                    "hosted_check_endpoint": _hosted_check_endpoint(
                        execution, configuration.task_definition.task
                    ),
                    "request_timeout_seconds": execution.request_timeout_seconds,
                    "max_retries": execution.max_retries,
                }
            )
        case unexpected_execution:
            assert_never(unexpected_execution)
    return step_version_from_contract(contract_name, version_contract)


def _black_frame_spans_ns(episode: Episode, camera_topic: str) -> tuple[tuple[int, int], ...]:
    """Log-time spans the camera instrument reads as black, at its default settings.

    The same instrument and defaults as ``camera_frame_stats``, so a frame this
    skips is one that check records inside a ``black:<camera>`` interval.
    """
    stamps_ns = episode.channel(camera_topic).timestamps
    if stamps_ns.size == 0:
        return ()
    stream_start_ns = int(stamps_ns[0])
    statistics = measure_video_frame_statistics_for_hflow(
        episode.video(camera_topic), settings=FrameStatisticsSettings()
    )
    return tuple(
        (
            stream_start_ns + int(interval.start_seconds * NANOSECONDS_PER_SECOND),
            stream_start_ns + int(interval.end_seconds * NANOSECONDS_PER_SECOND),
        )
        for interval in statistics.black_intervals
    )


def _register_build_ai_check(
    application: App,
    *,
    configuration: _RegisteredBuildAICheckConfiguration,
) -> CheckFunction:
    client_for_thread = threading.local()

    def model_client(execution: OpenAICompatibleExecution) -> Any:
        api_key = None
        if execution.api_key_environment_variable is not None:
            api_key = os.environ.get(execution.api_key_environment_variable)
            if not api_key:
                raise ValueError(
                    f"{execution.api_key_environment_variable} is required by "
                    f"{_check_name_for_task(configuration.task_definition.task)}"
                )
        cache_key = (execution.endpoint, api_key, execution.max_retries)
        if getattr(client_for_thread, "cache_key", None) != cache_key:
            try:
                openai_module = importlib.import_module("openai")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "the Build AI checks require the optional OpenAI-compatible client; "
                    "install hflow with `uv add 'hflow[openai]'`"
                ) from error
            client_for_thread.client = openai_module.OpenAI(
                api_key=api_key or "not-needed",
                base_url=execution.endpoint,
                max_retries=execution.max_retries,
            )
            client_for_thread.cache_key = cache_key
        return client_for_thread.client

    def evaluate_frame(image_bytes: bytes) -> tuple[VisionModelOutcome, str]:
        match configuration.execution:
            case OpenAICompatibleExecution() as execution:
                outcome = evaluate_image_with_model(
                    client=model_client(execution),
                    model=execution.model,
                    task_definition=configuration.task_definition,
                    image_data_url=image_bytes_data_url(image_bytes),
                    response_format=execution.response_format,
                    temperature=execution.temperature,
                    max_tokens=execution.max_tokens,
                )
                return outcome, execution.model
            case HFlowHostedExecution() as execution:
                outcome = _evaluate_image_with_hflow_hosted_service(
                    execution=execution,
                    task=configuration.task_definition.task,
                    image_bytes=image_bytes,
                )
                return outcome, _hosted_execution_label(
                    execution, configuration.task_definition.task
                )
            case unexpected_execution:
                assert_never(unexpected_execution)

    def evaluate_build_ai_check(episode: Episode) -> CheckResult:
        sampling = configuration.sampling
        if sampling is not None:
            camera_topic = episode.resolve_camera(configuration.camera)
            sampled_frames = episode.frames(
                camera_topic, fps=sampling.fps, start_s=sampling.start_s, end_s=sampling.end_s
            )
            if not sampled_frames:
                raise ValueError(
                    f"episode has no frames in the sampling window for camera {camera_topic!r}"
                )
            black_spans_ns = (
                _black_frame_spans_ns(episode, camera_topic) if sampling.skip_black_frames else ()
            )
            # One request at a time: the hosted quota admits a single request
            # per client, and a sequential loop keeps the observation order
            # equal to the frame order.
            frame_outcomes: list[tuple[int, SampledFrameOutcome, str | None]] = []
            for frame in sampled_frames:
                if any(start <= frame.log_time_ns < end for start, end in black_spans_ns):
                    frame_outcomes.append((frame.log_time_ns, SkippedBlackFrame(), None))
                    continue
                outcome, requested_model = evaluate_frame(frame.path.read_bytes())
                frame_outcomes.append((frame.log_time_ns, outcome, requested_model))
            return sampled_model_output_check_result(
                task=configuration.task_definition.task,
                camera_topic=camera_topic,
                sample_period_ns=int(NANOSECONDS_PER_SECOND / sampling.fps),
                frame_outcomes=frame_outcomes,
            )

        extracted_frames = episode.frames(
            configuration.camera,
            fps=1.0,
            start_s=configuration.frame_time_seconds,
            end_s=configuration.frame_time_seconds + 1.0,
        )
        if not extracted_frames:
            raise ValueError(
                f"episode has no frame at {configuration.frame_time_seconds:g} seconds for "
                f"camera {configuration.camera!r}"
            )
        selected_frame = extracted_frames[0]
        outcome, requested_model = evaluate_frame(selected_frame.path.read_bytes())
        return model_output_check_result(
            task=configuration.task_definition.task,
            requested_model=requested_model,
            outcome=outcome,
            observation_id=f"frame:{selected_frame.log_time_ns}",
            timestamp_ns=selected_frame.log_time_ns,
        )

    return application.check(
        name=_check_name_for_task(configuration.task_definition.task),
        version=_check_version(configuration),
        requires=("vision-model",),
    )(evaluate_build_ai_check)


def register_hand_visibility(
    application: App,
    *,
    execution: BuildAIExecution,
    camera: str | None = None,
    frame_time_seconds: float = 0.0,
    prompt: str = BUILD_AI_HAND_VISIBILITY_PROMPT,
    sampling: FrameSampling | None = None,
) -> CheckFunction:
    """Register the hand-visibility check with one of its two executions.

    The contract is a count of the wearer's visible hands, 0, 1, or 2, per
    frame. ``execution`` decides who answers: Build AI's published prompt
    through a model you name (:class:`OpenAICompatibleExecution`) or HFlow's
    hosted implementation (:class:`HFlowHostedExecution`); see the module
    docstring for how the two relate.

    ``sampling`` evaluates every frame at its rate over its window and records
    ``hands_absent:<camera>`` intervals (see :class:`FrameSampling`); without
    it the check reads the single frame at ``frame_time_seconds``.
    """
    return _register_build_ai_check(
        application,
        configuration=_RegisteredBuildAICheckConfiguration(
            execution=execution,
            task_definition=TaskDefinition(
                task=EvaluationTask.HAND_COUNT,
                prompt=prompt,
                response_schema=HAND_COUNT_RESPONSE_SCHEMA,
            ),
            published_prompt=BUILD_AI_HAND_VISIBILITY_PROMPT,
            camera=camera,
            frame_time_seconds=frame_time_seconds,
            sampling=sampling,
        ),
    )


def register_active_manipulation(
    application: App,
    *,
    execution: BuildAIExecution,
    camera: str | None = None,
    frame_time_seconds: float = 0.0,
    prompt: str = BUILD_AI_ACTIVE_MANIPULATION_PROMPT,
    sampling: FrameSampling | None = None,
) -> CheckFunction:
    """Register the active-manipulation check with one of its two executions.

    The contract is a yes or no on whether the wearer is actively manipulating
    something in the frame. ``execution`` decides who answers: Build AI's
    published prompt through a model you name
    (:class:`OpenAICompatibleExecution`) or HFlow's hosted implementation
    (:class:`HFlowHostedExecution`); see the module docstring for how the two
    relate.

    ``sampling`` evaluates every frame at its rate over its window and records
    ``no_manipulation:<camera>`` intervals (see :class:`FrameSampling`);
    without it the check reads the single frame at ``frame_time_seconds``.
    """
    return _register_build_ai_check(
        application,
        configuration=_RegisteredBuildAICheckConfiguration(
            execution=execution,
            task_definition=TaskDefinition(
                task=EvaluationTask.ACTIVE_MANIPULATION,
                prompt=prompt,
                response_schema=ACTIVE_MANIPULATION_RESPONSE_SCHEMA,
            ),
            published_prompt=BUILD_AI_ACTIVE_MANIPULATION_PROMPT,
            camera=camera,
            frame_time_seconds=frame_time_seconds,
            sampling=sampling,
        ),
    )
