"""Shared Build AI single-frame judgment contract.

The HFlow episode pipeline and the Inspect evaluation use the same prompts,
response schemas, and parsers. The direct OpenAI-compatible request function
in this module is used only by the HFlow checks; Inspect owns model execution
for the published Parquet evaluation.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import hflow

DEFAULT_PROMPTS_DIRECTORY = Path(__file__).with_name("prompts")


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
class VisionCheckOutcome:
    """One model response expressed on HFlow's check-result boundary."""

    check_result: hflow.CheckResult
    raw_response: str
    response_metadata: dict[str, object]
    predicted_value: int | str | None
    parse_error: str | None = None


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


def _strip_markdown_code_fence(response_text: str) -> str:
    stripped_response = response_text.strip()
    code_fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", stripped_response, flags=re.DOTALL | re.IGNORECASE
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
    """Encode an HFlow-extracted frame for an OpenAI-compatible image request."""
    return image_bytes_data_url(image_path.read_bytes())


def load_task_definitions(
    hand_count_prompt_path: Path, active_manipulation_prompt_path: Path
) -> dict[EvaluationTask, TaskDefinition]:
    return {
        EvaluationTask.HAND_COUNT: TaskDefinition(
            task=EvaluationTask.HAND_COUNT,
            prompt=hand_count_prompt_path.read_text(),
            response_schema=HAND_COUNT_RESPONSE_SCHEMA,
        ),
        EvaluationTask.ACTIVE_MANIPULATION: TaskDefinition(
            task=EvaluationTask.ACTIVE_MANIPULATION,
            prompt=active_manipulation_prompt_path.read_text(),
            response_schema=ACTIVE_MANIPULATION_RESPONSE_SCHEMA,
        ),
    }


def _response_format_payload(
    task_definition: TaskDefinition, response_format: ResponseFormat
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


def _response_metadata(response: object) -> dict[str, object]:
    response_metadata: dict[str, object] = {}
    response_model = getattr(response, "model", None)
    if isinstance(response_model, str):
        response_metadata["response_model"] = response_model
    usage = getattr(response, "usage", None)
    if usage is not None and callable(getattr(usage, "model_dump", None)):
        dumped_usage = usage.model_dump(exclude_none=True)
        if isinstance(dumped_usage, dict):
            response_metadata["usage"] = dumped_usage
    return response_metadata


def _task_measurement_prefix(task: EvaluationTask) -> str:
    if task is EvaluationTask.BOTH:
        raise AssertionError("BOTH is a CLI selection, not an executable task")
    return f"build_ai/{task.value.replace('-', '_')}"


def _hflow_check_result(
    *,
    task: EvaluationTask,
    requested_model: str,
    raw_response: str,
    response_metadata: dict[str, object],
    predicted_value: int | str | None,
    parse_error: str | None = None,
) -> hflow.CheckResult:
    measurement_prefix = _task_measurement_prefix(task)
    measurements: dict[str, hflow.MeasurementValue] = {
        f"{measurement_prefix}/raw_response": raw_response,
        f"{measurement_prefix}/requested_model": requested_model,
    }
    if predicted_value is not None:
        measurements[f"{measurement_prefix}/prediction"] = predicted_value
    response_model = response_metadata.get("response_model")
    if isinstance(response_model, str):
        measurements[f"{measurement_prefix}/response_model"] = response_model
    usage = response_metadata.get("usage")
    if isinstance(usage, dict):
        for usage_name, usage_value in usage.items():
            if isinstance(usage_value, int | float) and not isinstance(usage_value, bool):
                measurements[f"{measurement_prefix}/usage/{usage_name}"] = usage_value
    tags = [f"{measurement_prefix}/unparsed"] if parse_error is not None else []
    if parse_error is not None:
        measurements[f"{measurement_prefix}/parse_error"] = parse_error
    return hflow.CheckResult(measurements=measurements, tags=tags)


def evaluate_image_with_model(
    *,
    client: Any,
    model: str,
    task_definition: TaskDefinition,
    image_data_url: str,
    response_format: ResponseFormat,
    temperature: float | None,
    max_tokens: int,
) -> VisionCheckOutcome:
    """Run one Build AI image judgment and return its HFlow check result."""
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
        parse_error = str(error)
        return VisionCheckOutcome(
            check_result=_hflow_check_result(
                task=task_definition.task,
                requested_model=model,
                raw_response=raw_response,
                response_metadata=response_metadata,
                predicted_value=None,
                parse_error=parse_error,
            ),
            raw_response=raw_response,
            response_metadata=response_metadata,
            predicted_value=None,
            parse_error=parse_error,
        )
    return VisionCheckOutcome(
        check_result=_hflow_check_result(
            task=task_definition.task,
            requested_model=model,
            raw_response=raw_response,
            response_metadata=response_metadata,
            predicted_value=predicted_value,
        ),
        raw_response=raw_response,
        response_metadata=response_metadata,
        predicted_value=predicted_value,
    )
