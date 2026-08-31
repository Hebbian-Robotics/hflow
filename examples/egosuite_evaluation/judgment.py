"""Shared image-only hand-count judgment contract for the EgoSuite example.

The HFlow episode pipeline and Inspect evaluation use the same prompt, response
schema, image encoding, and parser. Inspect owns model execution for the
cross-model evaluation; the direct OpenAI-compatible request function here is
used by the HFlow check.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts") / "hand_count.txt"
HAND_COUNT_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"hand_count": {"type": "integer", "enum": [0, 1, 2]}},
    "required": ["hand_count"],
}


class ResponseFormat(StrEnum):
    JSON_SCHEMA = "json-schema"
    JSON_OBJECT = "json-object"
    TEXT = "text"


@dataclass(frozen=True)
class ModelResponseMetadata:
    """Provider response fields retained for every hand-count outcome."""

    response_model: str | None
    usage: dict[str, object]


@dataclass(frozen=True)
class ParsedHandCountOutcome:
    """A model response successfully parsed as a supported hand count."""

    raw_response: str
    response_metadata: ModelResponseMetadata
    predicted_hand_count: int


@dataclass(frozen=True)
class UnparsedHandCountOutcome:
    """A completed model response outside the supported hand-count vocabulary."""

    raw_response: str
    response_metadata: ModelResponseMetadata
    parse_error: str


HandCountOutcome = ParsedHandCountOutcome | UnparsedHandCountOutcome


def image_file_data_url(image_path: Path) -> str:
    """Encode an HFlow-extracted JPEG for an OpenAI-compatible image request."""

    image_bytes = image_path.read_bytes()
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"evaluation frame is not JPEG: {image_path}")
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


def _strip_markdown_code_fence(response_text: str) -> str:
    stripped_response = response_text.strip()
    code_fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return code_fence_match.group(1).strip() if code_fence_match else stripped_response


def parse_hand_count_response(response_text: str) -> int:
    """Parse the structured shape and compatible scalar responses."""

    stripped_response = _strip_markdown_code_fence(response_text)
    try:
        parsed_response: object = json.loads(stripped_response)
    except json.JSONDecodeError:
        parsed_response = stripped_response
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


def _response_format_payload(response_format: ResponseFormat) -> object:
    match response_format:
        case ResponseFormat.JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "egosuite_hand_count",
                    "schema": HAND_COUNT_RESPONSE_SCHEMA,
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
    normalized_response_model = response_model if isinstance(response_model, str) else None
    usage = getattr(response, "usage", None)
    parsed_usage: dict[str, object] = {}
    if usage is not None and callable(getattr(usage, "model_dump", None)):
        dumped_usage = usage.model_dump(exclude_none=True)
        if isinstance(dumped_usage, dict):
            parsed_usage = dumped_usage
    return ModelResponseMetadata(
        response_model=normalized_response_model,
        usage=parsed_usage,
    )


def evaluate_image_with_model(
    *,
    client: Any,
    model: str,
    prompt: str,
    image_data_url: str,
    response_format: ResponseFormat,
    temperature: float | None,
    max_tokens: int,
) -> HandCountOutcome:
    """Run one hand-count request through an OpenAI-compatible client."""

    request_parameters: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    response_format_payload = _response_format_payload(response_format)
    if response_format_payload is not None:
        request_parameters["response_format"] = response_format_payload
    if temperature is not None:
        request_parameters["temperature"] = temperature

    response = client.chat.completions.create(**request_parameters)
    response_metadata = _response_metadata(response)
    try:
        raw_response = _chat_completion_response_text(response)
    except ValueError as error:
        return UnparsedHandCountOutcome(
            raw_response="",
            response_metadata=response_metadata,
            parse_error=str(error),
        )
    try:
        predicted_hand_count = parse_hand_count_response(raw_response)
    except ValueError as error:
        return UnparsedHandCountOutcome(
            raw_response=raw_response,
            response_metadata=response_metadata,
            parse_error=str(error),
        )
    return ParsedHandCountOutcome(
        raw_response=raw_response,
        response_metadata=response_metadata,
        predicted_hand_count=predicted_hand_count,
    )
