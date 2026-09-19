"""Bounded VLM wire parsing; validation errors never expose response inputs."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

MAX_RESPONSE_BYTES = 64 * 1024
HandCount = Annotated[int, Field(ge=0, le=2)]


class AnswerModel(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class HandCountAnswer(AnswerModel):
    hand_count: HandCount


class ActiveManipulationAnswer(AnswerModel):
    answer: Literal["yes", "no"]


class ProviderResponse(AnswerModel):
    # Provider envelopes may add metadata unrelated to the answer contract.
    model_config = ConfigDict(extra="ignore", from_attributes=True)


class TextPart(ProviderResponse):
    type: Literal["text"]
    text: str


class CompletionMessage(ProviderResponse):
    content: str | list[TextPart] | None
    refusal: str | None = None
    tool_calls: list[object] | None = None
    function_call: object = None


class CompletionChoice(ProviderResponse):
    finish_reason: Literal["stop"]
    message: CompletionMessage


class CompletionResponse(ProviderResponse):
    choices: Annotated[list[CompletionChoice], Field(min_length=1, max_length=1)]


class ParsedHandCountResponse(ProviderResponse):
    outcome: Literal["parsed"]
    prediction: HandCount
    raw_response: str


class ParsedActiveManipulationResponse(ProviderResponse):
    outcome: Literal["parsed"]
    prediction: Literal["yes", "no"]
    raw_response: str


class UnparsedResponse(ProviderResponse):
    outcome: Literal["unparsed"]
    raw_response: str
    parse_error: Annotated[str, Field(min_length=1)]


HAND_COUNT_HOSTED_RESPONSE = TypeAdapter(
    Annotated[ParsedHandCountResponse | UnparsedResponse, Field(discriminator="outcome")]
)
ACTIVE_MANIPULATION_HOSTED_RESPONSE = TypeAdapter(
    Annotated[ParsedActiveManipulationResponse | UnparsedResponse, Field(discriminator="outcome")]
)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate response field")
        result[name] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("nonfinite response number")


def _finite_json_float(value: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError("nonfinite response number")
    return parsed_value


def require_bounded_response(response_text: str | bytes) -> None:
    try:
        response_bytes = (
            response_text.encode("utf-8") if isinstance(response_text, str) else response_text
        )
        if len(response_bytes) > MAX_RESPONSE_BYTES:
            raise ValueError
    except (ValueError, UnicodeError):
        raise ValueError("model response exceeds its text or byte limits") from None


def strict_response_json(response_text: str | bytes) -> object:
    require_bounded_response(response_text)
    # Keep syntax errors distinguishable for the explicitly supported text mode.
    # Duplicate keys and nonfinite numbers must never fall back to text parsing.
    return json.loads(
        response_text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
