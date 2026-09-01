"""Replay Build AI's Egocentric-10K and Egocentric-100K evaluations with Inspect.

The published Parquet files preserve Build AI's selected frames and Gemini
labels. This adapter streams those fixed inputs into Inspect AI while keeping
the OpenAI-compatible endpoint, model, prompts, and generation settings
swappable.

Run from the repository root. See ``docs/how-to/run-build-ai-evaluation.md``
for prerequisites, commands, costs, outputs, and methodology notes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Any, assert_never, cast
from urllib.parse import urlsplit, urlunsplit

import duckdb
from inspect_ai import SampleSource, Task, eval_set
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.model import (
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    ResponseSchema,
)
from inspect_ai.scorer import Score, Scorer, Target, categorical, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import JSONSchema

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from examples.build_ai_evaluation.judgment import (  # noqa: E402
    DEFAULT_PROMPTS_DIRECTORY,
    EvaluationTask,
    ResponseFormat,
    TaskDefinition,
    image_bytes_data_url,
    load_task_definitions,
    parse_hand_count_response,
    parse_task_response,
)

SCHEMA_VERSION = 1
DEFAULT_DATA_DIRECTORY = Path("data/build-ai-evaluation/datasets")
DEFAULT_RUNS_DIRECTORY = Path("data/build-ai-evaluation/runs")
INSPECT_LOGS_DIRECTORY_NAME = "logs"
RUN_METADATA_FILE_NAME = "run.json"
SUMMARY_FILE_NAME = "summary.json"
INSPECT_OPENAI_COMPATIBLE_SERVICE_NAME = "hflow-evaluation"


class DatasetVariant(StrEnum):
    EGOCENTRIC_10K = "10k"
    EGOCENTRIC_100K = "100k"


class SourceSelection(StrEnum):
    BUILD_AI = "build"
    EGO4D = "ego4d"
    EPIC_KITCHENS = "epic-kitchens"
    ALL = "all"


@dataclass(frozen=True)
class DatasetSpecification:
    repository_id: str
    revision: str
    parquet_file_by_source: dict[SourceSelection, str]


@dataclass(frozen=True)
class EvaluationFrame:
    frame_id: str
    source_dataset: str
    image_data_url: str
    expected_hand_count: int
    expected_active_manipulation: str


@dataclass(frozen=True)
class EvaluationConfiguration:
    dataset_variant: DatasetVariant
    selected_sources: tuple[SourceSelection, ...]
    selected_tasks: tuple[EvaluationTask, ...]
    data_directory: Path
    output_directory: Path
    model: str
    base_url: str
    api_key_environment_variable: str
    allow_missing_api_key: bool
    response_format: ResponseFormat
    temperature: float | None
    max_tokens: int
    max_retries: int
    worker_count: int
    row_limit_per_source: int | None
    label: str
    task_definitions: dict[EvaluationTask, TaskDefinition]


DATASET_SPECIFICATIONS = {
    DatasetVariant.EGOCENTRIC_10K: DatasetSpecification(
        repository_id="builddotai/Egocentric-10K-Evaluation",
        revision="d74b7883c998dd360e3f051830fcc792a83985e6",
        parquet_file_by_source={
            SourceSelection.BUILD_AI: "egocentric_10k.parquet",
            SourceSelection.EGO4D: "ego4d.parquet",
            SourceSelection.EPIC_KITCHENS: "epic_kitchens.parquet",
        },
    ),
    DatasetVariant.EGOCENTRIC_100K: DatasetSpecification(
        repository_id="builddotai/Egocentric-100K-Evaluation",
        revision="d0f69a56b0525c1bead80d918dc57ef83dcac899",
        parquet_file_by_source={
            SourceSelection.BUILD_AI: "egocentric_100k.parquet",
            SourceSelection.EGO4D: "ego4d.parquet",
            SourceSelection.EPIC_KITCHENS: "epic_kitchens.parquet",
        },
    ),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sanitize_base_url(base_url: str) -> str:
    """Remove URL userinfo and query values before persisting run metadata."""
    parsed_url = urlsplit(base_url)
    if parsed_url.hostname is None:
        return base_url
    host = parsed_url.hostname
    if parsed_url.port is not None:
        host = f"{host}:{parsed_url.port}"
    return urlunsplit((parsed_url.scheme, host, parsed_url.path, "", ""))


def _normalize_active_manipulation_reference(active_labor: object) -> str:
    normalized_active_labor = str(active_labor).strip().lower()
    normalized_label_by_source_value = {
        "yes": "yes",
        "true": "yes",
        "no": "no",
        "false": "no",
    }
    try:
        return normalized_label_by_source_value[normalized_active_labor]
    except KeyError as error:
        raise ValueError(
            f"active_labor must be yes/no or true/false, got {active_labor!r}"
        ) from error


def _image_bytes_from_parquet_value(image_value: object) -> bytes:
    if isinstance(image_value, bytes):
        return image_value
    if isinstance(image_value, memoryview):
        return image_value.tobytes()
    if isinstance(image_value, dict):
        embedded_bytes = image_value.get("bytes")
        if isinstance(embedded_bytes, bytes):
            return embedded_bytes
        if isinstance(embedded_bytes, memoryview):
            return embedded_bytes.tobytes()
    raise ValueError("Parquet image column does not contain embedded bytes")


def iter_evaluation_frames(
    parquet_path: Path,
    row_limit: int | None = None,
    source_dataset_name: str | None = None,
) -> Iterator[EvaluationFrame]:
    """Stream frames without retaining a multi-gigabyte Parquet file in memory."""
    if row_limit is not None and row_limit <= 0:
        raise ValueError("row limit must be positive")
    limit_clause = f" LIMIT {row_limit}" if row_limit is not None else ""
    connection = duckdb.connect()
    try:
        described_columns = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet_path)]
        ).fetchall()
        has_upstream_frame_id = any(column[0] == "frame_id" for column in described_columns)
        frame_id_selection = "frame_id, " if has_upstream_frame_id else ""
        cursor = connection.execute(
            f"SELECT {frame_id_selection}image, source_dataset, hand_count, active_labor "
            f"FROM read_parquet(?) {limit_clause}",
            [str(parquet_path)],
        )
        source_row_index = 0
        while row := cursor.fetchone():
            if has_upstream_frame_id:
                frame_id, image_value, source_dataset, hand_count, active_labor = row
            else:
                image_value, source_dataset, hand_count, active_labor = row
                frame_id = f"row-{source_row_index:05d}"
            try:
                normalized_active_labor = _normalize_active_manipulation_reference(active_labor)
            except ValueError as error:
                raise ValueError(
                    f"frame {frame_id!s} has invalid active_labor value {active_labor!r}"
                ) from error
            yield EvaluationFrame(
                frame_id=str(frame_id),
                source_dataset=source_dataset_name or str(source_dataset),
                image_data_url=image_bytes_data_url(_image_bytes_from_parquet_value(image_value)),
                expected_hand_count=parse_hand_count_response(str(hand_count)),
                expected_active_manipulation=normalized_active_labor,
            )
            source_row_index += 1
    finally:
        connection.close()


def _expected_value(frame: EvaluationFrame, task: EvaluationTask) -> int | str:
    match task:
        case EvaluationTask.HAND_COUNT:
            return frame.expected_hand_count
        case EvaluationTask.ACTIVE_MANIPULATION:
            return frame.expected_active_manipulation
        case EvaluationTask.BOTH:
            raise AssertionError("BOTH is a CLI selection, not an executable task")


def _inspect_sample(frame: EvaluationFrame, task_definition: TaskDefinition) -> Sample:
    return Sample(
        id=frame.frame_id,
        input=[
            ChatMessageUser(
                content=[
                    ContentText(text=task_definition.prompt),
                    ContentImage(image=frame.image_data_url),
                ]
            )
        ],
        target=str(_expected_value(frame, task_definition.task)),
        metadata={"source_dataset": frame.source_dataset},
    )


class _ParquetSampleSource(SampleSource):
    """Feed Inspect bounded batches while preserving pinned Parquet order."""

    def __init__(
        self,
        *,
        parquet_path: Path,
        source_dataset_name: str,
        task_definition: TaskDefinition,
        row_limit: int | None,
        batch_size: int,
    ) -> None:
        self._frame_iterator = iter_evaluation_frames(
            parquet_path,
            row_limit,
            source_dataset_name=source_dataset_name,
        )
        self._task_definition = task_definition
        self._batch_size = batch_size

    def _next_batch(self) -> list[Sample]:
        frames = list(islice(self._frame_iterator, self._batch_size))
        return [_inspect_sample(frame, self._task_definition) for frame in frames]

    def initial_samples(self) -> list[Sample]:
        return self._next_batch()

    async def next_samples(self) -> list[Sample] | None:
        samples = self._next_batch()
        return samples or None


def _score_parsed_response(
    state: TaskState,
    target: Target,
    task: EvaluationTask,
) -> Score | None:
    raw_response = state.output.completion
    try:
        predicted_value = parse_task_response(task, raw_response)
    except ValueError:
        return None
    return Score(
        value={
            "prediction": str(predicted_value),
            "agreement": str(predicted_value) == target.text,
        },
        answer=str(predicted_value),
    )


@scorer(
    metrics={"prediction": categorical(["0", "1", "2"]), "agreement": [mean()]},
    name="build_ai_hand_count",
)
def build_ai_hand_count_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score | None:
        return _score_parsed_response(state, target, EvaluationTask.HAND_COUNT)

    return score


@scorer(
    metrics={"prediction": categorical(["yes", "no"]), "agreement": [mean()]},
    name="build_ai_active_manipulation",
)
def build_ai_active_manipulation_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score | None:
        return _score_parsed_response(state, target, EvaluationTask.ACTIVE_MANIPULATION)

    return score


def _inspect_scorer(task: EvaluationTask) -> Scorer:
    match task:
        case EvaluationTask.HAND_COUNT:
            return build_ai_hand_count_scorer()
        case EvaluationTask.ACTIVE_MANIPULATION:
            return build_ai_active_manipulation_scorer()
        case EvaluationTask.BOTH:
            raise AssertionError("BOTH is a CLI selection, not an executable task")


def _inspect_response_schema(
    task_definition: TaskDefinition, response_format: ResponseFormat
) -> ResponseSchema | None:
    if response_format is not ResponseFormat.JSON_SCHEMA:
        return None
    return ResponseSchema(
        name=task_definition.task.value.replace("-", "_"),
        json_schema=JSONSchema.model_validate(task_definition.response_schema),
    )


def _inspect_generate_config(
    configuration: EvaluationConfiguration, task_definition: TaskDefinition
) -> GenerateConfig:
    extra_body: dict[str, Any] | None = None
    if configuration.response_format is ResponseFormat.JSON_OBJECT:
        extra_body = {"response_format": {"type": "json_object"}}
    return GenerateConfig(
        max_retries=configuration.max_retries,
        max_tokens=configuration.max_tokens,
        temperature=configuration.temperature,
        response_schema=_inspect_response_schema(task_definition, configuration.response_format),
        extra_body=extra_body,
    )


def _selected_sources(
    raw_sources: Sequence[SourceSelection] | None,
) -> tuple[SourceSelection, ...]:
    if not raw_sources or SourceSelection.ALL in raw_sources:
        return (
            SourceSelection.BUILD_AI,
            SourceSelection.EGO4D,
            SourceSelection.EPIC_KITCHENS,
        )
    return tuple(dict.fromkeys(raw_sources))


def _selected_tasks(raw_tasks: Sequence[EvaluationTask] | None) -> tuple[EvaluationTask, ...]:
    if not raw_tasks or EvaluationTask.BOTH in raw_tasks:
        return (EvaluationTask.HAND_COUNT, EvaluationTask.ACTIVE_MANIPULATION)
    return tuple(dict.fromkeys(raw_tasks))


def _download_dataset_files(configuration: EvaluationConfiguration) -> None:
    specification = DATASET_SPECIFICATIONS[configuration.dataset_variant]
    variant_directory = configuration.data_directory / configuration.dataset_variant.value
    variant_directory.mkdir(parents=True, exist_ok=True)
    missing_file_names = [
        specification.parquet_file_by_source[source]
        for source in configuration.selected_sources
        if not (variant_directory / specification.parquet_file_by_source[source]).is_file()
    ]
    if not missing_file_names:
        return
    command = [
        "hf",
        "download",
        specification.repository_id,
        *missing_file_names,
        "--type",
        "dataset",
        "--revision",
        specification.revision,
        "--local-dir",
        str(variant_directory),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("`hf` is required for --download but was not found on PATH") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"failed to download {specification.repository_id} at {specification.revision}"
        ) from error


def _required_parquet_paths(
    configuration: EvaluationConfiguration,
) -> list[tuple[SourceSelection, Path]]:
    specification = DATASET_SPECIFICATIONS[configuration.dataset_variant]
    variant_directory = configuration.data_directory / configuration.dataset_variant.value
    selected_paths = [
        (source, variant_directory / specification.parquet_file_by_source[source])
        for source in configuration.selected_sources
    ]
    missing_paths = [path for _source, path in selected_paths if not path.is_file()]
    if missing_paths:
        rendered_paths = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"missing pinned evaluation file(s):\n{rendered_paths}\n"
            "rerun with --download or place the files in that directory"
        )
    return selected_paths


def _required_run_metadata_string(
    document: Mapping[str, object], field_name: str, record_context: str
) -> str:
    field_value = document.get(field_name)
    if not isinstance(field_value, str) or not field_value:
        raise ValueError(f"{record_context} field {field_name!r} must be a string")
    return field_value


def _required_prompt_digests(document: Mapping[str, object], record_context: str) -> dict[str, str]:
    raw_prompts = document.get("prompts")
    if not isinstance(raw_prompts, dict):
        raise ValueError(f"{record_context} field 'prompts' must be a JSON object")
    digests: dict[str, str] = {}
    for task_key, entry in raw_prompts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(
                f"{record_context} prompts entry {task_key!r} must carry a string 'sha256'"
            )
        digests[str(task_key)] = entry["sha256"]
    return digests


@dataclass(frozen=True)
class BuildAIRunMetadata:
    """Saved run.json metadata, parsed once at the load boundary.

    ``document`` is the full persisted schema; the other fields are the
    consumed subset, typed so task creation and summary generation never
    index back into the JSON shape.
    """

    label: str
    fingerprint: str
    dataset_variant: str
    model: str
    prompt_sha256s: dict[str, str]
    document: Mapping[str, object]

    def to_json_dict(self) -> dict[str, object]:
        return dict(self.document)

    @classmethod
    def from_json_file(cls, metadata_path: Path) -> BuildAIRunMetadata:
        try:
            document = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read run metadata {metadata_path}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object")
        context = str(metadata_path)
        return cls(
            label=_required_run_metadata_string(document, "label", context),
            fingerprint=_required_run_metadata_string(document, "fingerprint", context),
            dataset_variant=_required_run_metadata_string(document, "dataset_variant", context),
            model=_required_run_metadata_string(document, "model", context),
            prompt_sha256s=_required_prompt_digests(document, context),
            document=document,
        )


def _run_metadata_document(configuration: EvaluationConfiguration) -> dict[str, object]:
    specification = DATASET_SPECIFICATIONS[configuration.dataset_variant]
    prompt_metadata = {
        task.value: {
            "sha256": _sha256_text(definition.prompt),
            "text": definition.prompt,
            "response_schema": definition.response_schema,
        }
        for task, definition in configuration.task_definitions.items()
        if task in configuration.selected_tasks
    }
    result_contract = {
        "adapter_schema_version": SCHEMA_VERSION,
        "inspect_ai_version": importlib.metadata.version("inspect-ai"),
        "dataset_variant": configuration.dataset_variant.value,
        "dataset_repository": specification.repository_id,
        "dataset_revision": specification.revision,
        "sources": [source.value for source in configuration.selected_sources],
        "tasks": [task.value for task in configuration.selected_tasks],
        "model": configuration.model,
        "base_url": _sanitize_base_url(configuration.base_url),
        "response_format": configuration.response_format.value,
        "temperature": configuration.temperature,
        "max_tokens": configuration.max_tokens,
        "row_limit_per_source": configuration.row_limit_per_source,
        "prompts": prompt_metadata,
    }
    serialized_contract = json.dumps(result_contract, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": SCHEMA_VERSION,
        "label": configuration.label,
        "fingerprint": _sha256_text(serialized_contract),
        **result_contract,
        "api_key_environment_variable": configuration.api_key_environment_variable,
        "worker_count": configuration.worker_count,
        "max_retries": configuration.max_retries,
    }


def _run_metadata(configuration: EvaluationConfiguration) -> BuildAIRunMetadata:
    document = _run_metadata_document(configuration)
    return BuildAIRunMetadata(
        label=configuration.label,
        fingerprint=str(document["fingerprint"]),
        dataset_variant=configuration.dataset_variant.value,
        model=configuration.model,
        prompt_sha256s={
            task.value: _sha256_text(definition.prompt)
            for task, definition in configuration.task_definitions.items()
            if task in configuration.selected_tasks
        },
        document=document,
    )


def _write_json_atomically(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _prepare_output_directory(configuration: EvaluationConfiguration) -> BuildAIRunMetadata:
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = configuration.output_directory / RUN_METADATA_FILE_NAME
    current_metadata = _run_metadata(configuration)
    if metadata_path.is_file():
        existing_metadata = BuildAIRunMetadata.from_json_file(metadata_path)
        if existing_metadata.fingerprint != current_metadata.fingerprint:
            raise ValueError(
                f"{metadata_path} describes a different experiment; choose another --output"
            )
        return existing_metadata
    _write_json_atomically(metadata_path, current_metadata.to_json_dict())
    return current_metadata


def _create_inspect_tasks(
    configuration: EvaluationConfiguration,
    selected_parquet_paths: Sequence[tuple[SourceSelection, Path]],
    run_metadata: BuildAIRunMetadata,
) -> list[Task]:
    fingerprint = run_metadata.fingerprint
    sample_batch_size = max(configuration.worker_count * 2, 1)
    tasks: list[Task] = []
    for source, parquet_path in selected_parquet_paths:
        for task in configuration.selected_tasks:
            task_definition = configuration.task_definitions[task]
            task_name = "_".join(
                (
                    "build_ai",
                    configuration.dataset_variant.value,
                    source.value.replace("-", "_"),
                    task.value.replace("-", "_"),
                )
            )
            tasks.append(
                Task(
                    name=task_name,
                    version=fingerprint[:16],
                    dataset=_ParquetSampleSource(
                        parquet_path=parquet_path,
                        source_dataset_name=parquet_path.stem,
                        task_definition=task_definition,
                        row_limit=configuration.row_limit_per_source,
                        batch_size=sample_batch_size,
                    ),
                    scorer=_inspect_scorer(task),
                    config=_inspect_generate_config(configuration, task_definition),
                    metadata={
                        "dataset_variant": configuration.dataset_variant.value,
                        "source": source.value,
                        "source_dataset": parquet_path.stem,
                        "evaluation_task": task.value,
                        "requested_model": configuration.model,
                        "prompt_sha256": _sha256_text(task_definition.prompt),
                        "run_fingerprint": fingerprint,
                    },
                )
            )
    return tasks


def _result_key(result: EvaluatedSample) -> tuple[str, str, str]:
    return (result.source_dataset, result.frame_id, result.task.value)


def summarize_results(
    run_metadata: BuildAIRunMetadata,
    results: Sequence[EvaluatedSample],
) -> dict[str, object]:
    """Compute prevalence and reference agreement without counting failures negative."""
    latest_results = {_result_key(result): result for result in results}
    source_names = sorted({result.source_dataset for result in latest_results.values()})
    summaries_by_source: dict[str, object] = {}
    for source_name in source_names:
        task_summaries: dict[str, object] = {}
        for task in (EvaluationTask.HAND_COUNT, EvaluationTask.ACTIVE_MANIPULATION):
            task_results = [
                result
                for result in latest_results.values()
                if result.source_dataset == source_name and result.task == task
            ]
            valid_pairs: list[tuple[int | str, int | str]] = []
            invalid_count = 0
            error_count = 0
            for result in task_results:
                match result.outcome:
                    case SuccessfulSampleOutcome(predicted_value=predicted_value):
                        valid_pairs.append((result.expected_value, predicted_value))
                    case InvalidResponseSampleOutcome():
                        invalid_count += 1
                    case ExecutionErrorSampleOutcome():
                        error_count += 1
                    case _:
                        assert_never(result.outcome)
            predicted_value_counts = Counter(
                str(predicted_value) for _, predicted_value in valid_pairs
            )
            reference_value_counts = Counter(str(result.expected_value) for result in task_results)
            agreement_count = sum(
                expected_value == predicted_value for expected_value, predicted_value in valid_pairs
            )
            valid_count = len(valid_pairs)
            numeric_usage_totals: Counter[str] = Counter()
            for result in task_results:
                usage = result.outcome.response_metadata.usage
                if usage is not None:
                    numeric_usage_totals.update(
                        {
                            str(name): float(value)
                            for name, value in usage.items()
                            if isinstance(value, int | float) and not isinstance(value, bool)
                        }
                    )
            latency_values = [
                float(result.outcome.response_metadata.latency_seconds)
                for result in task_results
                if result.outcome.response_metadata.latency_seconds is not None
            ]
            task_summaries[task.value] = {
                "attempted_count": len(task_results),
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "error_count": error_count,
                "predicted_value_counts": dict(sorted(predicted_value_counts.items())),
                "predicted_value_fractions": {
                    value: count / valid_count
                    for value, count in sorted(predicted_value_counts.items())
                }
                if valid_count
                else {},
                "reference_value_counts": dict(sorted(reference_value_counts.items())),
                "reference_value_fractions": {
                    value: count / reference_value_counts.total()
                    for value, count in sorted(reference_value_counts.items())
                }
                if reference_value_counts
                else {},
                "agreement_count": agreement_count,
                "agreement_fraction": agreement_count / valid_count if valid_count else None,
                "average_latency_seconds": (
                    sum(latency_values) / len(latency_values) if latency_values else None
                ),
                "usage_totals": dict(sorted(numeric_usage_totals.items())),
            }
        summaries_by_source[source_name] = task_summaries
    return {
        "schema_version": SCHEMA_VERSION,
        "label": run_metadata.label,
        "fingerprint": run_metadata.fingerprint,
        "dataset_variant": run_metadata.dataset_variant,
        "model": run_metadata.model,
        "prompts": {
            task: {"sha256": sha256} for task, sha256 in run_metadata.prompt_sha256s.items()
        },
        "sources": summaries_by_source,
    }


def _sample_expected_value(sample: EvalSample, task: EvaluationTask) -> int | str:
    raw_target = sample.target[0] if isinstance(sample.target, list) else sample.target
    return parse_task_response(task, raw_target)


@dataclass(frozen=True)
class SampleResponseMetadata:
    """Provider response fields retained for every sample outcome variant."""

    response_model: str | None
    latency_seconds: float | None
    usage: Mapping[str, object] | None


@dataclass(frozen=True)
class SuccessfulSampleOutcome:
    """A completed response parsed into the task's result vocabulary."""

    raw_response: str
    response_metadata: SampleResponseMetadata
    predicted_value: int | str


@dataclass(frozen=True)
class InvalidResponseSampleOutcome:
    """A completed response outside the task's result vocabulary."""

    raw_response: str
    response_metadata: SampleResponseMetadata
    parse_error: str


@dataclass(frozen=True)
class ExecutionErrorSampleOutcome:
    """Inspect reported an execution failure instead of a completed response."""

    raw_response: str
    response_metadata: SampleResponseMetadata
    error: str


SampleOutcome = SuccessfulSampleOutcome | InvalidResponseSampleOutcome | ExecutionErrorSampleOutcome


@dataclass(frozen=True)
class EvaluatedSample:
    """One evaluated Inspect sample: identification plus exactly one outcome variant."""

    source_dataset: str
    frame_id: str
    task: EvaluationTask
    expected_value: int | str
    outcome: SampleOutcome


def _sample_result(log: EvalLog, sample: EvalSample) -> EvaluatedSample:
    log_metadata = log.eval.metadata or {}
    task = EvaluationTask(str(log_metadata["evaluation_task"]))
    response_metadata = SampleResponseMetadata(
        response_model=sample.output.model or None,
        latency_seconds=sample.output.time,
        usage=(
            sample.output.usage.model_dump(exclude_none=True)
            if sample.output.usage is not None
            else None
        ),
    )
    if sample.error is not None or sample.output.error is not None:
        sample_error = sample.error
        error_message = sample_error.message if sample_error is not None else sample.output.error
        outcome: SampleOutcome = ExecutionErrorSampleOutcome(
            raw_response=sample.output.completion,
            response_metadata=response_metadata,
            error=str(error_message)[:1000],
        )
    else:
        try:
            predicted_value = parse_task_response(task, sample.output.completion)
        except ValueError as error:
            outcome = InvalidResponseSampleOutcome(
                raw_response=sample.output.completion,
                response_metadata=response_metadata,
                parse_error=str(error),
            )
        else:
            outcome = SuccessfulSampleOutcome(
                raw_response=sample.output.completion,
                response_metadata=response_metadata,
                predicted_value=predicted_value,
            )
    return EvaluatedSample(
        source_dataset=str(log_metadata["source_dataset"]),
        frame_id=str(sample.id),
        task=task,
        expected_value=_sample_expected_value(sample, task),
        outcome=outcome,
    )


def _results_from_inspect_logs(
    log_headers: Sequence[EvalLog],
) -> tuple[list[EvaluatedSample], list[str]]:
    results: list[EvaluatedSample] = []
    log_locations: list[str] = []
    for log_header in log_headers:
        log_locations.append(log_header.location)
        completed_log = read_eval_log(log_header.location)
        results.extend(
            _sample_result(completed_log, sample) for sample in completed_log.samples or []
        )
    return results, log_locations


def _fraction(summary: dict[str, object], value: str) -> float | None:
    fractions = summary.get("predicted_value_fractions", {})
    if not isinstance(fractions, dict):
        return None
    fraction = fractions.get(value)
    if isinstance(fraction, int | float):
        return float(fraction)
    valid_count = summary.get("valid_count")
    return 0.0 if isinstance(valid_count, int) and valid_count > 0 else None


def _format_percentage(value: float | None) -> str:
    return f"{100.0 * value:.2f}%" if value is not None else "-"


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _print_summary(summary: dict[str, object]) -> None:
    print(f"\nBuild AI evaluation: {summary['label']} ({summary['dataset_variant']})")
    print(
        "| source | hand n | 0 hands | 1+ hands | 2 hands | active n | "
        "active yes | hand agreement | active agreement |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    sources = cast(dict[str, dict[str, dict[str, object]]], summary["sources"])
    for source_name, task_summaries in sources.items():
        hand_summary = task_summaries[EvaluationTask.HAND_COUNT.value]
        active_summary = task_summaries[EvaluationTask.ACTIVE_MANIPULATION.value]
        zero_hand_fraction = _fraction(hand_summary, "0")
        two_hand_fraction = _fraction(hand_summary, "2")
        one_or_more_hand_fraction = (
            1.0 - zero_hand_fraction if zero_hand_fraction is not None else None
        )
        print(
            f"| {source_name} | {hand_summary['valid_count']} "
            f"| {_format_percentage(zero_hand_fraction)} "
            f"| {_format_percentage(one_or_more_hand_fraction)} "
            f"| {_format_percentage(two_hand_fraction)} "
            f"| {active_summary['valid_count']} "
            f"| {_format_percentage(_fraction(active_summary, 'yes'))} "
            f"| {_format_percentage(_optional_float(hand_summary['agreement_fraction']))} "
            f"| {_format_percentage(_optional_float(active_summary['agreement_fraction']))} |"
        )


def _inspect_model_name(requested_model: str) -> str:
    return f"openai-api/{INSPECT_OPENAI_COMPATIBLE_SERVICE_NAME}/{requested_model.lstrip('/')}"


def run_evaluation(configuration: EvaluationConfiguration, *, download: bool) -> dict[str, object]:
    api_key = os.environ.get(configuration.api_key_environment_variable)
    if not api_key and not configuration.allow_missing_api_key:
        raise ValueError(
            f"{configuration.api_key_environment_variable} is not set; use --allow-missing-api-key "
            "only for an endpoint that does not authenticate"
        )
    if download:
        _download_dataset_files(configuration)
    selected_parquet_paths = _required_parquet_paths(configuration)
    run_metadata = _prepare_output_directory(configuration)
    inspect_logs_directory = configuration.output_directory / INSPECT_LOGS_DIRECTORY_NAME
    inspect_logs_directory.mkdir(parents=True, exist_ok=True)
    inspect_tasks = _create_inspect_tasks(
        configuration,
        selected_parquet_paths,
        run_metadata,
    )

    inserted_placeholder_api_key = api_key is None
    if inserted_placeholder_api_key:
        os.environ[configuration.api_key_environment_variable] = "not-needed"
    try:
        all_tasks_succeeded, log_headers = eval_set(
            inspect_tasks,
            log_dir=str(inspect_logs_directory),
            model=_inspect_model_name(configuration.model),
            model_base_url=configuration.base_url,
            model_args={"api_key_var": configuration.api_key_environment_variable},
            metadata={
                "run_fingerprint": run_metadata.fingerprint,
                "label": configuration.label,
            },
            max_connections=configuration.worker_count,
            max_samples=configuration.worker_count,
            max_tasks=1,
            fail_on_error=False,
            retry_on_error=0,
            retry_attempts=1,
            log_format="eval",
            log_images=False,
            log_model_api=False,
            log_dir_allow_dirty=True,
        )
    finally:
        if inserted_placeholder_api_key:
            os.environ.pop(configuration.api_key_environment_variable, None)

    results, log_locations = _results_from_inspect_logs(log_headers)
    summary = summarize_results(run_metadata, results)
    summary["inspect_logs"] = log_locations
    summary_path = configuration.output_directory / SUMMARY_FILE_NAME
    _write_json_atomically(summary_path, summary)
    _print_summary(summary)
    print(f"\nInspect logs: {inspect_logs_directory}")
    print(f"summary: {summary_path}")
    if not all_tasks_succeeded:
        raise RuntimeError("one or more Inspect tasks failed; inspect the logs for details")
    return summary


def compare_summaries(summary_paths: Sequence[Path]) -> None:
    summaries = []
    for path in summary_paths:
        content = path.read_text()
        try:
            summaries.append(json.loads(content))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}: {error}") from error
    print(
        "| run | dataset | source | hand n | 1+ hands | 2 hands | active n | "
        "active yes | hand agreement | active agreement |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        for source_name, task_summaries in summary["sources"].items():
            hand_summary = task_summaries[EvaluationTask.HAND_COUNT.value]
            active_summary = task_summaries[EvaluationTask.ACTIVE_MANIPULATION.value]
            zero_hand_fraction = _fraction(hand_summary, "0")
            one_or_more_hand_fraction = (
                1.0 - zero_hand_fraction if zero_hand_fraction is not None else None
            )
            print(
                f"| {summary['label']} | {summary['dataset_variant']} | {source_name} "
                f"| {hand_summary['valid_count']} "
                f"| {_format_percentage(one_or_more_hand_fraction)} "
                f"| {_format_percentage(_fraction(hand_summary, '2'))} "
                f"| {active_summary['valid_count']} "
                f"| {_format_percentage(_fraction(active_summary, 'yes'))} "
                f"| {_format_percentage(_optional_float(hand_summary['agreement_fraction']))} "
                f"| {_format_percentage(_optional_float(active_summary['agreement_fraction']))} |"
            )


def _positive_integer(raw_value: str) -> int:
    parsed_value = int(raw_value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def _default_output_directory(dataset_variant: DatasetVariant, model: str) -> Path:
    sanitized_model_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-") or "model"
    return DEFAULT_RUNS_DIRECTORY / f"{dataset_variant.value}-{sanitized_model_name}"


def _run_configuration_from_arguments(arguments: argparse.Namespace) -> EvaluationConfiguration:
    if not arguments.model:
        raise ValueError("--model or OPENAI_MODEL is required")
    if not arguments.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    dataset_variant = DatasetVariant(arguments.dataset)
    output_directory = arguments.output or _default_output_directory(
        dataset_variant, arguments.model
    )
    task_definitions = load_task_definitions(
        arguments.hand_count_prompt, arguments.active_manipulation_prompt
    )
    return EvaluationConfiguration(
        dataset_variant=dataset_variant,
        selected_sources=_selected_sources(arguments.source),
        selected_tasks=_selected_tasks(arguments.task),
        data_directory=arguments.data_directory,
        output_directory=output_directory,
        model=arguments.model,
        base_url=arguments.base_url,
        api_key_environment_variable=arguments.api_key_env,
        allow_missing_api_key=arguments.allow_missing_api_key,
        response_format=ResponseFormat(arguments.response_format),
        temperature=arguments.temperature,
        max_tokens=arguments.max_tokens,
        max_retries=arguments.max_retries,
        worker_count=arguments.workers,
        row_limit_per_source=arguments.limit,
        label=arguments.label or arguments.model,
        task_definitions=task_definitions,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run an Inspect evaluation")
    run_parser.add_argument("--dataset", type=DatasetVariant, choices=DatasetVariant, required=True)
    run_parser.add_argument(
        "--source",
        action="append",
        type=SourceSelection,
        choices=SourceSelection,
        help="repeat to select corpora; default: all three published corpora",
    )
    run_parser.add_argument(
        "--task",
        action="append",
        type=EvaluationTask,
        choices=EvaluationTask,
        help="repeat to select tasks; default: both published tasks",
    )
    run_parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    run_parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    run_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_parser.add_argument("--allow-missing-api-key", action="store_true")
    run_parser.add_argument(
        "--response-format",
        type=ResponseFormat,
        choices=ResponseFormat,
        default=ResponseFormat.JSON_SCHEMA,
    )
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--max-tokens", type=_positive_integer, default=32)
    run_parser.add_argument("--max-retries", type=_positive_integer, default=5)
    run_parser.add_argument("--workers", type=_positive_integer, default=8)
    run_parser.add_argument(
        "--limit",
        type=_positive_integer,
        default=None,
        help="evaluate only the first N pinned rows per source",
    )
    run_parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    run_parser.add_argument("--download", action="store_true")
    run_parser.add_argument("--output", type=Path, default=None)
    run_parser.add_argument("--label", default=None, help="display label used by compare")
    run_parser.add_argument(
        "--hand-count-prompt",
        type=Path,
        default=DEFAULT_PROMPTS_DIRECTORY / "hand_count.txt",
    )
    run_parser.add_argument(
        "--active-manipulation-prompt",
        type=Path,
        default=DEFAULT_PROMPTS_DIRECTORY / "active_manipulation.txt",
    )

    compare_parser = subparsers.add_parser("compare", help="compare completed run summaries")
    compare_parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="paths to evaluation run summary JSON files to compare",
    )
    return parser


def main() -> None:
    parser = _argument_parser()
    arguments = parser.parse_args()
    try:
        match arguments.command:
            case "run":
                configuration = _run_configuration_from_arguments(arguments)
                run_evaluation(configuration, download=arguments.download)
            case "compare":
                compare_summaries(arguments.summaries)
            case unknown_command:
                raise AssertionError(f"unhandled command: {unknown_command}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
