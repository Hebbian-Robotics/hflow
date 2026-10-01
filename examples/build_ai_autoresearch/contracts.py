"""Owned experiment records and fixed three-class teacher-agreement scoring."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from examples.build_ai_autoresearch.prepare import Corpus, HandCount
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
RecordType = TypeVar("RecordType", bound=BaseModel)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SamplingBalance(StrEnum):
    UNIFORM = "uniform"
    CLASS_BALANCED = "class-balanced"


class TrainingRecipe(Record):
    learning_rate: Annotated[float, Field(gt=0, le=0.01, allow_inf_nan=False)] = 0.0002
    lora_rank: Literal[2, 4, 8] = 4
    sampling_balance: SamplingBalance = SamplingBalance.UNIFORM


class TrialBudget(Record):
    train_samples: Annotated[int, Field(ge=3, le=4096)] = 192
    development_samples: Annotated[int, Field(ge=3, le=512)] = 48
    confirmation_samples: Annotated[int, Field(ge=3, le=512)] = 48
    training_steps: Annotated[int, Field(ge=1, le=1024)] = 256
    max_trials: Annotated[int, Field(ge=1, le=8)] = 8
    cpu_threads: Annotated[int, Field(ge=1, le=16)] = 4
    seed: int = 42
    reference_reason: Annotated[str, Field(min_length=8, max_length=256)]


class SampleRecord(Record):
    sample_id: Sha256
    image_sha256: Sha256
    hand_count: HandCount
    corpora: tuple[Corpus, ...]

    @property
    def image_filename(self) -> str:
        return f"{self.image_sha256}.image"


class Protocol(Record):
    schema_version: Literal[1] = 1
    model_id: Literal["HuggingFaceTB/SmolVLM2-256M-Video-Instruct"] = (
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    )
    model_revision: Literal["067788b187b95ebe7b2e040b3e4299e342e5b8fd"] = (
        "067788b187b95ebe7b2e040b3e4299e342e5b8fd"
    )
    backend: Literal["transformers-cpu-reference"] = "transformers-cpu-reference"
    independence_scope: Literal["exact-pixel-frame-only"] = "exact-pixel-frame-only"
    image_edge: Literal[512] = 512
    max_new_tokens: Literal[4] = 4
    budget: TrialBudget
    preparation_sha256: Sha256
    confirmation_manifest_sha256: Sha256
    evaluator_files: dict[str, Sha256]
    train: tuple[SampleRecord, ...]
    development: tuple[SampleRecord, ...]

    @model_validator(mode="after")
    def check_sample_separation(self) -> Protocol:
        train_ids = {sample.sample_id for sample in self.train}
        development_ids = {sample.sample_id for sample in self.development}
        if len(train_ids) != len(self.train) or len(development_ids) != len(self.development):
            raise ValueError("sample identities must be unique within each split")
        if train_ids & development_ids:
            raise ValueError("training and development samples overlap")
        if (
            len(self.train) != self.budget.train_samples
            or len(self.development) != self.budget.development_samples
        ):
            raise ValueError("sample counts differ from the frozen budget")
        for samples in (self.train, self.development):
            if {sample.hand_count for sample in samples} != set(HandCount):
                raise ValueError("each development/training subset must contain all three classes")
        return self


class PredictionRecord(Record):
    sample: SampleRecord
    raw_answer: str
    prediction: HandCount | None

    @model_validator(mode="after")
    def verify_answer_parsing(self) -> PredictionRecord:
        answer = self.raw_answer.strip()
        expected = HandCount(int(answer)) if answer in ("0", "1", "2") else None
        if self.prediction != expected:
            raise ValueError("prediction differs from the strict parsed answer")
        return self


class Metrics(Record):
    sample_count: int
    correct_count: int
    invalid_count: int
    accuracy: FiniteFloat
    macro_f1: FiniteFloat
    class_support: tuple[int, int, int]
    class_f1: tuple[FiniteFloat, FiniteFloat, FiniteFloat]


def score_predictions(predictions: Sequence[PredictionRecord]) -> Metrics:
    if not predictions:
        raise ValueError("evaluation needs samples")
    if len({record.sample.sample_id for record in predictions}) != len(predictions):
        raise ValueError("evaluation contains duplicate sample identities")
    support = [0, 0, 0]
    true_positive = [0, 0, 0]
    false_positive = [0, 0, 0]
    false_negative = [0, 0, 0]
    invalid_count = 0
    for record in predictions:
        expected = record.sample.hand_count
        support[expected] += 1
        if record.prediction is None:
            invalid_count += 1
            false_negative[expected] += 1
        elif record.prediction == expected:
            true_positive[expected] += 1
        else:
            false_positive[record.prediction] += 1
            false_negative[expected] += 1
    class_f1 = [
        2 * correct / denominator if (denominator := 2 * correct + wrong + missed) else 0.0
        for correct, wrong, missed in zip(
            true_positive, false_positive, false_negative, strict=True
        )
    ]
    return Metrics(
        sample_count=len(predictions),
        correct_count=sum(true_positive),
        invalid_count=invalid_count,
        accuracy=sum(true_positive) / len(predictions),
        macro_f1=sum(class_f1) / 3,
        class_support=(support[0], support[1], support[2]),
        class_f1=(class_f1[0], class_f1[1], class_f1[2]),
    )


class Evaluation(Record):
    predictions: tuple[PredictionRecord, ...]
    metrics: Metrics
    per_corpus: dict[str, Metrics]
    elapsed_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def verify_scores(self) -> Evaluation:
        if self.metrics != score_predictions(self.predictions):
            raise ValueError("metrics differ from recorded predictions")
        expected_corpora = {
            corpus for prediction in self.predictions for corpus in prediction.sample.corpora
        }
        expected_metrics = {
            corpus.value: score_predictions(
                [
                    prediction
                    for prediction in self.predictions
                    if corpus in prediction.sample.corpora
                ]
            )
            for corpus in expected_corpora
        }
        if self.per_corpus != expected_metrics:
            raise ValueError("per-corpus metrics differ from recorded predictions")
        return self


class BaselineReport(Record):
    protocol_sha256: Sha256
    runtime: dict[str, str]
    evaluation: Evaluation


class TrialReport(Record):
    protocol_sha256: Sha256
    recipe: TrainingRecipe
    candidate_sha256: Sha256
    runtime: dict[str, str]
    checkpoint_files: dict[str, Sha256]
    training_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    training_losses: tuple[FiniteFloat, ...]
    trainable_parameters: int
    evaluation: Evaluation


class SelectedBaseline(Record):
    kind: Literal["baseline"] = "baseline"


class SelectedTrial(Record):
    kind: Literal["trial"] = "trial"
    trial: Annotated[str, Field(pattern=r"^trial-[0-9]{3}$")]
    report_sha256: Sha256


class SelectionReceipt(Record):
    protocol_sha256: Sha256
    baseline_sha256: Sha256
    selected: Annotated[SelectedBaseline | SelectedTrial, Field(discriminator="kind")]


class ConfirmationReport(Record):
    selection_sha256: Sha256
    protocol_sha256: Sha256
    evaluation: Evaluation
    runtime: dict[str, str]


class ExportReceipt(Record):
    selection_sha256: Sha256
    confirmation_sha256: Sha256
    model_files: dict[str, Sha256]
    runtime: dict[str, str]


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _unique_json_fields(fields: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in fields:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def read_record(path: Path, record_type: type[RecordType]) -> RecordType:
    encoded = path.read_bytes()
    json.loads(encoded, object_pairs_hook=_unique_json_fields)
    return record_type.model_validate_json(encoded)


def write_record(path: Path, record: BaseModel) -> None:
    with path.open("x") as output:
        output.write(record.model_dump_json(indent=2))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def checkpoint_digests(directory: Path) -> dict[str, str]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("checkpoint must contain regular owned files")
    return {str(path.relative_to(directory)): file_sha256(path) for path in files}
