"""Frozen data, bounded trials, and explicit one-time confirmation.

The editor changes training code. This driver owns data, budgets, scoring,
selection, and receipts; it never supplies confirmation images during search.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import typer
from examples.build_ai_autoresearch.contracts import (
    BaselineReport,
    ConfirmationReport,
    ExportReceipt,
    Protocol,
    SampleRecord,
    SelectedBaseline,
    SelectedTrial,
    SelectionReceipt,
    Sha256,
    TrialBudget,
    TrialReport,
    WorkerOutcome,
    checkpoint_digests,
    file_sha256,
    read_record,
    write_record,
)
from examples.build_ai_autoresearch.prepare import Corpus, HandCount
from examples.build_ai_autoresearch.training_runner import run_worker, snapshot_training_source
from pydantic import BaseModel, ConfigDict

EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_DIRECTORY.parent.parent
EVALUATOR_PATHS = (
    *tuple(
        EXAMPLE_DIRECTORY / name
        for name in (
            "contracts.py",
            "experiment.py",
            "model_runtime.py",
            "prepare.py",
            "training_api.py",
            "training_runner.py",
            "training_worker.py",
        )
    ),
    REPOSITORY_ROOT / "uv.lock",
    REPOSITORY_ROOT / "src/hflow/build_ai_vlm_checks.py",
)
app = typer.Typer(no_args_is_help=True)


class PreparationEvidence(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, frozen=True)
    schema_version: int
    label_source: str
    independence_scope: str
    manifest_sha256: Sha256
    split_receipt_sha256: Sha256


class PartitionEvidence(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, frozen=True)
    name: str
    sha256: Sha256


class SplitEvidence(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, frozen=True)
    input_sha256: Sha256
    partitions: tuple[PartitionEvidence, ...]


@dataclass(frozen=True)
class PreparedSample:
    record: SampleRecord
    source_image: Path


def _evaluator_digests() -> dict[str, str]:
    return {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in EVALUATOR_PATHS}


def _prepared_evidence(prepared: Path) -> tuple[PreparationEvidence, dict[str, str]]:
    evidence = read_record(prepared / "preparation.json", PreparationEvidence)
    if (
        evidence.schema_version != 1
        or evidence.label_source != "published-gemini-teacher-labels"
        or evidence.independence_scope != "exact-pixel-frame-only"
    ):
        raise ValueError("unsupported preparation task or independence scope")
    if (
        file_sha256(prepared / "samples.parquet") != evidence.manifest_sha256
        or file_sha256(prepared / "splits/receipt.json") != evidence.split_receipt_sha256
    ):
        raise ValueError("prepared manifest or split receipt changed")
    splits = read_record(prepared / "splits/receipt.json", SplitEvidence)
    if splits.input_sha256 != evidence.manifest_sha256:
        raise ValueError("splits refer to another prepared manifest")
    partitions = {partition.name: partition.sha256 for partition in splits.partitions}
    if len(partitions) != len(splits.partitions) or set(partitions) != {
        "train",
        "development",
        "test",
    }:
        raise ValueError("preparation must contain distinct train/development/test partitions")
    return evidence, partitions


def _load_partition(prepared: Path, name: str, expected_sha256: str) -> list[PreparedSample]:
    path = prepared / "splits" / f"{name}.parquet"
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{name} manifest changed")
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT sample_id, pixel_sha256, image_path, image_sha256, hand_count, source_references_json FROM read_parquet(?) ORDER BY sample_id",
            [str(path)],
        ).fetchall()
    result: list[PreparedSample] = []
    for sample_id, pixel_digest, image_path, encoded_digest, hand_count, source_references in rows:
        if (
            sample_id != pixel_digest
            or not isinstance(image_path, str)
            or isinstance(hand_count, bool)
            or not isinstance(hand_count, int)
            or hand_count not in (0, 1, 2)
        ):
            raise ValueError("invalid sample identity, image path, or hand label")
        image = (prepared / image_path).resolve()
        if not image.is_relative_to(prepared.resolve()):
            raise ValueError("sample image escapes the prepared dataset root")
        references = json.loads(source_references)
        if not isinstance(references, list) or not references:
            raise ValueError("sample needs source references")
        corpora: set[Corpus] = set()
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError("source reference must be an object")
            corpora.add(Corpus(reference["corpus"]))
        record = SampleRecord(
            sample_id=sample_id,
            image_sha256=encoded_digest,
            hand_count=HandCount(hand_count),
            corpora=tuple(sorted(corpora)),
        )
        result.append(PreparedSample(record, image))
    if len({sample.record.sample_id for sample in result}) != len(result):
        raise ValueError("partition has duplicate pixel identities")
    return result


def choose_subset(
    samples: Sequence[PreparedSample], count: int, seed: int, *, balanced: bool
) -> tuple[PreparedSample, ...]:
    if count < 3 or len(samples) < count:
        raise ValueError("subset needs at least three samples and enough input rows")
    ordered = sorted(
        samples,
        key=lambda sample: hashlib.sha256(f"{seed}:{sample.record.sample_id}".encode()).digest(),
    )
    classes = {
        label: [sample for sample in ordered if sample.record.hand_count == label]
        for label in HandCount
    }
    if any(not group for group in classes.values()):
        raise ValueError("subset source must contain all three hand classes")
    if balanced:
        selected: list[PreparedSample] = []
        position = 0
        while len(selected) < count:
            for group in classes.values():
                if position < len(group) and len(selected) < count:
                    selected.append(group[position])
            position += 1
    else:
        selected = [group[0] for group in classes.values()]
        selected_ids = {sample.record.sample_id for sample in selected}
        selected.extend(sample for sample in ordered if sample.record.sample_id not in selected_ids)
        selected = selected[:count]
    return tuple(sorted(selected, key=lambda sample: sample.record.sample_id))


def _copy_images(samples: Sequence[PreparedSample], media_directory: Path) -> None:
    media_directory.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        if file_sha256(sample.source_image) != sample.record.image_sha256:
            raise ValueError(f"source image changed: {sample.record.sample_id}")
        destination = media_directory / sample.record.image_filename
        if not destination.exists():
            shutil.copyfile(sample.source_image, destination)
        if file_sha256(destination) != sample.record.image_sha256:
            raise ValueError(f"copied image differs: {sample.record.sample_id}")


def initialize_experiment(prepared: Path, experiment: Path, budget: TrialBudget) -> Protocol:
    _, partitions = _prepared_evidence(prepared)
    train = choose_subset(
        _load_partition(prepared, "train", partitions["train"]),
        budget.train_samples,
        budget.seed,
        balanced=False,
    )
    development = choose_subset(
        _load_partition(prepared, "development", partitions["development"]),
        budget.development_samples,
        budget.seed,
        balanced=True,
    )
    protocol = Protocol(
        budget=budget,
        preparation_sha256=file_sha256(prepared / "preparation.json"),
        confirmation_manifest_sha256=partitions["test"],
        evaluator_files=_evaluator_digests(),
        train=tuple(sample.record for sample in train),
        development=tuple(sample.record for sample in development),
    )
    experiment.mkdir(parents=True, exist_ok=False)
    try:
        _copy_images((*train, *development), experiment / "media")
        shutil.copyfile(EXAMPLE_DIRECTORY / "train.py", experiment / "train.py")
        (experiment / "trials").mkdir()
        write_record(experiment / "protocol.json", protocol)
    except Exception:
        shutil.rmtree(experiment)
        raise
    return protocol


def validate_experiment(experiment: Path) -> Protocol:
    protocol = read_record(experiment / "protocol.json", Protocol)
    if protocol.evaluator_files != _evaluator_digests():
        raise ValueError("evaluator or dependency lock changed; initialize a new experiment")
    for sample in (*protocol.train, *protocol.development):
        if file_sha256(experiment / "media" / sample.image_filename) != sample.image_sha256:
            raise ValueError("frozen development/training image changed")
    return protocol


def _validate_evaluation_samples(
    samples: Sequence[SampleRecord], report: BaselineReport | TrialReport
) -> None:
    if tuple(prediction.sample for prediction in report.evaluation.predictions) != tuple(samples):
        raise ValueError("report evaluated a different development subset")


def _baseline(experiment: Path, protocol: Protocol) -> BaselineReport:
    report = read_record(experiment / "baseline.json", BaselineReport)
    if report.protocol_sha256 != file_sha256(experiment / "protocol.json"):
        raise ValueError("baseline refers to another protocol")
    _validate_evaluation_samples(protocol.development, report)
    return report


def _completed_trials(experiment: Path, protocol: Protocol) -> list[tuple[Path, TrialReport]]:
    result: list[tuple[Path, TrialReport]] = []
    for directory in sorted((experiment / "trials").iterdir()):
        if not (directory / "report.json").exists():
            continue
        report = read_record(directory / "report.json", TrialReport)
        if (
            report.protocol_sha256 != file_sha256(experiment / "protocol.json")
            or not 1 <= len(report.training_losses) <= protocol.budget.max_training_steps
            or report.training_seconds > protocol.budget.training_seconds
        ):
            raise ValueError("trial protocol or completed step count differs")
        _validate_evaluation_samples(protocol.development, report)
        if file_sha256(directory / "train.py") != report.training_source_sha256:
            raise ValueError("trial training source changed")
        if checkpoint_digests(directory / "adapter") != report.checkpoint_files:
            raise ValueError("trial checkpoint changed")
        result.append((directory, report))
    return result


@app.command("init")
def initialize(
    prepared: Path,
    experiment: Path,
    reference_reason: str,
    train_samples: int = 192,
    development_samples: int = 48,
    confirmation_samples: int = 48,
    training_seconds: float = 300.0,
    max_training_steps: int = 1024,
    max_trials: int = 8,
    cpu_threads: int = 4,
) -> None:
    """Freeze a CPU reference protocol and copy only train/development images."""
    protocol = initialize_experiment(
        prepared,
        experiment,
        TrialBudget(
            train_samples=train_samples,
            development_samples=development_samples,
            confirmation_samples=confirmation_samples,
            training_seconds=training_seconds,
            max_training_steps=max_training_steps,
            max_trials=max_trials,
            cpu_threads=cpu_threads,
            reference_reason=reference_reason,
        ),
    )
    print(
        f"Frozen {len(protocol.train)} training and {len(protocol.development)} development samples; test images are not copied."
    )


@app.command("baseline-transformers-reference")
def baseline(experiment: Path) -> None:
    """Evaluate the pinned untouched model on the frozen development subset."""
    from examples.build_ai_autoresearch.model_runtime import CpuReferenceRuntime

    protocol = validate_experiment(experiment)
    if (experiment / "baseline.json").exists():
        raise FileExistsError("baseline already exists")
    runtime = CpuReferenceRuntime(protocol)
    protocol_sha256 = file_sha256(experiment / "protocol.json")
    evaluation = runtime.evaluate(protocol.development, experiment / "media")
    validate_experiment(experiment)
    if file_sha256(experiment / "protocol.json") != protocol_sha256:
        raise ValueError("protocol changed during baseline evaluation")
    write_record(
        experiment / "baseline.json",
        BaselineReport(
            protocol_sha256=file_sha256(experiment / "protocol.json"),
            runtime=runtime.runtime_identity,
            evaluation=evaluation,
        ),
    )
    print(
        f"Baseline macro-F1: {evaluation.metrics.macro_f1:.6f}; invalid: {evaluation.metrics.invalid_count}"
    )


@app.command("trial-transformers-reference")
def trial(experiment: Path) -> None:
    """Execute a train.py snapshot within its time budget; evaluate separately."""
    from examples.build_ai_autoresearch.model_runtime import CpuReferenceRuntime

    protocol = validate_experiment(experiment)
    baseline_report = _baseline(experiment, protocol)
    if (experiment / "selection.json").exists():
        raise ValueError("selection is frozen; development search has ended")
    attempts = len(list((experiment / "trials").iterdir()))
    if attempts >= protocol.budget.max_trials:
        raise ValueError("trial budget is exhausted; failed attempts count")
    directory = experiment / "trials" / f"trial-{attempts:03}"
    directory.mkdir(exist_ok=False)
    try:
        training_source_digest = snapshot_training_source(experiment / "train.py", directory)
        run_worker(
            [
                sys.executable,
                "-m",
                "examples.build_ai_autoresearch.training_worker",
                str(experiment.resolve()),
                str(directory.resolve()),
            ],
            directory,
            protocol.budget,
        )
        training = read_record(directory / "outcome.json", WorkerOutcome)
        if (
            training.training_source_sha256 != training_source_digest
            or training.training_seconds > protocol.budget.training_seconds
            or not 1 <= len(training.training_losses) <= protocol.budget.max_training_steps
        ):
            raise ValueError("worker source or compute-budget receipt differs")
        if checkpoint_digests(directory / "adapter") != training.checkpoint_files:
            raise ValueError("worker checkpoint changed")
        validate_experiment(experiment)
        if file_sha256(experiment / "protocol.json") != baseline_report.protocol_sha256:
            raise ValueError("protocol changed during training")
        # Candidate code never executes in the evaluator process; reload its saved LoRA on fresh base weights.
        runtime = CpuReferenceRuntime(protocol, directory / "adapter")
        evaluation = runtime.evaluate(protocol.development, experiment / "media")
        validate_experiment(experiment)
        _baseline(experiment, protocol)
        if file_sha256(experiment / "protocol.json") != baseline_report.protocol_sha256:
            raise ValueError("protocol changed during the trial")
        if (
            file_sha256(experiment / "train.py") != training_source_digest
            or file_sha256(directory / "train.py") != training_source_digest
        ):
            raise ValueError("training source changed during the trial")
        if (
            runtime.runtime_identity != baseline_report.runtime
            or training.runtime != baseline_report.runtime
        ):
            raise ValueError("trial runtime differs from the baseline runtime")
        write_record(
            directory / "report.json",
            TrialReport(
                protocol_sha256=baseline_report.protocol_sha256,
                training_source_sha256=training_source_digest,
                runtime=runtime.runtime_identity,
                checkpoint_files=training.checkpoint_files,
                training_seconds=training.training_seconds,
                training_losses=training.training_losses,
                trainable_parameters=training.trainable_parameters,
                evaluation=evaluation,
            ),
        )
        print(
            f"{directory.name}: macro-F1 {evaluation.metrics.macro_f1:.6f}; baseline {baseline_report.evaluation.metrics.macro_f1:.6f}; {len(training.training_losses)} steps in {training.training_seconds:.2f}s"
        )
    except Exception as error:
        with (directory / "failure.json").open("x") as output:
            json.dump({"error": str(error)}, output, indent=2)
        raise


def freeze_selection(experiment: Path) -> SelectionReceipt:
    protocol = validate_experiment(experiment)
    baseline_report = _baseline(experiment, protocol)
    best_score = baseline_report.evaluation.metrics.macro_f1
    selected: SelectedBaseline | SelectedTrial = SelectedBaseline()
    for directory, report in _completed_trials(experiment, protocol):
        if report.runtime != baseline_report.runtime:
            raise ValueError("trial runtime differs from the baseline runtime")
        if report.evaluation.metrics.macro_f1 > best_score:
            best_score = report.evaluation.metrics.macro_f1
            selected = SelectedTrial(
                trial=directory.name, report_sha256=file_sha256(directory / "report.json")
            )
    receipt = SelectionReceipt(
        protocol_sha256=file_sha256(experiment / "protocol.json"),
        baseline_sha256=file_sha256(experiment / "baseline.json"),
        selected=selected,
    )
    write_record(experiment / "selection.json", receipt)
    return receipt


@app.command("freeze")
def freeze(experiment: Path) -> None:
    """Freeze the best completed development result; ties retain the baseline."""
    receipt = freeze_selection(experiment)
    print(receipt.selected.model_dump_json())


@app.command("confirm-transformers-reference")
def confirm(experiment: Path, prepared: Path) -> None:
    """Operator-only: evaluate frozen selection once on previously hidden test data."""
    protocol = validate_experiment(experiment)
    selection = read_record(experiment / "selection.json", SelectionReceipt)
    selection_sha256 = file_sha256(experiment / "selection.json")
    if selection.protocol_sha256 != file_sha256(
        experiment / "protocol.json"
    ) or selection.baseline_sha256 != file_sha256(experiment / "baseline.json"):
        raise ValueError("frozen protocol or baseline changed")
    if file_sha256(prepared / "preparation.json") != protocol.preparation_sha256:
        raise ValueError("confirmation uses another prepared dataset")
    if (experiment / "confirmation").exists():
        raise FileExistsError("confirmation was already attempted")
    adapter = selected_adapter(experiment, selection)
    output = experiment / "confirmation"
    output.mkdir(exist_ok=False)
    samples = choose_subset(
        _load_partition(prepared, "test", protocol.confirmation_manifest_sha256),
        protocol.budget.confirmation_samples,
        protocol.budget.seed,
        balanced=True,
    )
    development_ids = {sample.sample_id for sample in (*protocol.train, *protocol.development)}
    if any(sample.record.sample_id in development_ids for sample in samples):
        raise ValueError("confirmation samples overlap development/training data")
    _copy_images(samples, output / "media")
    from examples.build_ai_autoresearch.model_runtime import CpuReferenceRuntime

    runtime = CpuReferenceRuntime(protocol, adapter)
    baseline_report = _baseline(experiment, protocol)
    if runtime.runtime_identity != baseline_report.runtime:
        raise ValueError("confirmation runtime differs from development")
    evaluation = runtime.evaluate(tuple(sample.record for sample in samples), output / "media")
    validate_experiment(experiment)
    selected_adapter(experiment, selection)
    if (
        file_sha256(experiment / "selection.json") != selection_sha256
        or file_sha256(experiment / "protocol.json") != selection.protocol_sha256
        or file_sha256(experiment / "baseline.json") != selection.baseline_sha256
    ):
        raise ValueError("selection, protocol, or baseline changed during confirmation")
    write_record(
        output / "report.json",
        ConfirmationReport(
            selection_sha256=selection_sha256,
            protocol_sha256=selection.protocol_sha256,
            evaluation=evaluation,
            runtime=runtime.runtime_identity,
        ),
    )
    print(
        f"Confirmation macro-F1: {evaluation.metrics.macro_f1:.6f}; {evaluation.metrics.sample_count} samples"
    )


def selected_adapter(experiment: Path, selection: SelectionReceipt) -> Path | None:
    adapter: Path | None = None
    match selection.selected:
        case SelectedTrial(trial=name, report_sha256=expected_digest):
            directory = experiment / "trials" / name
            if file_sha256(directory / "report.json") != expected_digest:
                raise ValueError("selected trial report changed")
            report = read_record(directory / "report.json", TrialReport)
            if file_sha256(directory / "train.py") != report.training_source_sha256:
                raise ValueError("selected training source changed")
            baseline_report = read_record(experiment / "baseline.json", BaselineReport)
            if (
                report.protocol_sha256 != selection.protocol_sha256
                or report.runtime != baseline_report.runtime
            ):
                raise ValueError("selected trial protocol or runtime differs")
            adapter = directory / "adapter"
            if checkpoint_digests(adapter) != report.checkpoint_files:
                raise ValueError("selected checkpoint changed")
        case SelectedBaseline():
            pass
    return adapter


@app.command("export-transformers-reference")
def export(experiment: Path, output: Path) -> None:
    """Export the confirmed selection as a model/processor directory for serving."""
    from examples.build_ai_autoresearch.model_runtime import CpuReferenceRuntime

    protocol = validate_experiment(experiment)
    selection = read_record(experiment / "selection.json", SelectionReceipt)
    selection_digest = file_sha256(experiment / "selection.json")
    confirmation = read_record(experiment / "confirmation/report.json", ConfirmationReport)
    if (
        confirmation.selection_sha256 != selection_digest
        or confirmation.protocol_sha256 != file_sha256(experiment / "protocol.json")
        or selection.baseline_sha256 != file_sha256(experiment / "baseline.json")
    ):
        raise ValueError("confirmation does not bind the current selection, protocol, and baseline")
    runtime = CpuReferenceRuntime(protocol, selected_adapter(experiment, selection))
    if runtime.runtime_identity != confirmation.runtime:
        raise ValueError("export runtime differs from confirmation")
    output.mkdir(parents=True, exist_ok=False)
    runtime.export(output / "model")
    write_record(
        output / "receipt.json",
        ExportReceipt(
            selection_sha256=selection_digest,
            confirmation_sha256=file_sha256(experiment / "confirmation/report.json"),
            model_files=checkpoint_digests(output / "model"),
            runtime=runtime.runtime_identity,
        ),
    )
    print(f"Exported confirmed model and processor to {output / 'model'}")


if __name__ == "__main__":
    app()
