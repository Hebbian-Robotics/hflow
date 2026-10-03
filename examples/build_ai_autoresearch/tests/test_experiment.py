"""Search scores and frozen selections preserve the experiment's boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest
from examples.build_ai_autoresearch.contracts import (
    BaselineReport,
    Evaluation,
    PredictionRecord,
    SampleRecord,
    SelectedBaseline,
    SelectedTrial,
    TrialBudget,
    TrialReport,
    checkpoint_digests,
    file_sha256,
    read_record,
    score_predictions,
    write_record,
)
from examples.build_ai_autoresearch.experiment import (
    freeze_selection,
    initialize_experiment,
    validate_experiment,
)
from examples.build_ai_autoresearch.prepare import (
    PUBLISHED_SOURCES,
    Corpus,
    HandCount,
    SourceFile,
    prepare_dataset,
)
from examples.build_ai_autoresearch.tests.test_preparation import _image_bytes, _write_source
from examples.build_ai_autoresearch.training_runner import snapshot_training_source


def _evaluation(samples: tuple[SampleRecord, ...], correct: bool) -> Evaluation:
    predictions = tuple(
        PredictionRecord(
            sample=sample,
            raw_answer=str(int(sample.hand_count)) if correct else "uncertain",
            prediction=sample.hand_count if correct else None,
        )
        for sample in samples
    )
    return Evaluation(
        predictions=predictions,
        metrics=score_predictions(predictions),
        per_corpus={Corpus.BUILD.value: score_predictions(predictions)},
        elapsed_seconds=0.1,
    )


@pytest.fixture
def experiment_directory(tmp_path: Path) -> Path:
    source = tmp_path / "source.parquet"
    _write_source(source, [(_image_bytes(f"#{index:06x}"), index % 3) for index in range(300)])
    prepared = tmp_path / "prepared"
    prepare_dataset(
        [SourceFile(replace(PUBLISHED_SOURCES[0], expected_sha256=None), source)], prepared
    )
    experiment = tmp_path / "experiment"
    initialize_experiment(
        prepared,
        experiment,
        TrialBudget(
            train_samples=6,
            development_samples=6,
            confirmation_samples=6,
            training_seconds=10.0,
            max_training_steps=1,
            max_trials=2,
            reference_reason="unit outcome fixture",
        ),
    )
    return experiment


def _baseline(experiment: Path, *, correct: bool) -> None:
    protocol = validate_experiment(experiment)
    write_record(
        experiment / "baseline.json",
        BaselineReport(
            protocol_sha256=file_sha256(experiment / "protocol.json"),
            runtime={"backend": "fixture"},
            evaluation=_evaluation(protocol.development, correct),
        ),
    )


def _trial(experiment: Path, *, correct: bool) -> Path:
    protocol = validate_experiment(experiment)
    directory = experiment / "trials/trial-000"
    adapter = directory / "adapter"
    adapter.mkdir(parents=True)
    source_digest = snapshot_training_source(experiment / "train.py", directory)
    (adapter / "owned-checkpoint.bin").write_bytes(b"checkpoint receipt fixture")
    write_record(
        directory / "report.json",
        TrialReport(
            protocol_sha256=file_sha256(experiment / "protocol.json"),
            training_source_sha256=source_digest,
            runtime={"backend": "fixture"},
            checkpoint_files=checkpoint_digests(adapter),
            training_seconds=0.1,
            training_losses=(1.0,),
            trainable_parameters=4,
            evaluation=_evaluation(protocol.development, correct),
        ),
    )
    return directory


def test_invalid_answers_remain_in_the_score_denominator() -> None:
    records = tuple(
        PredictionRecord(
            sample=SampleRecord(
                sample_id=f"{label:064x}",
                image_sha256=f"{label:064x}",
                hand_count=HandCount(label),
                corpora=(Corpus.BUILD,),
            ),
            raw_answer=answer,
            prediction=HandCount(int(answer)) if answer in ("0", "1", "2") else None,
        )
        for label, answer in ((0, "0"), (1, "0"), (2, "two"))
    )
    metrics = score_predictions(records)
    assert metrics.sample_count == 3
    assert metrics.invalid_count == 1
    assert metrics.accuracy == pytest.approx(1 / 3)
    assert metrics.macro_f1 == pytest.approx(2 / 9)
    assert metrics.class_support == (1, 1, 1)


def test_initialization_copies_only_frozen_development_and_training(
    experiment_directory: Path,
) -> None:
    protocol = validate_experiment(experiment_directory)
    assert {path.name for path in (experiment_directory / "media").iterdir()} == {
        sample.image_filename for sample in (*protocol.train, *protocol.development)
    }
    assert not (experiment_directory / "confirmation").exists()
    changed_image = experiment_directory / "media" / protocol.development[0].image_filename
    changed_image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        validate_experiment(experiment_directory)


@pytest.mark.parametrize("baseline_correct", [False, True])
def test_freeze_selects_improvement_and_keeps_baseline_on_ties(
    experiment_directory: Path, baseline_correct: bool
) -> None:
    _baseline(experiment_directory, correct=baseline_correct)
    _trial(experiment_directory, correct=True)
    selection = freeze_selection(experiment_directory)
    assert isinstance(selection.selected, SelectedBaseline if baseline_correct else SelectedTrial)
    with pytest.raises(FileExistsError):
        freeze_selection(experiment_directory)


def test_changed_checkpoint_cannot_be_selected(experiment_directory: Path) -> None:
    _baseline(experiment_directory, correct=False)
    trial = _trial(experiment_directory, correct=True)
    (trial / "adapter/owned-checkpoint.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        freeze_selection(experiment_directory)
    assert not (experiment_directory / "selection.json").exists()


def test_editable_training_code_is_snapshotted_without_changing_the_evaluator(
    experiment_directory: Path,
) -> None:
    protocol = validate_experiment(experiment_directory)
    _baseline(experiment_directory, correct=False)
    directory = _trial(experiment_directory, correct=True)
    original_snapshot = (directory / "train.py").read_bytes()
    with (experiment_directory / "train.py").open("a") as editable:
        editable.write("\n# A new hypothesis can edit actual training code.\n")
    assert validate_experiment(experiment_directory) == protocol
    selection = freeze_selection(experiment_directory)
    assert isinstance(selection.selected, SelectedTrial)
    assert (directory / "train.py").read_bytes() == original_snapshot
    (directory / "train.py").write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="training source changed"):
        freeze_selection(experiment_directory)


def test_duplicate_budget_fields_are_rejected(tmp_path: Path) -> None:
    budget = tmp_path / "budget.json"
    budget.write_text(
        '{"training_seconds":10,"training_seconds":20,"reference_reason":"fixture check"}'
    )
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_record(budget, TrialBudget)
