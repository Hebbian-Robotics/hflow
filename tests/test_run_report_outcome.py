"""The one-outcome invariant on CheckRunReport and EnrichmentRunReport."""

from typing import Any, cast

import pytest

import hflow


def _registered_check() -> Any:
    return cast(Any, object())


def _registered_enrichment() -> Any:
    return cast(Any, object())


def test_a_check_report_cannot_be_built_without_an_outcome() -> None:
    with pytest.raises(TypeError):
        cast(Any, hflow.CheckRunReport)(check=_registered_check())


def test_an_enrichment_report_cannot_be_built_without_an_outcome() -> None:
    with pytest.raises(TypeError):
        cast(Any, hflow.EnrichmentRunReport)(enrichment=_registered_enrichment())


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (hflow.Measured(hflow.CheckResult()), hflow.CheckStatus.MEASURED),
        (hflow.Measured(hflow.CheckResult(verdict=True)), hflow.CheckStatus.PASSED),
        (hflow.Measured(hflow.CheckResult(verdict=False)), hflow.CheckStatus.FAILED),
        (hflow.Errored("boom"), hflow.CheckStatus.ERROR),
        (
            hflow.NotRun(hflow.SkippedByQuarantine(("quarantined:x",))),
            hflow.CheckStatus.SKIPPED,
        ),
        (
            hflow.NotRun(hflow.SupersededByPipeline(("fps",))),
            hflow.CheckStatus.SUPERSEDED,
        ),
    ],
)
def test_every_check_status_still_comes_from_the_same_condition(
    outcome: hflow.CheckOutcome, expected: hflow.CheckStatus
) -> None:
    run = hflow.CheckRunReport(check=_registered_check(), outcome=outcome)
    assert run.status is expected


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (hflow.Measured(hflow.EnrichmentResult()), hflow.CheckStatus.MEASURED),
        (hflow.Errored("boom"), hflow.CheckStatus.ERROR),
        (
            hflow.NotRun(hflow.SkippedByQuarantine(("quarantined:x",))),
            hflow.CheckStatus.SKIPPED,
        ),
        (
            hflow.NotRun(hflow.SupersededByPipeline(("caption",))),
            hflow.CheckStatus.SUPERSEDED,
        ),
        (
            hflow.PublishFailed(result=hflow.EnrichmentResult(), error="boom"),
            hflow.CheckStatus.ERROR,
        ),
    ],
)
def test_every_enrichment_status_still_comes_from_the_same_condition(
    outcome: hflow.EnrichmentOutcome, expected: hflow.CheckStatus
) -> None:
    run = hflow.EnrichmentRunReport(enrichment=_registered_enrichment(), outcome=outcome)
    assert run.status is expected


def test_a_check_report_reads_back_exactly_the_outcome_it_holds() -> None:
    result = hflow.CheckResult(measurements={"fps": 30.0})
    measured = hflow.CheckRunReport(check=_registered_check(), outcome=hflow.Measured(result))
    assert (measured.result, measured.error, measured.not_run) == (result, None, None)

    errored = hflow.CheckRunReport(check=_registered_check(), outcome=hflow.Errored("boom"))
    assert (errored.result, errored.error, errored.not_run) == (None, "boom", None)

    skipped = hflow.SkippedByQuarantine(("quarantined:x",))
    not_run = hflow.CheckRunReport(check=_registered_check(), outcome=hflow.NotRun(skipped))
    assert (not_run.result, not_run.error, not_run.not_run) == (None, None, skipped)


def test_a_publish_failure_keeps_the_labels_it_already_had() -> None:
    result = hflow.EnrichmentResult(labels={"caption": "a robot"})
    run = hflow.EnrichmentRunReport(
        enrichment=_registered_enrichment(),
        outcome=hflow.PublishFailed(result=result, error="artifact 'sheet' could not be published"),
    )
    assert run.status is hflow.CheckStatus.ERROR
    assert run.result is result
    assert run.error == "artifact 'sheet' could not be published"
