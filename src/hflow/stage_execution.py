"""Stage-run semantics: one owner for what every execution backend must do.

The generated Airflow sub-DAGs (``hflow.runtime._templates``) are thin
callers into this module, so the batch/online lane planning, the pipeline
loading contract, the per-episode accounting loop, and the error/quarantine
budgets live in the library rather than inside generated code strings. Any
other backend that runs the pipeline -- a hosted executor, a different
scheduler, a plain script -- reuses these functions instead of
re-implementing semantics by copying template text. (Bundles pin
``hflow==<renderer's version>``, so a rendered DAG and the library it calls
can never skew.)

Budget semantics (the mass-failure gates): a run tolerates up to
``max(8, ceil(1% of total))`` failures of a kind; checks decide quarantine,
so the quarantine budget applies only in the meta stage, and a run where
EVERY episode errored always fails regardless of budget.
"""

import math
import os
import traceback
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from hflow.batching import plan_batches
from hflow.catalog import QuarantineHistory
from hflow.steps import IngestMode, Stage
from hflow.storage import is_bucket_url, parse_storage_root

if TYPE_CHECKING:
    from hflow.app import App


class PlannedStageBatch(TypedDict):
    """One mapped batch, as plain JSON-able data (it crosses XCom borders)."""

    items: list[str]
    start_delay_s: float


class StageBatchCounts(TypedDict):
    """One batch's outcome tally, as plain JSON-able data (crosses XCom too)."""

    processed: int
    quarantined: int
    errors: int


# Where the runtime mounts the user's pipeline directory by default;
# HFLOW_USER_DIR relocates it on platforms that cannot mount there.
USER_DIRECTORY_DEFAULT = "/opt/user"
USER_DIRECTORY_ENVIRONMENT_VARIABLE = "HFLOW_USER_DIR"

# The mass-failure budget: generous enough that a handful of bad episodes
# never blocks a run, tight enough that systematic failure stays loud.
RUN_FAILURE_BUDGET_MINIMUM = 8
RUN_FAILURE_BUDGET_FRACTION = 0.01

# The default shard count for the batch lane, capped by item count.
DEFAULT_BATCH_COUNT_LIMIT = 4
DEFAULT_STAGGER_INTERVAL_S = 2.0


def run_failure_budget(total_episodes: int) -> int:
    """How many failures of one kind a run tolerates before failing loudly."""
    return max(RUN_FAILURE_BUDGET_MINIMUM, math.ceil(RUN_FAILURE_BUDGET_FRACTION * total_episodes))


def resolve_user_pipeline_path(pipeline_filename: str) -> str:
    """Where the runtime finds the user's pipeline file.

    Managed platforms cannot always mount ``user/`` at the default location;
    ``HFLOW_USER_DIR`` relocates it without re-rendering (DEPLOY.md names the
    per-platform value).
    """
    user_directory = os.environ.get(USER_DIRECTORY_ENVIRONMENT_VARIABLE) or USER_DIRECTORY_DEFAULT
    return user_directory.rstrip("/") + "/" + pipeline_filename


def load_pipeline_application(pipeline_path: str, app_variable: str) -> "App":
    """Import the user's pipeline file by path and resolve its App.

    This is the execution contract a deployment wraps its isolation around:
    the file is arbitrary user code, imported (executed) in the user venv,
    and must expose an ``hflow.App`` under ``app_variable``.

    A thin adapter over :func:`hflow.app.load_pipeline_application`, which
    owns the loading itself so this path cannot drift from the CLI's and the
    workspace UI's. The only thing that differs here is the error type: a
    task boundary has no caller that recovers, so a failure is
    ``RuntimeError`` rather than the ``ValueError`` the CLI catches to exit 2
    and the server catches to degrade to an unavailable pipeline page.
    """
    from hflow.app import load_pipeline_application as load_application

    try:
        return load_application(pipeline_path, app_variable)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def require_application_data_root(application: "App", expected_data_root: str) -> None:
    """Refuse to process when the App points anywhere but the runtime's root.

    Episode URIs resolve under the runtime's data root; an app pointing
    elsewhere would silently write into the task's own filesystem.
    """
    if str(application.data_root) != expected_data_root:
        raise RuntimeError(
            f"pipeline data_root must be {expected_data_root} inside the runtime; "
            f"it is {application.data_root}"
        )


def resolve_episode_reference(data_root: str, uri: str) -> "Path | str":
    """A conf URI (relative to the data root) as a processable reference."""
    if is_bucket_url(data_root):
        return data_root.rstrip("/") + "/" + uri.lstrip("/")
    return Path(data_root) / uri


def plan_stage_batches(
    uris: Sequence[str],
    *,
    mode: str,
    batch_count: int | None,
    data_root: str,
) -> list[PlannedStageBatch]:
    """Bin-pack uris into staggered batches; ``online`` is one immediate batch.

    The online lane is latency-first: one run per episode as it lands -- no
    batching, no stagger, ``batch_count`` ignored. Returns plain JSON-able
    dicts because the result crosses task (XCom) boundaries.
    """
    try:
        # Parse the conf string at this boundary; steps.IngestMode owns the
        # vocabulary (the master DAG validates against a render-time copy).
        ingest_mode = IngestMode(mode)
    except ValueError:
        raise ValueError(f"unknown mode {mode!r}; valid modes: {', '.join(IngestMode)}") from None
    if not uris:
        return []
    if ingest_mode is IngestMode.ONLINE:
        return [{"items": [str(uri) for uri in uris], "start_delay_s": 0.0}]
    data_root_storage = parse_storage_root(data_root)
    item_sizes = {str(uri): data_root_storage.file_size(str(uri)) for uri in uris}
    resolved_batch_count = (
        int(batch_count)
        if batch_count is not None
        else min(DEFAULT_BATCH_COUNT_LIMIT, len(item_sizes))
    )
    planned = plan_batches(
        item_sizes,
        batch_count=resolved_batch_count,
        stagger_interval_s=DEFAULT_STAGGER_INTERVAL_S,
    )
    return [{"items": list(batch.items), "start_delay_s": batch.start_delay_s} for batch in planned]


def process_stage_batch(
    application: "App",
    uris: Sequence[str],
    stage_name: str,
    orchestrator_run_id: str | None = None,
) -> StageBatchCounts:
    """Run one stage over every episode in a batch, counting outcomes.

    Per-episode crashes are counted as errors, never batch-fatal; the budget
    gates apply the run budget to the tallies, so mass failure stays loud
    while a stray bad episode never blocks a run.

    ``orchestrator_run_id`` is recorded on every episode this batch appends,
    so the catalog can be asked which orchestrated run produced a row.

    Named for the ROLE, not for Airflow, because this module is the one owner
    of run semantics across execution backends and nothing here should have to
    change to serve a second one. Whatever scheduled the work supplies its own
    identifier for the attempt and this records it verbatim; the generated
    Airflow DAGs hand over ``{{ run_id }}``, and a different backend would
    hand over whatever it calls the same thing. Optional, so a caller with no
    orchestrator at all (the dev loop, a bare script) keeps working and
    records NULL.
    """
    # Stage(stage_name) parses the conf string at this boundary: an unknown
    # stage is a loud ValueError before any episode is touched.
    stage = Stage(stage_name)
    data_root = str(application.data_root)
    counts: StageBatchCounts = {"processed": 0, "quarantined": 0, "errors": 0}
    # The labels and media gates read the episode's cataloged quarantine
    # state; opening that reader once per batch rather than once per episode
    # is the difference between one mirror sync and one per episode. Stages
    # that decide quarantine themselves never read it.
    with _batch_quarantine_history(application, stage) as quarantine_history:
        for uri in uris:
            try:
                episode_reference = resolve_episode_reference(data_root, str(uri))
                report = application.process(
                    episode_reference,
                    record=True,
                    stages={stage},
                    quarantine_history=quarantine_history,
                    orchestrator_run_id=orchestrator_run_id,
                )
            except Exception as error:
                traceback.print_exc()
                # A source that never canonicalized has no catalog row to be,
                # so without this the only trace of it is this traceback in
                # whatever log happened to be watching -- and the in-process
                # executor has no Airflow task log behind it at all.
                _record_failure_quietly(
                    application,
                    source_uri=str(uri),
                    stage=stage,
                    error=error,
                    orchestrator_run_id=orchestrator_run_id,
                )
                counts["errors"] += 1
                continue
            if report.has_errors:
                # app.process collects per-step diagnostics for the dev loop
                # and catalog, so step failures are explicit report outcomes
                # rather than escaping exceptions. They still count against
                # the runtime's infrastructure-error budget.
                counts["errors"] += 1
            elif report.quarantined:
                counts["quarantined"] += 1
            else:
                counts["processed"] += 1
    return counts


def run_stages_directly(
    application: "App",
    uris: Sequence[str],
    stages: Iterable[Stage],
    *,
    orchestrator_run_id: str | None = None,
) -> dict[Stage, StageBatchCounts]:
    """Run the stage graph in this process, with the runtime's own semantics.

    The no-scheduler backend: the same per-episode accounting and the same
    mass-failure budgets the generated DAGs apply, minus the scheduler. Nobody
    should reimplement the loop, which is why this lives beside
    :func:`process_stage_batch` rather than in the CLI -- a hand-rolled
    ``for uri in uris: app.process(uri)`` silently drops the budgets, and a
    run that quarantined half its input would then report success.

    Lane planning is deliberately absent. Bin-packing and staggered starts
    exist to spread work over workers that run concurrently; here there is one
    process, so every episode is one batch and a stagger would be pure delay.

    Stages run in stage-graph order (sync, then meta, then labels, then
    media), and each stage's gate applies before the next begins -- so a
    corpus that fails the error budget in ``sync`` never spends an ffmpeg
    decode pass on ``meta``. The budgets themselves are per-stage exactly as
    in the sub-DAGs: the quarantine budget only in ``meta``, because checks
    are what decide quarantine.
    """
    enabled_stages = set(stages)
    counts_by_stage: dict[Stage, StageBatchCounts] = {}
    for stage in Stage:
        if stage not in enabled_stages:
            continue
        counts = process_stage_batch(
            application, uris, stage.value, orchestrator_run_id=orchestrator_run_id
        )
        counts_by_stage[stage] = counts
        if stage is Stage.META:
            summarize_quarantine_budget([counts])
        else:
            summarize_error_budget([counts])
    return counts_by_stage


def _record_failure_quietly(
    application: "App",
    *,
    source_uri: str,
    stage: Stage,
    error: Exception,
    orchestrator_run_id: str | None,
) -> None:
    """Write one ledger row, and never let that write fail the run.

    The ledger exists to explain a failure, so it must not be able to cause
    one: an episode that failed has already been counted, and losing the
    explanation is strictly better than turning one bad recording into a dead
    batch.
    """
    from hflow.ingest_ledger import record_ingest_failure

    try:
        record_ingest_failure(
            application.workspace.catalog_root,
            source_uri=source_uri,
            stage=stage.value,
            pipeline_version=application.pipeline_version,
            error=error,
            orchestrator_run_id=orchestrator_run_id,
        )
    except Exception:  # pragma: no cover - defensive
        traceback.print_exc()


@contextmanager
def _batch_quarantine_history(
    application: "App", stage: Stage
) -> Iterator[QuarantineHistory | None]:
    """One catalog reader for a whole batch, or ``None`` where the stage
    never asks: only stages running without ``meta`` consult the catalog for
    quarantine, and ``meta`` itself decides it in memory."""
    if stage is Stage.META:
        yield None
        return
    with QuarantineHistory(application.workspace.catalog_root) as history:
        yield history


def _tally_batch_counts(batch_counts: Sequence[StageBatchCounts]) -> tuple[int, int, int]:
    """(total, quarantined, errors) across every batch's counts."""
    total = sum(
        counts["processed"] + counts["quarantined"] + counts["errors"] for counts in batch_counts
    )
    quarantined = sum(counts["quarantined"] for counts in batch_counts)
    errors = sum(counts["errors"] for counts in batch_counts)
    return total, quarantined, errors


def _raise_if_error_budget_exceeded(total: int, errors: int, budget: int) -> None:
    """The error gate both summaries share: over budget, or every episode errored."""
    if errors > budget or (errors and errors == total):
        raise RuntimeError(
            f"{errors} of {total} episodes had processing errors "
            f"(budget {budget}) -- infrastructure, not data; see the "
            "process_batch task logs"
        )


def summarize_error_budget(batch_counts: Sequence[StageBatchCounts]) -> dict[str, int]:
    """Fail loudly when processing errors exceed the run budget.

    Quarantine budgets live only in the meta stage (checks decide
    quarantine); here quarantined episodes just count toward the total, and
    a run where every episode errors always fails.
    """
    total, _quarantined, errors = _tally_batch_counts(batch_counts)
    budget = run_failure_budget(total)
    _raise_if_error_budget_exceeded(total, errors, budget)
    return {"total": total, "errors": errors, "budget": budget}


def summarize_quarantine_budget(batch_counts: Sequence[StageBatchCounts]) -> dict[str, int]:
    """Fail the run loudly on mass failure of either kind (the meta gate).

    Checks decide quarantine, so the quarantine budget lives only in the
    meta stage. The same budget also applies to per-episode exceptions and
    step errors, and a run where every episode errored always fails
    regardless of budget. (All-quarantined is deliberately NOT an automatic
    failure: quarantine is data policy, not infrastructure trouble, and the
    quarantine budget alone decides when it is loud.)
    """
    total, quarantined, errors = _tally_batch_counts(batch_counts)
    budget = run_failure_budget(total)
    if quarantined > budget:
        raise RuntimeError(
            f"quarantined {quarantined} of {total} episodes, over the budget "
            f"of {budget} -- quarantine is a tag, never a deletion; inspect "
            "the catalog's quarantine tags"
        )
    _raise_if_error_budget_exceeded(total, errors, budget)
    return {
        "total": total,
        "quarantined": quarantined,
        "errors": errors,
        "budget": budget,
    }
