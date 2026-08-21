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

Budget semantics (Dyna's mass-failure gates): a run tolerates up to
``max(8, ceil(1% of total))`` failures of a kind; checks decide quarantine,
so the quarantine budget applies only in the meta stage, and a run where
EVERY episode errored always fails regardless of budget.
"""

import importlib.util
import math
import os
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from hflow.batching import plan_batches
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
    """
    from hflow.app import App

    spec = importlib.util.spec_from_file_location("hflow_user_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load user pipeline at {pipeline_path}")
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    application = getattr(pipeline_module, app_variable, None)
    if not isinstance(application, App):
        raise RuntimeError(f"{pipeline_path} has no hflow.App named {app_variable!r}")
    return application


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

    The online lane (Figure 4) is latency-first: one run per episode as it
    lands -- no batching, no stagger, ``batch_count`` ignored. Returns plain
    JSON-able dicts because the result crosses task (XCom) boundaries.
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
    application: "App", uris: Sequence[str], stage_name: str
) -> StageBatchCounts:
    """Run one stage over every episode in a batch, counting outcomes.

    Per-episode crashes are counted as errors, never batch-fatal; the budget
    gates apply the run budget to the tallies, so mass failure stays loud
    while a stray bad episode never blocks a run.
    """
    data_root = str(application.data_root)
    counts: StageBatchCounts = {"processed": 0, "quarantined": 0, "errors": 0}
    for uri in uris:
        try:
            episode_reference = resolve_episode_reference(data_root, str(uri))
            # Stage(stage_name) parses the conf string at this boundary: an
            # unknown stage is a loud ValueError, counted as that episode's
            # error like any other infrastructure failure.
            report = application.process(episode_reference, record=True, stages={Stage(stage_name)})
        except Exception:
            traceback.print_exc()
            counts["errors"] += 1
            continue
        if report.has_errors:
            # app.process collects per-step diagnostics for the dev loop and
            # catalog, so step failures are explicit report outcomes rather
            # than escaping exceptions. They still count against the
            # runtime's infrastructure-error budget.
            counts["errors"] += 1
        elif report.quarantined:
            counts["quarantined"] += 1
        else:
            counts["processed"] += 1
    return counts


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
