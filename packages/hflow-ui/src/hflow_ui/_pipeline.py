"""The pipeline page API: the startup-imported App described over the catalog.

``--pipeline path/to/pipeline.py[:app]`` names a Python file this server
imports -- EXECUTES -- exactly once at startup via the shared
:func:`hflow.import_pipeline_application` seam (the one owner of the "address
a pipeline by file" contract, used by the CLI too; producing a manifest
requires the live functions, because step versions are content hashes of
them). The operator opts into running their own pipeline code by passing the
flag; an import failure never crashes the server -- the error string is
remembered, the config capability reports false, and /api/v1/pipeline answers
409 with the stored reason.
"""

from dataclasses import dataclass

from fastapi import APIRouter, HTTPException

from hflow import App, import_pipeline_application
from hflow.curation import stale_episodes
from hflow.format import EPISODE_FORMAT_VERSION
from hflow.manifest import PipelineManifest, StepManifest
from hflow.steps import Stage
from hflow.workspace import Workspace
from hflow_ui import _catalog, _connections
from hflow_ui._contract import (
    ObservedCheckVersion,
    PipelineResponse,
    PipelineStageLane,
    PipelineStepManifest,
    StaleSummary,
    StepTier,
)
from hflow_ui._settings import UiSettings


@dataclass(frozen=True)
class PipelineLoaded:
    """The one startup import produced a live App."""

    application: App


@dataclass(frozen=True)
class PipelineUnavailable:
    """No App for this launch, and exactly why."""

    detail: str


# Two states, never both and never neither -- the same sum ``_runtime`` uses
# for its resolution, so the two capabilities behind the same 409 refusal are
# modelled the same way and the refusal's detail cannot be null.
PipelineState = PipelineLoaded | PipelineUnavailable


def load_pipeline_state(pipeline_spec: str | None) -> PipelineState:
    """Run the one startup import and remember its outcome, whatever it is."""
    if pipeline_spec is None:
        return PipelineUnavailable(
            detail=(
                "no --pipeline configured: relaunch `hflow ui` with "
                "--pipeline path/to/pipeline.py[:app] to serve the pipeline page"
            )
        )
    try:
        return PipelineLoaded(application=import_pipeline_application(pipeline_spec))
    except ValueError as error:
        return PipelineUnavailable(detail=str(error))


def registered_step_tier(step: StepManifest) -> StepTier:
    """Which cheap-first tier this step runs in (1 first, 2 second).

    Mirrors :meth:`hflow.App._ordered_checks` and ``_ordered_enrichments``
    EXACTLY: both sort on ``bool(requires) or uses is not None``, so tier 2 is
    precisely the steps declaring a required channel or an endpoint alias.
    Within a tier there is no ordering at all -- registration order is what
    the stable sort preserves, not a dependency.

    The rule ideally belongs in the SDK -- a ``tier`` on
    ``hflow.manifest.StepManifest`` that ``App`` sorts on and ``hflow
    manifest`` renders, so the CLI could answer "in what order do my steps
    run?" too. Until it lives there, this is this package's ONE copy: both
    endpoints project from :func:`registered_steps_by_stage` rather than
    restating the expression a second time.
    """
    return 2 if (bool(step.requires) or step.uses is not None) else 1


def _in_execution_order(
    steps: tuple[StepManifest, ...],
) -> tuple[tuple[StepManifest, StepTier], ...]:
    # Stable sort on the tier alone: the same sort App._ordered_checks makes,
    # so the served order IS the execution order.
    return tuple(
        (step, registered_step_tier(step)) for step in sorted(steps, key=registered_step_tier)
    )


def registered_steps_by_stage(
    manifest: PipelineManifest,
) -> dict[Stage, tuple[tuple[StepManifest, StepTier], ...]]:
    """Which registered steps run in which stage, in the order they run.

    The ONE owner of that mapping for this package: the pipeline page's lanes
    and the graph's per-stage user steps are the same steps in the same order,
    differing only in whether the payload carries the tier -- so the two pages
    can never show one pipeline as two.

    Stage ownership is the engine's (``hflow.steps``/``App.process``):
    registered checks run in META ("checks + catalog registration"), user
    enrichments in LABELS ("Labels & artifacts"), while SYNC (the canonical
    transform plus derived channels) and MEDIA (the engine's contact-sheet
    step) are engine-owned lanes carrying no user-registered steps.
    """
    steps_by_stage: dict[Stage, tuple[tuple[StepManifest, StepTier], ...]] = dict.fromkeys(
        Stage, ()
    )
    steps_by_stage[Stage.META] = _in_execution_order(manifest.checks)
    steps_by_stage[Stage.LABELS] = _in_execution_order(manifest.enrichments)
    return steps_by_stage


def _stage_lanes(manifest: PipelineManifest) -> list[PipelineStageLane]:
    """The stage lanes of the pipeline page, in stage-graph order."""
    steps_by_stage = registered_steps_by_stage(manifest)
    return [
        PipelineStageLane(
            stage=stage,
            engine_owned=stage in (Stage.SYNC, Stage.MEDIA),
            steps=[
                PipelineStepManifest.from_step_manifest(step)
                for step, _tier in steps_by_stage[stage]
            ],
        )
        for stage in Stage
    ]


def _observed_versions_and_stale(
    data_root: str, application: App
) -> tuple[list[ObservedCheckVersion], StaleSummary | None]:
    """What the catalog has SEEN of this pipeline: per-(check, version)
    first/last-seen aggregates, plus the stale count against the App's
    current versions. A workspace with no catalog yet has observed nothing
    and its staleness is unknowable -- ([], None), not an error."""
    with _connections.opened_workspace_connection_or_none(data_root) as connection:
        if connection is None:
            return [], None
        observed = [
            ObservedCheckVersion.model_validate(row)
            for row in _catalog.fetched_json_safe_rows(
                connection.execute(
                    "SELECT check_name, check_version, "
                    f"{_catalog.utc_iso_text('min(recorded_at)', 'first_seen')}, "
                    f"{_catalog.utc_iso_text('max(recorded_at)', 'last_seen')}, "
                    "count(*) AS run_count "
                    "FROM check_runs GROUP BY check_name, check_version "
                    "ORDER BY check_name, check_version"
                )
            )
        ]
    current_pipeline_version = application.pipeline_version
    try:
        # A pipeline defines the whole current target, format version
        # included -- the same pairing the CLI's `hflow stale --pipeline` uses.
        stale = stale_episodes(
            Workspace.parse(data_root).catalog_root,
            pipeline_version=current_pipeline_version,
            schema_version=EPISODE_FORMAT_VERSION,
        )
    except (FileNotFoundError, ValueError):
        return observed, None
    return observed, StaleSummary(pipeline_version=current_pipeline_version, count=len(stale))


def create_pipeline_router(settings: UiSettings, state: PipelineState) -> APIRouter:
    """The pipeline route, closed over the one startup import's outcome."""
    router = APIRouter(prefix="/api/v1")

    @router.get("/pipeline")
    def read_pipeline() -> PipelineResponse:
        if isinstance(state, PipelineUnavailable):
            raise HTTPException(status_code=409, detail=state.detail)
        manifest = state.application.manifest()
        observed, stale = _observed_versions_and_stale(settings.data_root, state.application)
        return PipelineResponse(
            manifest=manifest.to_json_dict(),
            stages=_stage_lanes(manifest),
            observed=observed,
            stale=stale,
        )

    return router
