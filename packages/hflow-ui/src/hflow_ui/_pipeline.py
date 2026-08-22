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
from hflow.manifest import PipelineManifest
from hflow.steps import Stage
from hflow.workspace import Workspace
from hflow_ui import _catalog, _connections
from hflow_ui._contract import (
    ObservedCheckVersion,
    PipelineResponse,
    PipelineStageLane,
    PipelineStepManifest,
    StaleSummary,
)
from hflow_ui._settings import UiSettings


@dataclass(frozen=True)
class PipelineState:
    """The one startup import's outcome: the live App, or why there is none."""

    application: App | None
    unavailable_detail: str | None

    @property
    def available(self) -> bool:
        return self.application is not None


def load_pipeline_state(pipeline_spec: str | None) -> PipelineState:
    """Run the one startup import and remember its outcome, whatever it is."""
    if pipeline_spec is None:
        return PipelineState(
            application=None,
            unavailable_detail=(
                "no --pipeline configured: relaunch `hflow ui` with "
                "--pipeline path/to/pipeline.py[:app] to serve the pipeline page"
            ),
        )
    try:
        application = import_pipeline_application(pipeline_spec)
    except ValueError as error:
        return PipelineState(application=None, unavailable_detail=str(error))
    return PipelineState(application=application, unavailable_detail=None)


def _stage_lanes(manifest: PipelineManifest) -> list[PipelineStageLane]:
    """Manifest steps grouped into the ingest stage lanes, in stage-graph order.

    The REAL stage semantics from ``hflow.steps``/``App.process``: registered
    checks run in the META stage ("checks + catalog registration"), user
    enrichments in LABELS ("Labels & artifacts"), while SYNC (the canonical
    transform plus derived channels) and MEDIA (the engine's contact-sheet
    step) are engine-owned lanes carrying no user-registered steps.
    """
    steps_by_stage: dict[Stage, list[PipelineStepManifest]] = {stage: [] for stage in Stage}
    steps_by_stage[Stage.META] = [
        PipelineStepManifest.from_step_manifest(step) for step in manifest.checks
    ]
    steps_by_stage[Stage.LABELS] = [
        PipelineStepManifest.from_step_manifest(step) for step in manifest.enrichments
    ]
    return [
        PipelineStageLane(
            stage=stage.value,
            engine_owned=stage in (Stage.SYNC, Stage.MEDIA),
            steps=steps_by_stage[stage],
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
        if state.application is None:
            raise HTTPException(status_code=409, detail=state.unavailable_detail)
        manifest = state.application.manifest()
        observed, stale = _observed_versions_and_stale(settings.data_root, state.application)
        return PipelineResponse(
            manifest=manifest.to_json_dict(),
            stages=_stage_lanes(manifest),
            observed=observed,
            stale=stale,
        )

    return router
