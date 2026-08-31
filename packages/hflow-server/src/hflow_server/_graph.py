"""The visualization API: the ingest DAG's shape, and one run's live state.

Two nested layers meet on these endpoints, and neither may be drawn as the
other:

- **Orchestration** -- a real DAG with real edges, served straight from
  :func:`hflow.runtime.ingest_dag_topology` (the library's description of the
  DAGs ``hflow up`` renders, pinned to the templates by the core suite). The
  master resolves the run profile, then walks the stage chain gating and
  triggering each sub-DAG; every sub-DAG plans batches, fans ``process_batch``
  out over them, and closes on a budget gate.
- **User steps** -- the registered checks and enrichments of a ``--pipeline``
  App, which have NO dependency edges on each other. They all run INSIDE one
  ``process_batch`` task of the stage that owns their kind, ordered only by
  the engine's two-tier cheap-first policy (:meth:`hflow.App._ordered_checks`:
  a step declaring ``requires`` or ``uses`` runs in the second tier). Drawing
  arrows between them would be a lie; the payload states the tiers instead.

The one real cross-step edge is the quarantine gate, and it is served as its
own object rather than as an edge in either graph.

Both endpoints degrade instead of failing: the pipeline graph answers with
``dag_ids_known``/``steps_known`` flags when no runtime or no pipeline is
addressed, and the run graph refuses with the runs monitor's 409/502 idiom.
"""

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

from fastapi import APIRouter, HTTPException

from hflow.app import MEDIA_CONTACT_SHEET_STEP_NAME
from hflow.manifest import PipelineManifest
from hflow.runtime import (
    AirflowClient,
    AirflowClientError,
    AirflowDagRun,
    AirflowTaskInstance,
    DagTaskNode,
    DagTopology,
    IngestTopology,
    StageTopology,
    ingest_dag_topology,
)
from hflow.steps import Stage
from hflow_server._contract import (
    DagTaskNodePayload,
    DagTopologyPayload,
    MappedFanOutSummary,
    PipelineEngineStep,
    PipelineGraphResponse,
    PipelineGraphStage,
    PipelineUserStep,
    QuarantineGate,
    RunGraphMaster,
    RunGraphResponse,
    RunGraphStage,
    RunTaskInstance,
    StageRunMatch,
)
from hflow_server._pipeline import PipelineLoaded, PipelineState, registered_steps_by_stage
from hflow_server._runtime import (
    ResolvedRuntime,
    RuntimeResolver,
    airflow_failure_refusal,
    optional_string,
    resolved_runtime_or_refuse,
)

# The display copy for the four stages. Restated here (rather than imported
# from hflow.runtime._bundle's STAGE_TITLES/STAGE_DESCRIPTIONS, which are
# private and worded for Airflow's own UI) so the browser never hardcodes it:
# the thin-client rule applies to prose too.
_STAGE_TITLES: dict[Stage, str] = {
    Stage.SYNC: "Transform & sync",
    Stage.META: "Metadata",
    Stage.LABELS: "Labels & artifacts",
    Stage.MEDIA: "Media",
}
_STAGE_DESCRIPTIONS: dict[Stage, str] = {
    Stage.SYNC: "canonical transform + derived channels (critical path)",
    Stage.META: "checks + catalog registration",
    Stage.LABELS: "enrichments (non-critical)",
    Stage.MEDIA: "derived media artifacts",
}

# The master id shown when no runtime is addressed: the DAGs do not exist
# yet, so the graph is drawn under a display-only name (the pipeline's own
# name when one is imported, else this) and ``dag_ids_known`` is false.
DISPLAY_ONLY_MASTER_DAG_ID = "ingest"

_DAG_ID_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")

# How many of a stage sub-DAG's runs the run-graph heuristic looks at.
_STAGE_RUN_SEARCH_LIMIT = 25

# How long after a master run ENDED a stage run may still start and count as
# its own. Normally zero is enough (the master defers until each stage run
# finishes), but a master that fails, times out, or is cleared the moment
# after firing a trigger ends before the run it just caused appears -- and the
# two timestamps come from different components' clocks. Generous enough to
# cover that, far short of the gap between two ingests.
_STAGE_RUN_START_GRACE_AFTER_MASTER_END = timedelta(minutes=5)

# Airflow reports a task instance that has not been scheduled yet with a null
# state; the mapped fan-out summary needs a key for those.
_UNSET_TASK_STATE = "no_status"


def _dag_task_node_payload(node: DagTaskNode) -> DagTaskNodePayload:
    return DagTaskNodePayload(
        task_id=node.task_id,
        summary=node.summary,
        mapped=node.mapped,
        deferred=node.deferred,
    )


def _dag_topology_payload(topology: DagTopology) -> DagTopologyPayload:
    return DagTopologyPayload(
        dag_id=topology.dag_id,
        tasks=[_dag_task_node_payload(node) for node in topology.tasks],
        edges=[(upstream, downstream) for upstream, downstream in topology.edges],
    )


def _display_master_dag_id(pipeline_name: str | None) -> str:
    """A stand-in master id for a workspace with no rendered bundle.

    Never presented as real: the response's ``dag_ids_known`` is false, and
    the sub-DAG ids derived from it are display-only too. The real id is
    ``<pipeline file stem>_ingest``, which only a rendered bundle knows.
    """
    if pipeline_name is None:
        return DISPLAY_ONLY_MASTER_DAG_ID
    sanitized = _DAG_ID_UNSAFE_CHARACTERS.sub("-", pipeline_name).strip("-")
    return sanitized or DISPLAY_ONLY_MASTER_DAG_ID


def _user_steps(stage: Stage, manifest: PipelineManifest | None) -> list[PipelineUserStep]:
    """The registered steps running inside this stage's ``process_batch``.

    Which stage owns which steps, and the order they run in, both come from
    :func:`hflow_server._pipeline.registered_steps_by_stage` -- the package's one
    owner of that mapping -- so this lane and the pipeline page's lane are
    the same steps in the same order, and only ``tier`` is served here.
    """
    if manifest is None:
        return []
    return [
        PipelineUserStep.from_step_manifest_in_tier(step, tier)
        for step, tier in registered_steps_by_stage(manifest)[stage]
    ]


def _engine_steps(stage: Stage, manifest: PipelineManifest | None) -> list[PipelineEngineStep]:
    """The engine's own work inside this stage's ``process_batch``.

    Not registrations -- these are what ``App.process`` does around the user's
    steps, and no manifest lists them: the canonical transform (sync), the
    catalog append (meta), and the contact-sheet renderer (media).
    """
    if stage is Stage.SYNC:
        overridden = manifest is not None and manifest.has_transform_override
        derived_channel_count = len(manifest.derived_channels) if manifest is not None else 0
        summary = (
            "rewrite the source recording into a canonical MCAP and publish it"
            if not overridden
            else "rewrite the source recording with this pipeline's transform override "
            "and publish it"
        )
        if derived_channel_count:
            summary += (
                f"; computes {derived_channel_count} registered derived "
                f"channel{'s' if derived_channel_count != 1 else ''} over the source"
            )
        return [PipelineEngineStep(name="canonical transform", summary=summary)]
    if stage is Stage.META:
        return [
            PipelineEngineStep(
                name="catalog registration",
                summary="append this run's episode row and every step's evidence "
                "(check runs, measurements, intervals, tags) to the catalog",
            )
        ]
    if stage is Stage.MEDIA:
        return [
            PipelineEngineStep(
                name=MEDIA_CONTACT_SHEET_STEP_NAME,
                summary="render one contact sheet per camera and record it as a "
                "catalog artifact; absent when the episode has no cameras",
            )
        ]
    return []


# What a failed critical check actually does in App.process: the episode is
# tagged (never deleted), the meta stage skips its REMAINING checks, and every
# enrichment in labels and media is recorded as skipped.
_QUARANTINE_GATE_EXPLANATION = (
    "a False verdict from a critical check quarantines the episode: meta skips its "
    "remaining checks, and every enrichment in the labels and media stages is recorded "
    "as skipped. Quarantine is a tag, never a deletion."
)
_NO_CRITICAL_CHECKS_EXPLANATION = (
    "this pipeline registers no critical checks, so no check can quarantine an episode. "
    "A critical check's False verdict would make meta skip its remaining checks and every "
    "enrichment in the labels and media stages."
)


def _quarantine_gate(manifest: PipelineManifest | None) -> QuarantineGate | None:
    """The one real edge between user steps, or null when no pipeline is known."""
    if manifest is None:
        return None
    critical_step_names = [step.name for step in manifest.checks if step.critical]
    return QuarantineGate(
        from_stage=Stage.META,
        to_stages=[Stage.LABELS, Stage.MEDIA],
        critical_step_names=critical_step_names,
        explanation=(
            _QUARANTINE_GATE_EXPLANATION if critical_step_names else _NO_CRITICAL_CHECKS_EXPLANATION
        ),
    )


def _stage_graph(
    stage_topology: StageTopology, manifest: PipelineManifest | None
) -> PipelineGraphStage:
    stage = stage_topology.stage
    return PipelineGraphStage(
        stage=stage,
        title=_STAGE_TITLES[stage],
        description=_STAGE_DESCRIPTIONS[stage],
        gate_task_id=stage_topology.gate_task_id,
        trigger_task_id=stage_topology.trigger_task_id,
        enabling_profiles=list(stage_topology.enabling_profiles),
        dag=_dag_topology_payload(stage_topology.dag),
        engine_steps=_engine_steps(stage, manifest),
        user_steps=_user_steps(stage, manifest),
    )


def _parsed_timestamp(value: object) -> datetime | None:
    """One Airflow ISO-8601 timestamp as an aware datetime, or None.

    Airflow renders UTC as a trailing ``Z``; it is normalized here rather than
    left to ``fromisoformat``'s version-dependent tolerance, and a naive value
    is read as UTC so a comparison against another timestamp never raises.
    Anything unparseable is None -- a timestamp this build cannot read must
    not fail the request.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _duration_seconds(instance: AirflowTaskInstance) -> float | None:
    """One task instance's wall duration, computed here rather than trusted.

    Airflow's own ``duration`` field is the fallback for an instance whose
    timestamps this build cannot parse.
    """
    started_at = _parsed_timestamp(instance.start_date)
    ended_at = _parsed_timestamp(instance.end_date)
    if started_at is not None and ended_at is not None:
        return (ended_at - started_at).total_seconds()
    if instance.duration is not None and isfinite(instance.duration):
        return instance.duration
    return None


def _task_instance(instance: AirflowTaskInstance) -> RunTaskInstance:
    """One Airflow task instance reduced to what the graph draws."""
    return RunTaskInstance(
        task_id=instance.task_id,
        state=instance.state,
        start_date=instance.start_date,
        end_date=instance.end_date,
        queued_at=instance.queued_at,
        try_number=instance.try_number,
        map_index=instance.map_index,
        duration_s=_duration_seconds(instance),
    )


def _sorted_task_instances(
    instances: list[AirflowTaskInstance], topology: DagTopology
) -> list[RunTaskInstance]:
    """Task instances in TOPOLOGY order (then by map index), not API order."""
    topology_positions = {node.task_id: index for index, node in enumerate(topology.tasks)}
    unknown_task_position = len(topology_positions)
    return sorted(
        (_task_instance(instance) for instance in instances),
        key=lambda task: (
            topology_positions.get(task.task_id or "", unknown_task_position),
            task.task_id or "",
            task.map_index,
        ),
    )


def _mapped_summary(
    tasks: list[RunTaskInstance], stage_topology: StageTopology
) -> MappedFanOutSummary | None:
    """The fan-out's counts: how many mapped instances are in which state.

    The mapped task id comes from the topology (the node flagged ``mapped``),
    so this never restates a task name the library owns. Every instance of
    that task lands in exactly one ``by_state`` bucket -- an unscheduled one
    under ``no_status`` -- which is what makes the served summary complete
    enough that a client never has to recount the raw instances.
    """
    mapped_task_ids = [node.task_id for node in stage_topology.dag.tasks if node.mapped]
    if not mapped_task_ids:
        return None
    # Every generated stage sub-DAG has exactly one mapped node
    # (``process_batch``); a topology that grows a second one needs a summary
    # per mapped task, not a silently truncated one.
    mapped_task_id = mapped_task_ids[0]
    mapped_instances = [task for task in tasks if task.task_id == mapped_task_id]
    if not mapped_instances:
        return None
    state_counts = Counter(task.state or _UNSET_TASK_STATE for task in mapped_instances)
    return MappedFanOutSummary(
        task_id=mapped_task_id,
        # Before the fan-out expands, Airflow reports ONE instance with
        # map_index -1; it is counted, because "1 unexpanded instance" is the
        # truth at that moment.
        total=len(mapped_instances),
        by_state=dict(sorted(state_counts.items())),
    )


@dataclass(frozen=True)
class _MatchedStageRun:
    """The stage run a master run most plausibly triggered, and how it matched."""

    run: AirflowDagRun
    match: StageRunMatch


@dataclass(frozen=True)
class _MasterRunWindow:
    """When a master run was live: the interval its stage runs must start in.

    ``ended_at`` is None while the run is still going, which leaves the window
    open-ended on the right -- the only case where "no upper bound" is true.
    """

    started_at: datetime
    ended_at: datetime | None

    def contains_stage_run_start(self, started_at: datetime) -> bool:
        if started_at < self.started_at:
            return False
        if self.ended_at is None:
            return True
        return started_at <= self.ended_at + _STAGE_RUN_START_GRACE_AFTER_MASTER_END


def _master_run_window(master_run: AirflowDagRun) -> _MasterRunWindow | None:
    """One master run's live interval, or None when it has not started yet."""
    started_at = _parsed_timestamp(master_run.start_date)
    if started_at is None:
        return None
    return _MasterRunWindow(
        started_at=started_at, ended_at=_parsed_timestamp(master_run.end_date)
    )


def _matched_stage_run(
    stage_runs: list[AirflowDagRun], window: _MasterRunWindow | None
) -> _MatchedStageRun | None:
    """The EARLIEST run of one stage sub-DAG that started inside the master's window.

    The master triggers each stage with a deferring
    ``TriggerDagRunOperator(wait_for_completion=True)`` and chains the stages
    in order (``hflow.runtime`` renders them that way), so a stage run the
    master caused always STARTS while the master run is still live. Bounding
    the search by the master's own end is therefore not a guess, and it is
    what stops an old master run from adopting an unrelated stage run that
    happens to be newer -- the stage lanes only ever look back
    ``_STAGE_RUN_SEARCH_LIMIT`` runs, so without the bound every candidate
    qualified and the newest won.

    Earliest-in-window, not newest: when two master runs overlap, this
    master's own stage run is the first one after its start, while the newest
    is biased toward the other master's. The cost is that a stage triggered
    twice inside ONE master run (a retried trigger task) shows the first
    attempt -- accepted, because preferring the newest is exactly what let an
    unrelated run be adopted.

    HONEST LIMITATION, restated in the payload as ``"match": "heuristic"``:
    the master lets Airflow mint the sub-DAG's run id and forwards a conf that
    carries no back-reference, so the API offers nothing that ties a stage run
    to the master run that triggered it. Two master runs whose windows OVERLAP
    can still be attributed the same stage run. A master run that has not
    started yet (no ``start_date``) matches nothing rather than guessing.
    """
    if window is None:
        return None
    earliest_run: AirflowDagRun | None = None
    earliest_started_at: datetime | None = None
    for run in stage_runs:
        started_at = _parsed_timestamp(run.start_date)
        if started_at is None or not window.contains_stage_run_start(started_at):
            continue
        if earliest_started_at is None or started_at < earliest_started_at:
            earliest_run, earliest_started_at = run, started_at
    if earliest_run is None:
        return None
    return _MatchedStageRun(run=earliest_run, match="heuristic")


def _empty_stage_graph(stage_topology: StageTopology) -> RunGraphStage:
    """A stage that never ran for this master run: explicit nulls, not omissions."""
    return RunGraphStage(
        stage=stage_topology.stage,
        dag_id=stage_topology.dag.dag_id,
        dag_run_id=None,
        state=None,
        match=None,
        tasks=[],
        mapped_summary=None,
    )


def create_graph_router(pipeline_state: PipelineState, resolver: RuntimeResolver) -> APIRouter:
    """The visualization routes, closed over one launch's pipeline and runtime.

    Read-only throughout, so unlike the other routers these need no settings:
    the pipeline comes from the one startup import and the runtime from the
    shared resolver.
    """
    router = APIRouter(prefix="/api/v1")

    def stage_task_instances(
        client: AirflowClient, dag_id: str, dag_run_id: str
    ) -> list[dict[str, Any]]:
        try:
            return client.task_instances(dag_id, dag_run_id)
        except AirflowClientError:
            # A stage sub-DAG that vanished (or a run Airflow expired) leaves
            # that lane without task detail; the master's own state -- the
            # page's point -- is already in hand, so this is a thinner
            # drawing, not a failed request.
            return []

    @router.get("/pipeline/graph")
    def read_pipeline_graph() -> PipelineGraphResponse:
        """The merged picture: the DAG topology plus the pipeline's user steps.

        Three degraded states, each explicit rather than an error: no runtime
        addressed (``dag_ids_known: false``, display-only ids), no
        ``--pipeline`` (``steps_known: false``, no user steps and no
        quarantine gate), and both at once -- the common first-run case.
        """
        resolution = resolver.resolve()
        dag_ids_known = isinstance(resolution, ResolvedRuntime)
        application = (
            pipeline_state.application if isinstance(pipeline_state, PipelineLoaded) else None
        )
        master_dag_id = (
            resolution.dag_id
            if isinstance(resolution, ResolvedRuntime)
            else _display_master_dag_id(application.name if application is not None else None)
        )
        manifest = application.manifest() if application is not None else None
        topology: IngestTopology = ingest_dag_topology(master_dag_id)
        return PipelineGraphResponse(
            dag_ids_known=dag_ids_known,
            steps_known=manifest is not None,
            master=_dag_topology_payload(topology.master),
            stages=[_stage_graph(stage_topology, manifest) for stage_topology in topology.stages],
            quarantine_gate=_quarantine_gate(manifest),
        )

    @router.get("/runtime/runs/{dag_run_id}/graph")
    def read_run_graph(dag_run_id: str) -> RunGraphResponse:
        """One master run's live state over the same topology.

        The master run is addressed directly; each stage's sub-DAG run is
        resolved by the documented heuristic in :func:`_matched_stage_run`.
        """
        runtime = resolved_runtime_or_refuse(resolver)
        topology = ingest_dag_topology(runtime.dag_id)
        try:
            master_run = runtime.client.dag_run(runtime.dag_id, dag_run_id)
        except AirflowClientError as error:
            if error.status == 404:
                # A definitively unknown run is a missing resource, not an
                # upstream failure -- and the detail names only ids the
                # caller already sent.
                raise HTTPException(
                    status_code=404,
                    detail=f"no run {dag_run_id!r} of dag {runtime.dag_id!r}",
                ) from error
            raise airflow_failure_refusal(error, source=runtime.source) from error
        try:
            master_instances = runtime.client.task_instances(runtime.dag_id, dag_run_id)
        except AirflowClientError as error:
            raise airflow_failure_refusal(error, source=runtime.source) from error
        master_window = _master_run_window(master_run)

        stages: list[RunGraphStage] = []
        for stage_topology in topology.stages:
            stage_dag_id = stage_topology.dag.dag_id
            try:
                stage_runs = runtime.client.dag_runs(
                    stage_dag_id, limit=_STAGE_RUN_SEARCH_LIMIT, order_by="-id"
                )
            except AirflowClientError:
                # An unregistered stage sub-DAG (a partial profile, or a
                # bundle mid-render) is a stage that never ran here.
                stage_runs = []
            matched = _matched_stage_run(stage_runs, master_window)
            if matched is None:
                stages.append(_empty_stage_graph(stage_topology))
                continue
            stage_run_id = matched.run.dag_run_id
            stage_tasks = (
                _sorted_task_instances(
                    stage_task_instances(runtime.client, stage_dag_id, stage_run_id),
                    stage_topology.dag,
                )
                if stage_run_id is not None
                else []
            )
            stages.append(
                RunGraphStage(
                    stage=stage_topology.stage,
                    dag_id=stage_dag_id,
                    dag_run_id=stage_run_id,
                    state=matched.run.state,
                    match=matched.match,
                    tasks=stage_tasks,
                    mapped_summary=_mapped_summary(stage_tasks, stage_topology),
                )
            )

        return RunGraphResponse(
            master=RunGraphMaster(
                dag_run_id=master_run.dag_run_id or dag_run_id,
                state=master_run.state,
                tasks=_sorted_task_instances(master_instances, topology.master),
            ),
            stages=stages,
        )

    return router
