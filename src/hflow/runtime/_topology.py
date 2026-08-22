"""The generated ingest DAGs' shape, as data.

The templates in :mod:`hflow.runtime._templates` render the master DAG and its
four stage sub-DAGs; this module states the same task graph as inspectable
values, so a UI, a control plane, or a doc generator can draw what a bundle
will do without parsing generated Python. The templates remain the
implementation; :func:`ingest_dag_topology` is the description, and
``tests/test_runtime_topology.py`` pins the two together by checking every
task id here against the rendered source.

Two layers meet in these DAGs, and the distinction matters to anyone
rendering them:

- **Orchestration** (here): real dependency edges. The master validates the
  conf, then walks the stage chain, gating each stage on the run profile and
  waiting for its sub-DAG. Each sub-DAG plans batches, fans out over them,
  and closes with a budget gate.
- **User steps** (:class:`hflow.PipelineManifest`): registered checks and
  enrichments, which have NO dependency edges on each other. They run inside
  one ``process_batch`` task of the stage that owns their kind.
"""

from dataclasses import dataclass, field

from hflow.runtime._bundle import sub_dag_id_for_stage
from hflow.steps import RUN_PROFILES, Stage

# The master's first task: validates the trigger conf against the vocabulary
# baked into the bundle and publishes the enabled stage list.
RESOLVE_PROFILE_TASK_ID = "resolve_profile"

# Per stage, the master renders a skip gate and a deferred trigger.
STAGE_GATE_TASK_PREFIX = "enabled_"
STAGE_TRIGGER_TASK_PREFIX = "trigger_"

# Every sub-DAG's shape: bin-pack, fan out, then gate on the budget.
PLAN_TASK_ID = "plan"
PROCESS_BATCH_TASK_ID = "process_batch"
QUARANTINE_BUDGET_GATE_TASK_ID = "quarantine_budget_gate"
ERROR_BUDGET_GATE_TASK_ID = "error_budget_gate"


def budget_gate_task_id(stage: Stage) -> str:
    """The gate task closing one stage's sub-DAG.

    Meta owns the quarantine budget because it is the stage that runs checks
    and therefore the only one that can quarantine; the others gate on the
    error budget alone. Mirrors the template selection in
    ``_bundle.render_sub_dag_source``, exhaustively and for the same reason:
    a new stage must make its author choose a gate here too, rather than
    inheriting the error budget from an ``else``.
    """
    match stage:
        case Stage.META:
            return QUARANTINE_BUDGET_GATE_TASK_ID
        case Stage.SYNC | Stage.LABELS | Stage.MEDIA:
            return ERROR_BUDGET_GATE_TASK_ID


@dataclass(frozen=True)
class DagTaskNode:
    """One task in a generated DAG."""

    task_id: str
    # What the task does, in the vocabulary a reader of the UI has -- not
    # Airflow's operator names.
    summary: str
    # True for the dynamically mapped task: one instance per planned batch,
    # so a renderer draws it as a fan-out rather than a single box.
    mapped: bool = False
    # True when the task defers (releases its worker slot while waiting);
    # a renderer should say "waiting", never "stalled".
    deferred: bool = False


@dataclass(frozen=True)
class DagTopology:
    """One generated DAG: its tasks and the edges between them."""

    dag_id: str
    tasks: tuple[DagTaskNode, ...]
    # (upstream task id, downstream task id) pairs, declaration order.
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StageTopology:
    """One stage sub-DAG, plus how the master reaches it."""

    stage: Stage
    dag: DagTopology
    # The master's tasks that gate and trigger this stage.
    gate_task_id: str
    trigger_task_id: str
    # The run profiles that enable this stage; a profile outside this set
    # skips the stage at its gate.
    enabling_profiles: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IngestTopology:
    """The whole rendered bundle's task graph: master plus stage sub-DAGs."""

    master: DagTopology
    stages: tuple[StageTopology, ...]


def _sub_dag_topology(master_dag_id: str, stage: Stage) -> DagTopology:
    gate_task_id = budget_gate_task_id(stage)
    return DagTopology(
        dag_id=sub_dag_id_for_stage(master_dag_id, stage),
        tasks=(
            DagTaskNode(
                task_id=PLAN_TASK_ID,
                summary="bin-pack the trigger's uris into batches (online: one immediate batch)",
            ),
            DagTaskNode(
                task_id=PROCESS_BATCH_TASK_ID,
                summary=f"run the {stage.value} stage over every episode in one batch",
                mapped=True,
            ),
            DagTaskNode(
                task_id=gate_task_id,
                summary=(
                    "fail the run when quarantines exceed the budget"
                    if gate_task_id == QUARANTINE_BUDGET_GATE_TASK_ID
                    else "fail the run when errors exceed the budget"
                ),
            ),
        ),
        edges=(
            (PLAN_TASK_ID, PROCESS_BATCH_TASK_ID),
            (PROCESS_BATCH_TASK_ID, gate_task_id),
        ),
    )


def _enabling_profiles(stage: Stage) -> tuple[str, ...]:
    return tuple(sorted(name for name, stages in RUN_PROFILES.items() if stage in stages))


def ingest_dag_topology(master_dag_id: str) -> IngestTopology:
    """The task graph a bundle rendered for ``master_dag_id`` will run.

    Derived from the same facts the renderer uses (the stage vocabulary, the
    sub-DAG id derivation, and the per-stage gate choice), so the description
    cannot drift from the bundle as long as the pinning test passes.
    """
    master_tasks: list[DagTaskNode] = [
        DagTaskNode(
            task_id=RESOLVE_PROFILE_TASK_ID,
            summary="validate the run profile and mode; publish the enabled stages",
        )
    ]
    master_edges: list[tuple[str, str]] = []
    stages: list[StageTopology] = []

    previous_trigger_task_id: str | None = None
    for stage in Stage:
        gate_task_id = f"{STAGE_GATE_TASK_PREFIX}{stage.value}"
        trigger_task_id = f"{STAGE_TRIGGER_TASK_PREFIX}{stage.value}"
        master_tasks.append(
            DagTaskNode(
                task_id=gate_task_id,
                summary=f"skip {stage.value} when the run profile disables it",
            )
        )
        master_tasks.append(
            DagTaskNode(
                task_id=trigger_task_id,
                summary=f"trigger the {stage.value} sub-DAG and wait for it to finish",
                deferred=True,
            )
        )
        # resolve_profile feeds every gate its enabled-stage list, and the
        # previous stage's trigger must finish before the next gate opens --
        # that pair of edges is what makes the stages a chain, not a fan-out.
        master_edges.append((RESOLVE_PROFILE_TASK_ID, gate_task_id))
        master_edges.append((gate_task_id, trigger_task_id))
        if previous_trigger_task_id is not None:
            master_edges.append((previous_trigger_task_id, gate_task_id))
        previous_trigger_task_id = trigger_task_id

        stages.append(
            StageTopology(
                stage=stage,
                dag=_sub_dag_topology(master_dag_id, stage),
                gate_task_id=gate_task_id,
                trigger_task_id=trigger_task_id,
                enabling_profiles=_enabling_profiles(stage),
            )
        )

    return IngestTopology(
        master=DagTopology(
            dag_id=master_dag_id,
            tasks=tuple(master_tasks),
            edges=tuple(master_edges),
        ),
        stages=tuple(stages),
    )
