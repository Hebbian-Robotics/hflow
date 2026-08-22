"""The described topology must match the DAGs the renderer actually writes.

``hflow.runtime.ingest_dag_topology`` exists so a UI can draw a bundle's task
graph without parsing generated Python. That is only safe while the two agree,
so these tests read the rendered source and check every task id and edge the
description claims.
"""

from itertools import pairwise
from pathlib import Path

import pytest

from hflow.runtime import (
    RuntimeConfig,
    ingest_dag_topology,
    render_bundle,
    sub_dag_id_for_stage,
)
from hflow.runtime._topology import (
    PLAN_TASK_ID,
    PROCESS_BATCH_TASK_ID,
    RESOLVE_PROFILE_TASK_ID,
    budget_gate_task_id,
)
from hflow.steps import Stage

MASTER_DAG_ID = "kitchen_ingest"


@pytest.fixture(scope="module")
def rendered_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle_dir = tmp_path_factory.mktemp("topology-bundle")
    pipeline_file = bundle_dir / "pipeline.py"
    pipeline_file.write_text("import hflow\n\napp = hflow.App('kitchen')\n")
    paths = render_bundle(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=bundle_dir / "data"),
        bundle_dir / "runtime",
    )
    return paths.bundle_dir / "dags"


def _dag_source(dags_directory: Path, file_name: str) -> str:
    return (dags_directory / file_name).read_text()


def test_master_task_ids_appear_in_the_rendered_master_dag(rendered_bundle: Path) -> None:
    source = _dag_source(rendered_bundle, "ingest.py")
    topology = ingest_dag_topology(MASTER_DAG_ID)

    assert f"def {RESOLVE_PROFILE_TASK_ID}(" in source
    # The gate and trigger ids are built from f-strings in the template, so the
    # rendered source carries one shape covering every stage; what varies per
    # stage is the id the description derives from it.
    assert 'task_id=f"enabled_{stage_name}"' in source
    assert 'task_id=f"trigger_{stage_name}"' in source
    for stage_topology in topology.stages:
        assert stage_topology.gate_task_id == f"enabled_{stage_topology.stage.value}"
        assert stage_topology.trigger_task_id == f"trigger_{stage_topology.stage.value}"


def test_master_edges_describe_the_stage_chain() -> None:
    topology = ingest_dag_topology(MASTER_DAG_ID)
    edges = set(topology.master.edges)
    stages = list(Stage)

    for stage in stages:
        assert (RESOLVE_PROFILE_TASK_ID, f"enabled_{stage.value}") in edges
        assert (f"enabled_{stage.value}", f"trigger_{stage.value}") in edges
    for upstream, downstream in pairwise(stages):
        assert (f"trigger_{upstream.value}", f"enabled_{downstream.value}") in edges

    # The chain, not a fan-out: nothing runs a later stage before the earlier
    # stage's trigger has finished waiting.
    assert (f"trigger_{stages[-1].value}", f"enabled_{stages[0].value}") not in edges


@pytest.mark.parametrize("stage", list(Stage))
def test_sub_dag_tasks_match_the_rendered_stage_dag(rendered_bundle: Path, stage: Stage) -> None:
    source = _dag_source(rendered_bundle, f"ingest_{stage.value}.py")
    stage_topology = ingest_dag_topology(MASTER_DAG_ID).stage(stage)

    assert stage_topology.dag.dag_id == sub_dag_id_for_stage(MASTER_DAG_ID, stage)
    for task in stage_topology.dag.tasks:
        assert f"def {task.task_id}(" in source, f"{task.task_id} missing from {stage.value}"
    assert stage_topology.dag.edges == (
        (PLAN_TASK_ID, PROCESS_BATCH_TASK_ID),
        (PROCESS_BATCH_TASK_ID, budget_gate_task_id(stage)),
    )
    assert f"{PROCESS_BATCH_TASK_ID}.expand(" in source  # the fan-out the UI draws


def test_only_meta_gates_on_the_quarantine_budget() -> None:
    assert budget_gate_task_id(Stage.META) == "quarantine_budget_gate"
    for stage in (Stage.SYNC, Stage.LABELS, Stage.MEDIA):
        assert budget_gate_task_id(stage) == "error_budget_gate"


def test_process_batch_is_the_only_mapped_task() -> None:
    for stage_topology in ingest_dag_topology(MASTER_DAG_ID).stages:
        mapped = [task.task_id for task in stage_topology.dag.tasks if task.mapped]
        assert mapped == [PROCESS_BATCH_TASK_ID]


def test_enabling_profiles_name_which_profiles_run_each_stage() -> None:
    """The profile table as the UI reads it, stated rather than recomputed.

    A profile missing from a stage's tuple is exactly what the master's gate
    skips on -- metadata_backfill runs meta and nothing else.
    """
    topology = ingest_dag_topology(MASTER_DAG_ID)
    assert topology.stage(Stage.SYNC).enabling_profiles == ("full",)
    assert topology.stage(Stage.META).enabling_profiles == ("full", "metadata_backfill")
    assert topology.stage(Stage.LABELS).enabling_profiles == ("full", "relabel")
    assert topology.stage(Stage.MEDIA).enabling_profiles == ("full",)
