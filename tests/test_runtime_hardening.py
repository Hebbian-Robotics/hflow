"""Generated runtime bundle security, path, and lifecycle regressions."""

import stat
from pathlib import Path

import pytest
import yaml

from hflow.cli import main as cli_main
from hflow.runtime import RuntimeConfig, render_bundle


def _pipeline_file(tmp_path: Path) -> Path:
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("import hflow\napp = hflow.App('t', data_root='/opt/airflow/data')\n")
    return pipeline


def _config(
    tmp_path: Path,
    *,
    data_root: Path | None = None,
    hflow_source: Path | None = None,
    dag_id: str | None = None,
    app_variable: str = "app",
) -> RuntimeConfig:
    return RuntimeConfig(
        pipeline_file=_pipeline_file(tmp_path),
        data_root=data_root if data_root is not None else tmp_path / "data",
        hflow_source=hflow_source,
        dag_id=dag_id,
        app_variable=app_variable,
    )


def test_env_file_is_owner_only_and_healed_on_rerender(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle_dir = tmp_path / "bundle"
    paths = render_bundle(config, bundle_dir)
    assert stat.S_IMODE(paths.env_file.stat().st_mode) == 0o600

    paths.env_file.chmod(0o644)
    original_content = paths.env_file.read_text()
    render_bundle(config, bundle_dir)
    assert stat.S_IMODE(paths.env_file.stat().st_mode) == 0o600
    assert paths.env_file.read_text() == original_content


def test_bundles_get_unique_stable_project_names(tmp_path: Path) -> None:
    first = render_bundle(_config(tmp_path), tmp_path / "proj-a" / "data" / "runtime")
    second = render_bundle(_config(tmp_path), tmp_path / "proj-b" / "data" / "runtime")
    names = []
    for paths in (first, second):
        parsed = yaml.safe_load(paths.compose_file.read_text())
        names.append(parsed["name"])
        assert parsed["name"].startswith("hflow-")
    assert names[0] != names[1], "default bundles must never share a Compose project"

    rerendered = render_bundle(_config(tmp_path), tmp_path / "proj-a" / "data" / "runtime")
    assert yaml.safe_load(rerendered.compose_file.read_text())["name"] == names[0]


def test_awkward_paths_render_parseable_compose(tmp_path: Path) -> None:
    awkward_root = tmp_path / "o'brien" / "$JWT_SECRET-data"
    awkward_root.mkdir(parents=True)
    paths = render_bundle(
        _config(tmp_path, data_root=awkward_root, hflow_source=awkward_root),
        tmp_path / "bundle",
    )
    parsed = yaml.safe_load(paths.compose_file.read_text())
    volumes = parsed["services"]["airflow-scheduler"]["volumes"]
    data_mounts = [volume for volume in volumes if volume.endswith(":/opt/airflow/data")]
    # YAML sees the doubled '' as one quote; $ stays doubled for compose to undo.
    assert data_mounts and "$$JWT_SECRET" in data_mounts[0]


def test_hostile_dag_identifiers_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dag_id"):
        render_bundle(_config(tmp_path, dag_id='x"; import os #'), tmp_path / "b1")
    with pytest.raises(ValueError, match="app_variable"):
        render_bundle(_config(tmp_path, app_variable="app; run()"), tmp_path / "b2")


def test_generated_gate_budgets_errors_too(tmp_path: Path) -> None:
    paths = render_bundle(_config(tmp_path), tmp_path / "bundle")
    # The gates live in the stage sub-DAGs (the master only triggers them);
    # every one of them must call a library budget (whose error-budgeting
    # behavior tests/test_stage_execution.py verifies directly).
    for sub_dag_file in paths.sub_dag_files:
        dag_source = sub_dag_file.read_text()
        compile(dag_source, str(sub_dag_file), "exec")
        assert "from hflow.stage_execution import summarize_" in dag_source
        assert "_budget(batch_counts)" in dag_source


def test_ingest_rejects_absolute_and_escaping_uris(tmp_path: Path) -> None:
    assert cli_main(["ingest", "/home/user/x.mcap", "--bundle-dir", str(tmp_path)]) == 2
    assert cli_main(["ingest", "../outside.mcap", "--bundle-dir", str(tmp_path)]) == 2
    assert cli_main(["ingest", "a/../../outside.mcap", "--bundle-dir", str(tmp_path)]) == 2
