"""render_deploy_bundle unit tests: bundle layout, the shared ingest-DAG
template rendered against deploy values, DEPLOY.md contents, data-root URI
validation, and the CLI wiring (no Docker, no Airflow, no platform APIs)."""

from dataclasses import replace
from pathlib import Path

import pytest

import hflow
from hflow.cli import main
from hflow.runtime._deploy import (
    DEFAULT_DEPLOY_VENV_PYTHON,
    DeployConfig,
    DeployPaths,
    render_deploy_bundle,
    validate_data_root_uri,
)
from hflow.runtime._templates import DAG_BUNDLE_CONFIG_LIST_JSON
from hflow.steps import Stage

DATA_ROOT_PATH = "/mnt/robot-data"

PIPELINE_SOURCE = f"import hflow\n\napp = hflow.App('demo', data_root='{DATA_ROOT_PATH}')\n"


@pytest.fixture
def config(tmp_path: Path) -> DeployConfig:
    pipeline_file = tmp_path / "my_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("numpy>=2\n")
    return DeployConfig(
        pipeline_file=pipeline_file,
        data_root_uri=DATA_ROOT_PATH,
        requirements_file=requirements_file,
    )


def _render(config: DeployConfig, output_dir: Path) -> DeployPaths:
    return render_deploy_bundle(config, output_dir)


def test_bundle_layout_and_paths(config: DeployConfig, tmp_path: Path) -> None:
    paths = _render(config, tmp_path / "deploy")
    assert paths.output_dir == tmp_path / "deploy"
    # Every DAG file is named after its dag id (platforms sync whole dags/
    # folders, so generic names would collide across pipelines).
    assert paths.dag_file == paths.output_dir / "dags" / "my_pipeline_ingest.py"
    # Figure 4: the master plus its four stage sub-DAGs, five files total.
    assert paths.sub_dag_files == tuple(
        paths.output_dir / "dags" / f"my_pipeline_{stage.value}.py" for stage in Stage
    )
    assert len(list((paths.output_dir / "dags").glob("*.py"))) == 5
    assert paths.user_dir == paths.output_dir / "user"
    assert paths.deploy_md == paths.output_dir / "DEPLOY.md"
    assert paths.dag_id == "my_pipeline_ingest"
    for created in (paths.dag_file, *paths.sub_dag_files, paths.deploy_md):
        assert created.is_file()
    assert (paths.user_dir / "my_pipeline.py").read_text() == PIPELINE_SOURCE
    assert (paths.user_dir / "requirements.txt").read_text() == "numpy>=2\n"
    # The parser guard: platforms that sync user/ inside the dags folder must
    # never import the pipeline in Airflow's own environment.
    assert (paths.output_dir / "dags" / ".airflowignore").read_text() == "user/\n"


def test_requirements_absent_when_not_supplied(config: DeployConfig, tmp_path: Path) -> None:
    paths = _render(replace(config, requirements_file=None), tmp_path / "deploy")
    assert not (paths.user_dir / "requirements.txt").exists()


def test_dag_sources_compile_and_carry_deploy_values(config: DeployConfig, tmp_path: Path) -> None:
    paths = _render(config, tmp_path / "deploy")

    master_source = paths.dag_file.read_text()
    compile(master_source, str(paths.dag_file), "exec")
    assert 'dag_id="my_pipeline_ingest"' in master_source
    # The master runs in Airflow's environment: no venv, no data root at all.
    assert "external_python" not in master_source
    assert DEFAULT_DEPLOY_VENV_PYTHON not in master_source
    assert "TriggerDagRunOperator" in master_source
    for stage in Stage:
        assert f'"{stage.value}": "my_pipeline_{stage.value}"' in master_source

    for stage, sub_dag_file in zip(Stage, paths.sub_dag_files, strict=True):
        dag_source = sub_dag_file.read_text()

        # Compiles cleanly without importing airflow (compile only, never exec).
        compile(dag_source, str(sub_dag_file), "exec")

        assert f'dag_id="my_pipeline_{stage.value}"' in dag_source
        assert f'stages={{"{stage.value}"}}' in dag_source
        # The data root is parsed through the shared storage boundary for size
        # planning, while processing chooses a Path or full bucket URI.
        # Substituted as a repr() literal (not bare text in a pre-quoted
        # slot), so this holds for any data root content, not just this
        # plain-path case -- see
        # test_dag_sources_survive_paths_that_are_not_valid_python_literals.
        assert f"data_root = parse_storage_root({DATA_ROOT_PATH!r})" in dag_source
        assert "episode_reference = (" in dag_source
        assert f"expected_data_root = {DATA_ROOT_PATH!r}" in dag_source
        assert "if str(app.data_root) != expected_data_root:" in dag_source
        assert (
            '"pipeline data_root must be " + expected_data_root + " inside the runtime; "'
            in dag_source
        )
        assert "/opt/airflow/data" not in dag_source
        # All three tasks point at the (default) deploy venv interpreter.
        assert dag_source.count(f"@task.external_python(python={DEFAULT_DEPLOY_VENV_PYTHON!r}") == 3
        # The pipeline loads from user/, relocatable per platform via env var.
        assert "\"/opt/user/\" + 'my_pipeline.py'" in dag_source
        assert 'os.environ.get("HFLOW_USER_DIR")' in dag_source


def test_dag_sources_honor_custom_venv_python(config: DeployConfig, tmp_path: Path) -> None:
    paths = _render(
        replace(config, venv_python_path="/home/airflow/venvs/tasks/bin/python"),
        tmp_path / "deploy",
    )
    for sub_dag_file in paths.sub_dag_files:
        dag_source = sub_dag_file.read_text()
        compile(dag_source, str(sub_dag_file), "exec")
        assert (
            dag_source.count("@task.external_python(python='/home/airflow/venvs/tasks/bin/python'")
            == 3
        )
        assert DEFAULT_DEPLOY_VENV_PYTHON not in dag_source


def test_dag_sources_survive_paths_that_are_not_valid_python_literals(
    config: DeployConfig, tmp_path: Path
) -> None:
    """A Windows venv interpreter path or a data root containing a quote
    character must not corrupt the generated DAG source (#44): every value
    substituted into the templates is embedded as a repr() literal, which is
    always self-escaping, rather than as bare text inside a pre-quoted slot.

    Before the fix, the backslashes below produced a `SyntaxError` for every
    sub-DAG (`\\U` reads as a truncated unicode escape) with nothing
    surfacing the failure to whoever ran `hflow deploy`; a `data_root`
    containing `"` closed the surrounding string literal early instead.
    """
    windows_venv_python = r"C:\Users\ci\venvs\user\Scripts\python.exe"
    quote_bearing_data_root = '/data/root", os.system("unexpected"); x = ("'
    paths = _render(
        replace(
            config,
            venv_python_path=windows_venv_python,
            data_root_uri=quote_bearing_data_root,
        ),
        tmp_path / "deploy",
    )
    for sub_dag_file in paths.sub_dag_files:
        dag_source = sub_dag_file.read_text()
        compile(dag_source, str(sub_dag_file), "exec")
        assert dag_source.count(f"@task.external_python(python={windows_venv_python!r}") == 3
        # The whole crafted value round-trips as one repr() literal -- if the
        # embedded quote had broken out of the intended string instead, this
        # exact substring would not appear (and `compile()` above would
        # either have raised or accepted a different, corrupted program).
        assert f"data_root = parse_storage_root({quote_bearing_data_root!r})" in dag_source
        assert f"expected_data_root = {quote_bearing_data_root!r}" in dag_source


def test_dag_id_default_rule_and_override(config: DeployConfig, tmp_path: Path) -> None:
    assert config.resolved_dag_id() == "my_pipeline_ingest"
    paths = _render(replace(config, dag_id="custom_ingest"), tmp_path / "deploy")
    assert paths.dag_id == "custom_ingest"
    assert paths.dag_file.name == "custom_ingest.py"
    assert 'dag_id="custom_ingest"' in paths.dag_file.read_text()
    # Sub-DAG files follow the override too.
    assert [sub.name for sub in paths.sub_dag_files] == [
        "custom_sync.py",
        "custom_meta.py",
        "custom_labels.py",
        "custom_media.py",
    ]


def test_code_smuggling_values_are_refused(config: DeployConfig, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dag_id"):
        _render(replace(config, dag_id='x"; import os #'), tmp_path / "b1")
    with pytest.raises(ValueError, match="app_variable"):
        _render(replace(config, app_variable="app; run()"), tmp_path / "b2")


def test_deploy_md_contents(config: DeployConfig, tmp_path: Path) -> None:
    deploy_md = _render(config, tmp_path / "deploy").deploy_md.read_text()
    # The self-managed bundle-config JSON, verbatim from the shared constant
    # (references/airflow3-notes.md, "DAG bundles").
    assert DAG_BUNDLE_CONFIG_LIST_JSON in deploy_md
    assert "AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST" in deploy_md
    assert "my_pipeline_ingest" in deploy_md
    # The external-python venv's package list (virtualenv preinstalled,
    # pendulum + lazy_object_proxy for the operator's datetime probes).
    assert f"virtualenv pendulum lazy_object_proxy hflow=={hflow.__version__}" in deploy_md
    assert "user/requirements.txt" in deploy_md
    # Per-platform placement plus the env var the DAG expects.
    for platform_name in ("Astronomer", "MWAA", "Cloud Composer", "Self-managed"):
        assert platform_name in deploy_md
    assert "HFLOW_USER_DIR" in deploy_md
    assert DEFAULT_DEPLOY_VENV_PYTHON in deploy_md
    assert DATA_ROOT_PATH in deploy_md
    assert "Absolute paths must be mounted identically" in deploy_md
    # Figure 4: the master's profile/mode conf, the sub-DAG list, and the
    # Airflow-side requirement for the master's trigger tasks.
    for sub_stage in ("sync", "meta", "labels", "media"):
        assert f"my_pipeline_{sub_stage}.py" in deploy_md
    assert '"full" | "metadata_backfill" | "relabel"' in deploy_md
    assert '"mode": "batch"|"online"' in deploy_md
    assert "apache-airflow-providers-standard" in deploy_md
    assert "triggerer" in deploy_md


def test_deploy_md_explains_shared_absolute_paths(config: DeployConfig, tmp_path: Path) -> None:
    Path(config.pipeline_file).write_text(
        "import hflow\n\napp = hflow.App('demo', data_root='/mnt/robot-data')\n"
    )
    deploy_md = _render(
        replace(config, data_root_uri="/mnt/robot-data"), tmp_path / "deploy"
    ).deploy_md.read_text()
    assert "Absolute paths must be mounted identically" in deploy_md
    assert "/mnt/robot-data" in deploy_md


@pytest.mark.parametrize(
    "valid_uri, normalized",
    [
        ("/mnt/robot-data", "/mnt/robot-data"),
        ("/mnt/robot-data/", "/mnt/robot-data"),
        ("s3://bucket/prefix", "s3://bucket/prefix"),
        ("gs://bucket/", "gs://bucket"),
        ("abfs://container/prefix/", "abfs://container/prefix"),
    ],
)
def test_data_root_uri_accepts_absolute_paths(valid_uri: str, normalized: str) -> None:
    assert validate_data_root_uri(valid_uri) == normalized


@pytest.mark.parametrize(
    "invalid_uri",
    [
        "data",
        "./data",
        "data/episodes",
        "~/data",
        "file.mcap",
        "",
        "/",  # the filesystem root is never a data root; rstrip leaves nothing
        "http://bucket/prefix",  # not an object store we render for
    ],
)
def test_data_root_uri_rejects_non_deployable_roots(invalid_uri: str) -> None:
    with pytest.raises(ValueError, match="data root"):
        validate_data_root_uri(invalid_uri)


def test_deploy_config_validates_at_construction(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "my_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    with pytest.raises(ValueError, match="relative paths cannot resolve"):
        DeployConfig(pipeline_file=pipeline_file, data_root_uri="./data")
    # The normalized form replaces the raw input (parse at the boundary).
    trailing = DeployConfig(pipeline_file=pipeline_file, data_root_uri="/mnt/robot-data/")
    assert trailing.data_root_uri == "/mnt/robot-data"


def test_deploy_config_supports_object_store_urls_and_installs_backend(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "my_pipeline.py"
    pipeline_file.write_text(
        "import hflow\n\napp = hflow.App('demo', data_root='s3://bucket/prefix')\n"
    )
    config = DeployConfig(pipeline_file=pipeline_file, data_root_uri="s3://bucket/prefix/")
    paths = render_deploy_bundle(config, tmp_path / "deploy")
    assert config.data_root_uri == "s3://bucket/prefix"
    deploy_md = paths.deploy_md.read_text()
    assert f"'hflow[bucket]=={hflow.__version__}'" in deploy_md
    assert "HFLOW_MIRROR_DIR" in deploy_md
    for dag_file in paths.sub_dag_files:
        dag_source = dag_file.read_text()
        assert "parse_storage_root('s3://bucket/prefix')" in dag_source
        assert "is_bucket_url(data_root)" in dag_source


def test_render_warns_on_mismatched_pipeline_data_root_literal(
    config: DeployConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The render-time warning compares against the deployed data root."""
    Path(config.pipeline_file).write_text(
        "import hflow\n\napp = hflow.App('demo', data_root='/opt/airflow/data')\n"
    )
    with caplog.at_level("WARNING", logger="hflow.runtime._bundle"):
        _render(config, tmp_path / "deploy")
    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "'/opt/airflow/data'" in warning_text
    assert DATA_ROOT_PATH in warning_text


def test_render_stays_silent_when_pipeline_matches_deploy_root(
    config: DeployConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="hflow.runtime._bundle"):
        _render(config, tmp_path / "deploy")
    assert not caplog.records


def test_rerender_overwrites_generated_files(config: DeployConfig, tmp_path: Path) -> None:
    output_dir = tmp_path / "deploy"
    _render(config, output_dir)
    paths = _render(replace(config, dag_id="renamed_ingest"), output_dir)
    assert paths.dag_file.name == "renamed_ingest.py"
    assert 'dag_id="renamed_ingest"' in paths.dag_file.read_text()
    assert paths.sub_dag_files[0].name == "renamed_sync.py"
    assert "renamed_ingest" in paths.deploy_md.read_text()


def test_missing_pipeline_file_raises(config: DeployConfig, tmp_path: Path) -> None:
    broken = replace(config, pipeline_file=tmp_path / "nope.py")
    with pytest.raises(FileNotFoundError):
        render_deploy_bundle(broken, tmp_path / "deploy")


def test_cli_deploy_renders_and_prints_pointers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline_file = tmp_path / "demo_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    output_dir = tmp_path / "deploy"
    exit_code = main(
        [
            "deploy",
            "--pipeline",
            f"{pipeline_file}:my_app",
            "--data-root-uri",
            DATA_ROOT_PATH,
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    assert (output_dir / "dags" / "demo_pipeline_ingest.py").is_file()  # the master
    sub_dag_source = (output_dir / "dags" / "demo_pipeline_sync.py").read_text()
    assert 'getattr(pipeline_module, "my_app")' in sub_dag_source
    output = capsys.readouterr().out
    assert str(output_dir / "DEPLOY.md") in output
    assert "demo_pipeline_ingest" in output


def test_cli_deploy_honors_requirements_and_venv_python(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "demo_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("scipy>=1\n")
    output_dir = tmp_path / "deploy"
    exit_code = main(
        [
            "deploy",
            "--pipeline",
            str(pipeline_file),
            "--data-root-uri",
            DATA_ROOT_PATH,
            "--output-dir",
            str(output_dir),
            "--requirements",
            str(requirements_file),
            "--venv-python",
            "/custom/venv/bin/python",
        ]
    )
    assert exit_code == 0
    assert (output_dir / "user" / "requirements.txt").read_text() == "scipy>=1\n"
    dag_source = (output_dir / "dags" / "demo_pipeline_meta.py").read_text()
    assert dag_source.count("@task.external_python(python='/custom/venv/bin/python'") == 3


def test_cli_deploy_rejects_relative_data_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline_file = tmp_path / "demo_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    exit_code = main(
        [
            "deploy",
            "--pipeline",
            str(pipeline_file),
            "--data-root-uri",
            "./data",
            "--output-dir",
            str(tmp_path / "deploy"),
        ]
    )
    assert exit_code == 2
    errors = capsys.readouterr().err
    assert "deploy:" in errors
    assert "'./data'" in errors
    assert not (tmp_path / "deploy").exists()
