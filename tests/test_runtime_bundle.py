"""render_bundle unit tests: bundle layout, compose shape, .env create-if-absent,
and the generated master/sub-DAG sources (no Docker, no Airflow imports)."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

import hflow
from hflow.runtime import BundlePaths, RuntimeConfig, bundle_dag_ids, render_bundle
from hflow.runtime._bundle import infer_hflow_source
from hflow.steps import RUN_PROFILES, Stage

AIRFLOW_SERVICE_NAMES = (
    "airflow-apiserver",
    "airflow-scheduler",
    "airflow-dag-processor",
    "airflow-triggerer",
)

PIPELINE_SOURCE = "import hflow\n\napp = hflow.App('demo', data_root='/opt/airflow/data')\n"


@pytest.fixture
def config(tmp_path: Path) -> RuntimeConfig:
    pipeline_file = tmp_path / "my_pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("numpy>=2\n")
    hflow_source = tmp_path / "hflow-src"
    hflow_source.mkdir()
    (hflow_source / "pyproject.toml").write_text("[project]\nname = 'hflow'\n")
    return RuntimeConfig(
        pipeline_file=pipeline_file,
        data_root=tmp_path / "data",
        requirements_file=requirements_file,
        hflow_source=hflow_source,
    )


def _render(config: RuntimeConfig, bundle_dir: Path) -> tuple[BundlePaths, dict[str, Any]]:
    paths = render_bundle(config, bundle_dir)
    compose = yaml.safe_load(paths.compose_file.read_text())
    return paths, compose


def test_bundle_layout_and_paths(config: RuntimeConfig, tmp_path: Path) -> None:
    paths, _ = _render(config, tmp_path / "bundle")
    assert paths.bundle_dir == tmp_path / "bundle"
    assert paths.compose_file == paths.bundle_dir / "docker-compose.yaml"
    assert paths.env_file == paths.bundle_dir / ".env"
    assert paths.dag_file == paths.bundle_dir / "dags" / "ingest.py"
    assert paths.user_dir == paths.bundle_dir / "user"
    # Figure 4: the master plus its four stage sub-DAGs, five files total.
    assert paths.sub_dag_files == tuple(
        paths.bundle_dir / "dags" / f"ingest_{stage.value}.py" for stage in Stage
    )
    for created in (paths.compose_file, paths.env_file, paths.dag_file, *paths.sub_dag_files):
        assert created.is_file()
    assert len(list((paths.bundle_dir / "dags").glob("*.py"))) == 5
    assert (paths.bundle_dir / "logs").is_dir()
    assert (Path(config.data_root) / "xcom").is_dir()
    assert paths.api_base_url == "http://127.0.0.1:8080"
    assert paths.dag_id == "my_pipeline_ingest"


def test_compose_parses_with_expected_services(config: RuntimeConfig, tmp_path: Path) -> None:
    _, compose = _render(config, tmp_path / "bundle")
    assert set(compose["services"]) == {
        "postgres",
        "airflow-init",
        "user-venv-init",
        *AIRFLOW_SERVICE_NAMES,
    }
    assert set(compose["volumes"]) == {"postgres-db-volume", "user-venv"}
    assert compose["services"]["postgres"]["image"] == "${POSTGRES_IMAGE}"
    assert "pg_isready" in compose["services"]["postgres"]["healthcheck"]["test"]


def test_compose_common_environment(config: RuntimeConfig, tmp_path: Path) -> None:
    _, compose = _render(config, tmp_path / "bundle")
    scheduler_env = compose["services"]["airflow-scheduler"]["environment"]
    assert scheduler_env["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor"
    assert scheduler_env["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] == (
        "postgresql+psycopg2://airflow:airflow@postgres/airflow"
    )
    assert scheduler_env["AIRFLOW__CORE__EXECUTION_API_SERVER_URL"] == (
        "http://airflow-apiserver:8080/execution/"
    )
    assert scheduler_env["AIRFLOW__CORE__AUTH_MANAGER"] == (
        "airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager"
    )
    assert scheduler_env["AIRFLOW__API_AUTH__JWT_SECRET"] == "${JWT_SECRET}"
    assert scheduler_env["AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION"] == "false"
    assert scheduler_env["AIRFLOW__CORE__LOAD_EXAMPLES"] == "false"
    assert json.loads(scheduler_env["AIRFLOW__DAG_PROCESSOR__DAG_BUNDLE_CONFIG_LIST"]) == [
        {
            "name": "dags-folder",
            "classpath": "airflow.dag_processing.bundles.local.LocalDagBundle",
            "kwargs": {"path": "/opt/airflow/dags"},
        }
    ]
    assert scheduler_env["AIRFLOW__CORE__XCOM_BACKEND"] == (
        "airflow.providers.common.io.xcom.backend.XComObjectStorageBackend"
    )
    assert scheduler_env["AIRFLOW__COMMON_IO__XCOM_OBJECTSTORAGE_PATH"] == (
        "file:///opt/airflow/data/xcom"
    )
    assert scheduler_env["AIRFLOW__COMMON_IO__XCOM_OBJECTSTORAGE_THRESHOLD"] == "4096"
    # The pinned ffmpeg download caches in the user-venv volume, not the
    # ephemeral container home.
    assert scheduler_env["XDG_CACHE_HOME"] == "/opt/venvs/cache"


def test_compose_has_no_celery_or_redis(config: RuntimeConfig, tmp_path: Path) -> None:
    paths, _ = _render(config, tmp_path / "bundle")
    # Substituted host paths are the user's own (and here contain this very
    # test's name); only the template-authored text is under scrutiny.
    compose_text = paths.compose_file.read_text().lower().replace(str(tmp_path).lower(), "")
    for forbidden in ("redis", "celery", "flower"):
        assert forbidden not in compose_text


def test_compose_apiserver_port_and_healthcheck(config: RuntimeConfig, tmp_path: Path) -> None:
    _, compose = _render(config, tmp_path / "bundle")
    apiserver = compose["services"]["airflow-apiserver"]
    assert apiserver["command"] == "api-server"
    assert apiserver["ports"] == ["127.0.0.1:${API_PORT}:8080"]
    assert apiserver["healthcheck"]["test"] == [
        "CMD",
        "curl",
        "--fail",
        "http://localhost:8080/api/v2/monitor/health",
    ]


def test_compose_volumes(config: RuntimeConfig, tmp_path: Path) -> None:
    _, compose = _render(config, tmp_path / "bundle")
    data_root = str(Path(config.data_root).resolve())
    hflow_source = str(Path(config.hflow_source or "").resolve())
    scheduler_volumes = compose["services"]["airflow-scheduler"]["volumes"]
    assert scheduler_volumes == [
        "./dags:/opt/airflow/dags",
        "./logs:/opt/airflow/logs",
        f"{data_root}:/opt/airflow/data",
        "./user:/opt/user:ro",
        "user-venv:/opt/venvs",
        f"{hflow_source}:/opt/hflow-src:ro",
    ]
    venv_init_volumes = compose["services"]["user-venv-init"]["volumes"]
    assert venv_init_volumes == [
        "user-venv:/opt/venvs",
        "./user:/opt/user:ro",
        f"{hflow_source}:/opt/hflow-src:ro",
    ]


def test_compose_hflow_source_mount_absent_when_unset(
    config: RuntimeConfig, tmp_path: Path
) -> None:
    from dataclasses import replace

    paths, compose = _render(replace(config, hflow_source=None), tmp_path / "bundle")
    assert ":/opt/hflow-src:ro" not in paths.compose_file.read_text()
    assert compose["services"]["user-venv-init"]["volumes"] == [
        "user-venv:/opt/venvs",
        "./user:/opt/user:ro",
    ]
    _, script = compose["services"]["user-venv-init"]["command"]
    assert f"hflow_install_target='hflow=={hflow.__version__}'" in script
    assert 'pip install --no-cache-dir "$$hflow_install_target"' in script
    assert "if [ -d /opt/hflow-src ]" not in script


def test_infer_hflow_source_ignores_wheel_venv_inside_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout_directory = tmp_path / "hflow-checkout"
    (checkout_directory / "src" / "hflow").mkdir(parents=True)
    (checkout_directory / "pyproject.toml").write_text('[project]\nname = "hflow"\n')
    wheel_package_file = (
        checkout_directory
        / ".release-venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "hflow"
        / "__init__.py"
    )
    wheel_package_file.parent.mkdir(parents=True)
    wheel_package_file.touch()
    monkeypatch.setattr(hflow, "__file__", str(wheel_package_file))

    assert infer_hflow_source() is None


def test_infer_hflow_source_recognizes_imported_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout_directory = tmp_path / "hflow-checkout"
    source_package_file = checkout_directory / "src" / "hflow" / "__init__.py"
    source_package_file.parent.mkdir(parents=True)
    source_package_file.touch()
    (checkout_directory / "pyproject.toml").write_text('[project]\nname = "hflow"\n')
    monkeypatch.setattr(hflow, "__file__", str(source_package_file))

    assert infer_hflow_source() == checkout_directory


def test_compose_depends_on_gates(config: RuntimeConfig, tmp_path: Path) -> None:
    _, compose = _render(config, tmp_path / "bundle")
    for service_name in AIRFLOW_SERVICE_NAMES:
        depends_on = compose["services"][service_name]["depends_on"]
        assert depends_on["postgres"]["condition"] == "service_healthy"
        assert depends_on["airflow-init"]["condition"] == "service_completed_successfully"
        assert depends_on["user-venv-init"]["condition"] == "service_completed_successfully"
    assert compose["services"]["airflow-init"]["depends_on"] == {
        "postgres": {"condition": "service_healthy"}
    }
    assert "depends_on" not in compose["services"]["user-venv-init"]


def test_airflow_init_migrates_and_creates_admin_user(
    config: RuntimeConfig, tmp_path: Path
) -> None:
    _, compose = _render(config, tmp_path / "bundle")
    init_service = compose["services"]["airflow-init"]
    assert init_service["command"] == "version"
    init_env = init_service["environment"]
    assert init_env["_AIRFLOW_DB_MIGRATE"] == "true"
    assert init_env["_AIRFLOW_WWW_USER_CREATE"] == "true"
    assert init_env["_AIRFLOW_WWW_USER_USERNAME"] == "${AIRFLOW_ADMIN_USERNAME}"
    assert init_env["_AIRFLOW_WWW_USER_PASSWORD"] == "${AIRFLOW_ADMIN_PASSWORD}"
    # The anchor merge still applies: init shares the common env and image.
    assert init_env["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor"
    assert init_service["image"] == "${AIRFLOW_IMAGE}"


def test_user_venv_init_builds_with_content_hash_marker(
    config: RuntimeConfig, tmp_path: Path
) -> None:
    _, compose = _render(config, tmp_path / "bundle")
    venv_init = compose["services"]["user-venv-init"]
    assert venv_init["image"] == "${AIRFLOW_IMAGE}"
    # Root on purpose: a fresh named volume is root-owned, so the build must
    # run as root and hand /opt/venvs to the airflow uid before exiting.
    assert venv_init["user"] == "0:0"
    assert venv_init["entrypoint"] == "/bin/bash"
    flag, script = venv_init["command"]
    assert flag == "-c"
    # Compose interpolation escaping: shell variables stay $$-escaped in YAML.
    assert "sha256sum" in script
    assert "python -m venv /opt/venvs/user" in script
    assert "pip install --no-cache-dir -r /opt/user/requirements.txt" in script
    # The install target is a shell variable so bucket-mode bundles can add
    # the [bucket] extra; local mode renders the bare source path. Shell
    # variables stay $$-escaped in the YAML for Compose's interpolation.
    assert "hflow_install_target='/opt/hflow-src'" in script
    assert 'pip install --no-cache-dir "$$hflow_install_target"' in script
    assert "marker_file=/opt/venvs/user/.hflow-content-hash" in script
    assert "skipping rebuild" in script
    assert "exit 0" in script
    assert 'chown -R "${AIRFLOW_UID}:0" /opt/venvs' in script
    assert "mkdir -p /opt/venvs/cache" in script
    # The external-python bootstrap packages are installed and join
    # the content-hash input, so pre-existing venvs rebuild exactly once. The
    # $$ stays: shell variables are $$-escaped for Compose's interpolation.
    # Exactly pendulum, pinned to the image's constraint; lazy_object_proxy
    # never crosses into the venv so it is deliberately absent from this list.
    assert 'bootstrap_packages="pendulum==3.2.0"' in script
    assert "pip install --no-cache-dir $$bootstrap_packages" in script
    assert 'echo "bootstrap: $$bootstrap_packages"' in script


def test_user_venv_init_prewarms_ffmpeg_best_effort(config: RuntimeConfig, tmp_path: Path) -> None:
    """The pinned ffmpeg downloads at provision time on a best-effort basis."""
    _, compose = _render(config, tmp_path / "bundle")
    _, script = compose["services"]["user-venv-init"]["command"]
    prewarm_line = next(line for line in script.splitlines() if "ffmpeg_path" in line)
    # The cache env var points into the named volume, so the download survives
    # container restarts and is found by task processes.
    assert prewarm_line.strip().startswith("XDG_CACHE_HOME=/opt/venvs/cache")
    assert "from hflow.ffmpeg import ffmpeg_path; ffmpeg_path()" in prewarm_line
    # Best-effort: air-gapped provisioning must still succeed (task-time
    # fallback remains), so the prewarm never fails the init service.
    assert prewarm_line.rstrip().endswith("|| true")
    # Prewarm runs only on a rebuild: it sits after the marker-skip `exit 0`.
    assert script.index("exit 0") < script.index("ffmpeg_path")


def test_env_created_with_config_values_and_secrets(config: RuntimeConfig, tmp_path: Path) -> None:
    from dataclasses import replace

    paths, _ = _render(
        replace(config, api_port=9099, airflow_image="apache/airflow:3.3.1-python3.12"),
        tmp_path / "bundle",
    )
    env_values = dict(
        line.split("=", 1)
        for line in paths.env_file.read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert env_values["AIRFLOW_UID"] == str(os.getuid())
    assert env_values["API_PORT"] == "9099"
    assert env_values["AIRFLOW_IMAGE"] == "apache/airflow:3.3.1-python3.12"
    assert env_values["POSTGRES_IMAGE"] == "postgres:16"
    assert len(env_values["JWT_SECRET"]) >= 24
    assert env_values["AIRFLOW_ADMIN_USERNAME"] == "airflow"
    assert len(env_values["AIRFLOW_ADMIN_PASSWORD"]) >= 24  # generated, not "airflow"
    assert paths.api_base_url == "http://127.0.0.1:9099"
    assert paths.admin_username == "airflow"
    assert paths.admin_password == env_values["AIRFLOW_ADMIN_PASSWORD"]


def test_env_preserved_on_rerender(config: RuntimeConfig, tmp_path: Path) -> None:
    from dataclasses import replace

    bundle_dir = tmp_path / "bundle"
    first_paths, _ = _render(config, bundle_dir)
    original_env_text = first_paths.env_file.read_text()

    changed_config = replace(
        config,
        api_port=9999,
        airflow_image="apache/airflow:9.9.9",
        admin_password="should-not-take-effect",
        data_root=tmp_path / "other-data",
    )
    second_paths, _ = _render(changed_config, bundle_dir)

    # .env is create-if-absent: byte-identical after a re-render with new config.
    assert second_paths.env_file.read_text() == original_env_text
    # BundlePaths reports what is actually in effect (the preserved .env)...
    assert second_paths.api_base_url == first_paths.api_base_url == "http://127.0.0.1:8080"
    assert second_paths.admin_password == first_paths.admin_password
    # ...while derived files ARE refreshed from the new config.
    assert str(Path(tmp_path / "other-data").resolve()) in second_paths.compose_file.read_text()


def test_supplied_admin_password_lands_in_env(config: RuntimeConfig, tmp_path: Path) -> None:
    from dataclasses import replace

    paths, _ = _render(replace(config, admin_password="hunter2"), tmp_path / "bundle")
    assert "AIRFLOW_ADMIN_PASSWORD=hunter2" in paths.env_file.read_text()
    assert paths.admin_password == "hunter2"


def test_master_dag_source_compiles_and_encodes_contract(
    config: RuntimeConfig, tmp_path: Path
) -> None:
    paths, _ = _render(config, tmp_path / "bundle")
    dag_source = paths.dag_file.read_text()

    # Compiles cleanly without importing airflow (compile only, never exec).
    compile(dag_source, str(paths.dag_file), "exec")

    assert 'dag_id="my_pipeline_ingest"' in dag_source
    assert "schedule=None" in dag_source
    assert (
        'params={"uris": [], "profile": "full", "mode": "batch", "batch_count": None}' in dag_source
    )
    # The master runs ENTIRELY in Airflow's environment: no user venv, no
    # hflow import -- the profile vocabulary is baked in as a literal.
    assert "external_python" not in dag_source
    import_lines = [
        line
        for line in dag_source.splitlines()
        if line.startswith(("import ", "from ")) or line.lstrip().startswith(("import ", "from "))
    ]
    assert not any("hflow" in line for line in import_lines), import_lines
    assert '"full": ("sync", "meta", "labels", "media"),' in dag_source
    assert '"metadata_backfill": ("meta",),' in dag_source
    assert '"relabel": ("labels",),' in dag_source
    for profile_name in RUN_PROFILES:
        assert f'"{profile_name}":' in dag_source
    # resolve_profile validates the conf; per-stage gates skip disabled
    # triggers via AirflowSkipException, and the chain survives skips (but
    # never failures). Deliberately NOT @task.branch: skip_all_except would
    # re-enable every stage after the first enabled one (they are chained).
    assert "def resolve_profile(" in dag_source
    assert "unknown run profile" in dag_source
    assert "unknown mode" in dag_source
    assert "@task.branch\n" not in dag_source  # the docstring may explain why not
    assert "AirflowSkipException" in dag_source
    assert 'stage_gate.override(task_id=f"enabled_{stage_name}")' in dag_source
    assert '@task(trigger_rule="none_failed")' in dag_source
    # One TriggerDagRunOperator per stage, deferrable with a waiting master.
    assert (
        "from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator"
        in dag_source
    )
    assert 'task_id=f"trigger_{stage_name}"' in dag_source
    assert "wait_for_completion=True" in dag_source
    assert "deferrable=True" in dag_source
    for stage in Stage:
        assert f'"{stage.value}": "my_pipeline_{stage.value}"' in dag_source
    # uris/mode/batch_count pass through to every triggered sub-DAG.
    assert '"uris": "{{ params.uris }}"' in dag_source
    assert '"mode": "{{ params.mode }}"' in dag_source
    assert '"batch_count": "{{ params.batch_count }}"' in dag_source


def test_sub_dag_sources_compile_and_encode_contract(config: RuntimeConfig, tmp_path: Path) -> None:
    paths, _ = _render(config, tmp_path / "bundle")
    for stage, sub_dag_file in zip(Stage, paths.sub_dag_files, strict=True):
        dag_source = sub_dag_file.read_text()

        # Compiles cleanly without importing airflow (compile only, never exec).
        compile(dag_source, str(sub_dag_file), "exec")

        assert f'dag_id="my_pipeline_{stage.value}"' in dag_source
        assert "schedule=None" in dag_source
        assert 'params={"uris": [], "mode": "batch", "batch_count": None}' in dag_source
        # All three tasks run in the user venv via external python.
        assert dag_source.count("@task.external_python(python='/opt/venvs/user/bin/python'") == 3
        # Imports live inside the (indented) function bodies -- the operator
        # extracts each body to a temp file, so module-level names never survive.
        assert "\n        from hflow.batching import plan_batches" in dag_source
        assert "\n        import importlib.util" in dag_source
        assert "\n        import math" in dag_source
        # The user pipeline is imported by file path and app resolved by name.
        assert "\"/opt/user/\" + 'my_pipeline.py'" in dag_source
        assert "spec_from_file_location" in dag_source
        assert 'getattr(pipeline_module, "app")' in dag_source
        # Each sub-DAG runs exactly its own Figure 4 stage.
        assert f'stages={{"{stage.value}"}}' in dag_source
        # The online lane: one immediate batch, no stagger, batch_count ignored.
        assert 'if mode == "online":' in dag_source
        assert '"start_delay_s": 0.0' in dag_source
        # The stagger, mapped batches, and error budget are generated as one contract.
        assert 'time.sleep(float(batch["start_delay_s"]))' in dag_source
        assert "if report.has_errors:" in dag_source
        assert 'counts["errors"] += 1' in dag_source
        assert "process_batch.expand(batch=batches)" in dag_source
        assert "max(8, math.ceil(0.01 * total))" in dag_source
        # The gate materializes the mapped results (lazy XCom proxies cannot
        # cross the external-python pickle boundary; list-typed task_ids keeps
        # a single-batch run from being flattened) and keeps the edge explicit.
        assert "{{ ti.xcom_pull(task_ids=['process_batch']) | list }}" in dag_source
        assert "batch_counts >> gate" in dag_source
        # The process task authoritatively refuses an app
        # whose data_root is not the in-container mount point.
        assert "expected_data_root = '/opt/airflow/data'" in dag_source
        assert "if str(app.data_root) != expected_data_root:" in dag_source
        assert (
            '"pipeline data_root must be " + expected_data_root + " inside the runtime; "'
            in dag_source
        )
        # Checks decide quarantine: the quarantine budget lives ONLY in meta;
        # every other stage keeps the error-budget half.
        if stage is Stage.META:
            assert "def quarantine_budget_gate(" in dag_source
            assert "quarantined > budget" in dag_source
        else:
            assert "def error_budget_gate(" in dag_source
            assert "quarantined > budget" not in dag_source
        assert "errors > budget" in dag_source
        assert "errors == total" in dag_source


def test_dag_sources_carry_airflow_ui_polish(config: RuntimeConfig, tmp_path: Path) -> None:
    """Demo criterion: the five-DAG structure must be legible in the UI --
    doc_md pages, a shared pipeline tag for one-click filtering, per-role
    tags, display names, and trigger tasks that explain their deferred wait.
    All five kwargs verified present in apache/airflow:3.3.1."""
    from hflow.runtime._bundle import STAGE_DESCRIPTIONS

    paths, _ = _render(config, tmp_path / "bundle")

    master_source = paths.dag_file.read_text()
    assert "doc_md=DOC_MD" in master_source
    assert 'tags=["my_pipeline", "master"]' in master_source
    assert 'dag_display_name="my_pipeline · ingest (master)"' in master_source
    assert 'description="Resolves the run profile' in master_source
    # The doc_md profile table is baked from RUN_PROFILES, not hand-written.
    assert "| run profile | stages enabled |" in master_source
    assert "| `full` | sync, meta, labels, media |" in master_source
    assert "| `relabel` | labels |" in master_source
    # The teal deferred state is explained where a demo audience will look.
    assert 'task_display_name=f"trigger {stage_name} · waits (deferred)"' in master_source
    assert "waits (deferred, checked every " in master_source

    for stage, sub_dag_file in zip(Stage, paths.sub_dag_files, strict=True):
        sub_source = sub_dag_file.read_text()
        compile(sub_source, str(sub_dag_file), "exec")
        assert "doc_md=DOC_MD" in sub_source
        assert f'tags=["my_pipeline", "stage:{stage.value}"]' in sub_source
        assert f'dag_display_name="my_pipeline · {stage.value}"' in sub_source
        assert STAGE_DESCRIPTIONS[stage] in sub_source
        # The doc names the master and the direct-trigger rerun affordance.
        assert "my_pipeline_ingest" in sub_source
        assert "per-stage reruns" in sub_source


def test_bundle_dag_ids_derivation() -> None:
    assert bundle_dag_ids("my_pipeline_ingest") == [
        "my_pipeline_ingest",
        "my_pipeline_sync",
        "my_pipeline_meta",
        "my_pipeline_labels",
        "my_pipeline_media",
    ]
    # A custom id without the _ingest suffix still derives usable sub ids.
    assert bundle_dag_ids("customdag")[1] == "customdag_sync"


def test_render_warns_on_mismatched_data_root_literal(
    config: RuntimeConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A differing literal data root warns at render time."""
    Path(config.pipeline_file).write_text(
        "import hflow\n\napp = hflow.App('demo', data_root='./data')\n"
    )
    with caplog.at_level("WARNING", logger="hflow.runtime._bundle"):
        render_bundle(config, tmp_path / "bundle")
    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "data_root" in warning_text
    assert "'./data'" in warning_text
    assert "/opt/airflow/data" in warning_text


@pytest.mark.parametrize(
    "pipeline_source",
    [
        # The correct literal, in both bare-string and Path-wrapped forms.
        "import hflow\n\napp = hflow.App('demo', data_root='/opt/airflow/data')\n",
        (
            "from pathlib import Path\nimport hflow\n\n"
            "app = hflow.App('demo', data_root=Path('/opt/airflow/data'))\n"
        ),
        # A variable, not a literal: unknowable at render time, so no warning
        # (the in-container check remains authoritative).
        "import hflow\n\nroot = compute_root()\napp = hflow.App('demo', data_root=root)\n",
    ],
)
def test_render_stays_silent_without_a_differing_data_root_literal(
    config: RuntimeConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    pipeline_source: str,
) -> None:
    Path(config.pipeline_file).write_text(pipeline_source)
    with caplog.at_level("WARNING", logger="hflow.runtime._bundle"):
        render_bundle(config, tmp_path / "bundle")
    assert not caplog.records


def test_dag_id_override(config: RuntimeConfig, tmp_path: Path) -> None:
    from dataclasses import replace

    paths, _ = _render(replace(config, dag_id="custom_ingest"), tmp_path / "bundle")
    assert paths.dag_id == "custom_ingest"
    assert 'dag_id="custom_ingest"' in paths.dag_file.read_text()
    # Sub-DAG ids follow the override: _ingest is replaced by the stage name.
    assert '"sync": "custom_sync"' in paths.dag_file.read_text()
    assert 'dag_id="custom_media"' in paths.sub_dag_files[-1].read_text()


def test_pipeline_and_requirements_copied(config: RuntimeConfig, tmp_path: Path) -> None:
    paths, _ = _render(config, tmp_path / "bundle")
    assert (paths.user_dir / "my_pipeline.py").read_text() == PIPELINE_SOURCE
    assert (paths.user_dir / "requirements.txt").read_text() == "numpy>=2\n"


def test_missing_pipeline_file_raises(config: RuntimeConfig, tmp_path: Path) -> None:
    from dataclasses import replace

    broken = replace(config, pipeline_file=tmp_path / "nope.py")
    with pytest.raises(FileNotFoundError):
        render_bundle(broken, tmp_path / "bundle")


class TestBucketModeBundle:
    """render_bundle with a bucket data root: no data mount, native store I/O."""

    BUCKET_URL = "gs://demo-bucket/robot-data"
    PIPELINE = f"import hflow\n\napp = hflow.App('demo', data_root='{BUCKET_URL}')\n"

    @pytest.fixture
    def bucket_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeConfig:
        # Renders must be deterministic regardless of the developer's shell:
        # blank every credential variable the renderer inspects.
        for variable_name in (
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_SERVICE_ACCOUNT_KEY",
        ):
            monkeypatch.delenv(variable_name, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no gcloud ADC file
        pipeline_file = tmp_path / "my_pipeline.py"
        pipeline_file.write_text(self.PIPELINE)
        hflow_source = tmp_path / "hflow-src"
        hflow_source.mkdir()
        (hflow_source / "pyproject.toml").write_text("[project]\nname = 'hflow'\n")
        return RuntimeConfig(
            pipeline_file=pipeline_file,
            data_root=self.BUCKET_URL,
            hflow_source=hflow_source,
        )

    def test_no_data_mount_and_bundle_local_xcom(
        self, bucket_config: RuntimeConfig, tmp_path: Path
    ) -> None:
        paths, compose = _render(bucket_config, tmp_path / "bundle")
        volumes = compose["x-airflow-common"]["volumes"]
        assert not any(":/opt/airflow/data" in str(volume) for volume in volumes)
        assert "./xcom:/opt/airflow/xcom-data" in volumes
        environment = compose["x-airflow-common"]["environment"]
        assert (
            environment["AIRFLOW__COMMON_IO__XCOM_OBJECTSTORAGE_PATH"]
            == "file:///opt/airflow/xcom-data"
        )
        # The bind source is pre-created so docker cannot root-own it.
        assert (paths.bundle_dir / "xcom").is_dir()

    def test_task_venv_installs_bucket_extra(
        self, bucket_config: RuntimeConfig, tmp_path: Path
    ) -> None:
        _, compose = _render(bucket_config, tmp_path / "bundle")
        _, script = compose["services"]["user-venv-init"]["command"]
        assert "hflow_install_target='/opt/hflow-src[bucket]'" in script
        # The target joins the content hash so switching a bundle between
        # local and bucket roots rebuilds the venv exactly once.
        assert 'echo "install: $$hflow_install_target"' in script

    def test_published_install_includes_bucket_extra(
        self, bucket_config: RuntimeConfig, tmp_path: Path
    ) -> None:
        from dataclasses import replace

        _, compose = _render(
            replace(bucket_config, hflow_source=None), tmp_path / "published-bundle"
        )
        _, script = compose["services"]["user-venv-init"]["command"]
        assert f"hflow_install_target='hflow[bucket]=={hflow.__version__}'" in script
        assert ":/opt/hflow-src:ro" not in str(compose["x-airflow-common"]["volumes"])

    def test_dags_render_against_the_bucket_url(
        self, bucket_config: RuntimeConfig, tmp_path: Path
    ) -> None:
        paths, _ = _render(bucket_config, tmp_path / "bundle")
        for sub_dag_file in paths.sub_dag_files:
            dag_source = sub_dag_file.read_text()
            assert f"parse_storage_root({self.BUCKET_URL!r})" in dag_source
            assert f"expected_data_root = {self.BUCKET_URL!r}" in dag_source
            assert "if str(app.data_root) != expected_data_root:" in dag_source
        # The media plan filter probes episode channel lists, which would
        # download whole remote files at plan time: bucket bundles omit it.
        media_source = (paths.bundle_dir / "dags" / "ingest_media.py").read_text()
        assert "has_camera" not in media_source
        local_config = RuntimeConfig(
            pipeline_file=bucket_config.pipeline_file,
            data_root=tmp_path / "data",
            hflow_source=bucket_config.hflow_source,
        )
        local_paths = render_bundle(local_config, tmp_path / "local-bundle")
        local_media_source = (local_paths.bundle_dir / "dags" / "ingest_media.py").read_text()
        assert "has_camera" in local_media_source
        # The local filter body imports Path itself (the plan body no longer
        # does): a NameError here would only surface at task run time.
        assert "from pathlib import Path" in local_media_source

    def test_gcs_credentials_mount_only_when_a_file_exists(
        self, bucket_config: RuntimeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, compose = _render(bucket_config, tmp_path / "bundle-no-creds")
        environment = compose["x-airflow-common"]["environment"]
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment

        credentials_file = tmp_path / "service-account.json"
        credentials_file.write_text("{}")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))
        _, compose_with_creds = _render(bucket_config, tmp_path / "bundle-creds")
        environment = compose_with_creds["x-airflow-common"]["environment"]
        assert (
            environment["GOOGLE_APPLICATION_CREDENTIALS"] == "/opt/airflow/google-credentials.json"
        )
        volumes = compose_with_creds["x-airflow-common"]["volumes"]
        assert any(
            str(volume) == f"{credentials_file}:/opt/airflow/google-credentials.json:ro"
            for volume in volumes
        )

    def test_s3_credentials_pass_through_only_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline_file = tmp_path / "my_pipeline.py"
        pipeline_file.write_text(
            "import hflow\n\napp = hflow.App('demo', data_root='s3://bkt/data')\n"
        )
        for variable_name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(variable_name, raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        s3_config = RuntimeConfig(pipeline_file=pipeline_file, data_root="s3://bkt/data")
        _, compose = _render(s3_config, tmp_path / "bundle")
        environment = compose["x-airflow-common"]["environment"]
        # Names pass through as ${VAR} references -- values NEVER land in the
        # bundle; unset variables get no line at all (an empty string would
        # read as a present credential and break instance-role fallback).
        assert environment["AWS_ACCESS_KEY_ID"] == "${AWS_ACCESS_KEY_ID}"
        assert environment["AWS_SECRET_ACCESS_KEY"] == "${AWS_SECRET_ACCESS_KEY}"
        assert "AWS_SESSION_TOKEN" not in environment
