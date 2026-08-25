"""render_bundle unit tests: bundle layout, compose shape, .env create-if-absent,
and the generated master/sub-DAG sources (no Docker, no Airflow imports)."""

import errno
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
    # The ingest stage graph: the master plus its four stage sub-DAGs, five
    # files total.
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
    # The database password is generated per bundle; the compose default
    # keeps pre-manifest bundles (whose preserved .env lacks the key) working.
    assert scheduler_env["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] == (
        "postgresql+psycopg2://airflow:${POSTGRES_PASSWORD:-airflow}@postgres/airflow"
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
    # Loopback by default; API_BIND_HOST in .env widens it deliberately.
    assert apiserver["ports"] == ["${API_BIND_HOST:-127.0.0.1}:${API_PORT}:8080"]
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
    assert '"$$hflow_install_target"' in script
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
    # uv, which the Airflow image ships on PATH, with the image's own
    # interpreter: a managed CPython would break the external-python pickle
    # boundary between Airflow's process and the task's.
    assert 'uv venv --python "$$(command -v python)" /opt/venvs/user' in script
    assert "export UV_PYTHON_DOWNLOADS=never" in script
    # The image exports VIRTUAL_ENV pointing at Airflow's OWN environment.
    assert "unset VIRTUAL_ENV" in script
    assert "export UV_PROJECT_ENVIRONMENT=/opt/venvs/user" in script
    # A locked project gets the versions it locked; --inexact keeps uv from
    # pruning pendulum and hflow, which are ours rather than the user's, and
    # --no-install-project keeps a build backend out of the read-only mount.
    assert "uv sync --project /opt/user --frozen --inexact --no-install-project" in script
    assert 'uv pip install --python "$$venv_python" -r /opt/user/requirements.txt' in script
    # The install target is a shell variable so bucket-mode bundles can add
    # the [bucket] extra; local mode renders the bare source path. Shell
    # variables stay $$-escaped in the YAML for Compose's interpolation.
    assert "hflow_install_target='/opt/hflow-src'" in script
    assert '"$$hflow_install_target"' in script
    # A user lockfile can now win the resolution, so the version the DAGs were
    # rendered by has to be checked rather than assumed.
    assert "these DAGs were rendered by" in script
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
    assert '$$bootstrap_packages "$$hflow_install_target"' in script
    assert 'echo "bootstrap: $$bootstrap_packages"' in script
    # A project's own inputs join the rebuild hash, so editing the pipeline
    # still refreshes code without rebuilding dependencies.
    assert "cat /opt/user/pyproject.toml" in script
    assert "cat /opt/user/uv.lock" in script


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
        # extracts each body to a temp file, so module-level names never
        # survive. The tasks are thin callers into hflow.stage_execution (one
        # owner of the run semantics; see tests/test_stage_execution.py for
        # the semantics themselves).
        assert "\n        from hflow.stage_execution import plan_stage_batches" in dag_source
        assert "\n        from hflow.stage_execution import (" in dag_source
        assert "load_pipeline_application" in dag_source
        assert "resolve_user_pipeline_path('my_pipeline.py')" in dag_source
        assert 'load_pipeline_application(pipeline_path, "app")' in dag_source
        # Each sub-DAG runs exactly its own stage of the stage graph.
        assert (
            f'process_stage_batch(app, batch["items"], "{stage.value}", orchestrator_run_id)'
            in dag_source
        )
        # The stagger, mapped batches, and budget gate are one contract.
        assert 'time.sleep(float(batch["start_delay_s"]))' in dag_source
        # partial() binds this sub-DAG's own run id across the whole fan-out.
        # It is a rendered template argument because the callable runs under
        # expect_airflow=False and so cannot read the Airflow context, and it
        # is what lets a catalog row name the run that produced it.
        assert (
            'process_batch.partial(orchestrator_run_id="{{ run_id }}").expand(batch=batches)'
            in dag_source
        )
        # The gate materializes the mapped results (lazy XCom proxies cannot
        # cross the external-python pickle boundary; list-typed task_ids keeps
        # a single-batch run from being flattened) and keeps the edge explicit.
        assert "{{ ti.xcom_pull(task_ids=['process_batch']) | list }}" in dag_source
        assert "batch_counts >> gate" in dag_source
        # The process task exports the environment-resolved data root, then
        # authoritatively refuses an app whose data_root points elsewhere.
        assert "expected_data_root = '/opt/airflow/data'" in dag_source
        assert "os.environ[DATA_ROOT_ENVIRONMENT_VARIABLE] = expected_data_root" in dag_source
        assert "require_application_data_root(app, expected_data_root)" in dag_source
        # Checks decide quarantine: the quarantine budget lives ONLY in meta;
        # every other stage keeps the error-budget half.
        if stage is Stage.META:
            assert "def quarantine_budget_gate(" in dag_source
            assert "summarize_quarantine_budget" in dag_source
            assert "summarize_error_budget" not in dag_source
        else:
            assert "def error_budget_gate(" in dag_source
            assert "summarize_error_budget" in dag_source
            assert "summarize_quarantine_budget" not in dag_source


def test_bundle_manifest_describes_the_bundle_and_load_bundle_prefers_it(
    config: RuntimeConfig, tmp_path: Path
) -> None:
    """hflow-bundle.json is the machine-readable bundle description a
    provisioning service (and load_bundle) reads instead of regexing
    generated code; pre-manifest bundles still load via the legacy path."""
    from dataclasses import replace

    from hflow.runtime import load_bundle

    paths, _ = _render(replace(config, task_queue="workspace-a"), tmp_path / "bundle")
    manifest_file = paths.bundle_dir / "hflow-bundle.json"
    manifest_payload = json.loads(manifest_file.read_text())
    assert manifest_payload["manifest_version"] == 1
    assert manifest_payload["kind"] == "compose"
    assert manifest_payload["hflow_version"] == hflow.__version__
    assert manifest_payload["dag_id"] == "my_pipeline_ingest"
    assert manifest_payload["sub_dag_ids"] == {
        "sync": "my_pipeline_sync",
        "meta": "my_pipeline_meta",
        "labels": "my_pipeline_labels",
        "media": "my_pipeline_media",
    }
    assert manifest_payload["data_root"] == "/opt/airflow/data"
    assert manifest_payload["app_variable"] == "app"
    assert manifest_payload["pipeline_filename"] == "my_pipeline.py"
    assert manifest_payload["requirements_included"] is True
    assert manifest_payload["task_queue"] == "workspace-a"

    # The manifest is authoritative for the dag id when present...
    manifest_payload["dag_id"] = "renamed_by_manifest"
    manifest_file.write_text(json.dumps(manifest_payload))
    assert load_bundle(paths.bundle_dir).dag_id == "renamed_by_manifest"
    # ...a FUTURE-versioned manifest may have re-semantified its fields, so
    # it falls back to the generated source rather than being misread...
    manifest_payload["manifest_version"] = 999
    manifest_file.write_text(json.dumps(manifest_payload))
    assert load_bundle(paths.bundle_dir).dag_id == "my_pipeline_ingest"
    # ...an unusable manifest falls back to the generated source (legacy)...
    manifest_file.write_text("{ not json")
    assert load_bundle(paths.bundle_dir).dag_id == "my_pipeline_ingest"
    # ...and a pre-manifest bundle (no file at all) still loads.
    manifest_file.unlink()
    assert load_bundle(paths.bundle_dir).dag_id == "my_pipeline_ingest"


def test_task_queue_routes_every_stage_task(config: RuntimeConfig, tmp_path: Path) -> None:
    """The worker-pool routing seam: task_queue stamps all stage tasks, and
    the default keeps generated DAGs queue-free (executor default)."""
    from dataclasses import replace

    default_paths, _ = _render(config, tmp_path / "default-bundle")
    for sub_dag_file in default_paths.sub_dag_files:
        assert "queue=" not in sub_dag_file.read_text()

    queued_paths, _ = _render(replace(config, task_queue="workspace-a"), tmp_path / "bundle")
    for sub_dag_file in queued_paths.sub_dag_files:
        dag_source = sub_dag_file.read_text()
        compile(dag_source, str(sub_dag_file), "exec")
        assert dag_source.count(", queue='workspace-a')") == 3

    with pytest.raises(ValueError, match="task_queue"):
        render_bundle(replace(config, task_queue="bad queue'name"), tmp_path / "refused-bundle")


def test_xcom_objectstorage_url_override(config: RuntimeConfig, tmp_path: Path) -> None:
    """Multi-machine executors need an XCom store every host reaches; the
    override replaces the single-host file:// default."""
    from dataclasses import replace

    _, compose = _render(
        replace(config, xcom_objectstorage_url="s3://tenant-bucket/xcom"),
        tmp_path / "bundle",
    )
    scheduler_env = compose["services"]["airflow-scheduler"]["environment"]
    assert scheduler_env["AIRFLOW__COMMON_IO__XCOM_OBJECTSTORAGE_PATH"] == (
        "s3://tenant-bucket/xcom"
    )


def test_endpoint_environment_variables_pass_through_by_name(
    config: RuntimeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exported HFLOW_ENDPOINT_* variables at render time are wired into the
    containers' environment as name-only ${VAR} references (same contract as
    bucket credentials: values never land in the bundle), so App's endpoint
    overlay works inside the runtime's task processes."""
    monkeypatch.setenv("HFLOW_ENDPOINT_JUDGE", "http://judge:8000/v1")
    _, compose = _render(config, tmp_path / "bundle")
    scheduler_env = compose["services"]["airflow-scheduler"]["environment"]
    assert scheduler_env["HFLOW_ENDPOINT_JUDGE"] == "${HFLOW_ENDPOINT_JUDGE}"
    assert "http://judge:8000/v1" not in (tmp_path / "bundle" / "docker-compose.yaml").read_text()

    monkeypatch.delenv("HFLOW_ENDPOINT_JUDGE")
    _, rerendered_compose = _render(config, tmp_path / "bundle-without")
    assert (
        "HFLOW_ENDPOINT_JUDGE"
        not in (rerendered_compose["services"]["airflow-scheduler"]["environment"])
    )


def test_env_file_carries_generated_postgres_password_and_bind_host(
    config: RuntimeConfig, tmp_path: Path
) -> None:
    paths, _ = _render(config, tmp_path / "bundle")
    env_values = dict(
        line.split("=", 1)
        for line in paths.env_file.read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert env_values["API_BIND_HOST"] == "127.0.0.1"
    # A generated secret, never the historical fixed default.
    assert env_values["POSTGRES_PASSWORD"] != "airflow"
    assert len(env_values["POSTGRES_PASSWORD"]) >= 16


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
            assert f"data_root={self.BUCKET_URL!r}" in dag_source
            assert f"expected_data_root = {self.BUCKET_URL!r}" in dag_source
            assert "require_application_data_root(app, expected_data_root)" in dag_source
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


@pytest.mark.parametrize("bad_port", [70000, 65536, 0, -1])
def test_api_port_outside_the_tcp_range_is_rejected_at_construction(
    config: RuntimeConfig, bad_port: int
) -> None:
    """The invariant belongs to the field, so it holds for library callers too.

    render_bundle is never reached: Compose would otherwise be the first thing
    to complain, long after the bundle is on disk.
    """
    from dataclasses import replace

    with pytest.raises(ValueError, match=rf"api_port {bad_port} is not in 1-65535"):
        replace(config, api_port=bad_port)


@pytest.mark.parametrize("good_port", [1, 8080, 65535])
def test_api_port_inside_the_tcp_range_is_accepted(config: RuntimeConfig, good_port: int) -> None:
    """Both ends of the range are legal, so the check cannot be exclusive."""
    from dataclasses import replace

    assert replace(config, api_port=good_port).api_port == good_port


def test_api_port_true_is_rejected_even_though_it_passes_the_range(
    config: RuntimeConfig,
) -> None:
    """bool is the case the range check cannot catch.

    True == 1, so the range test passes it, and the value is only ever str()-ed
    after that. Without this it reaches the rendered .env as API_PORT=True.
    False is covered alongside the other wrong types below, where it is the
    weaker case: it is 0, so the range check would reject it anyway, just with
    the wrong reason.
    """
    from dataclasses import replace

    with pytest.raises(ValueError, match="api_port must be an int, not bool: True"):
        replace(config, api_port=True)


def test_api_port_accepts_an_int_enum_member(config: RuntimeConfig) -> None:
    """An IntEnum member is an int, and CONTRIBUTING asks for typed variants.

    This is why the test is `isinstance` and not `type(...) is int`. The member
    renders as bare digits on 3.11+, so the .env it produces is identical to the
    one a plain int produces.
    """
    from dataclasses import replace
    from enum import IntEnum

    class Port(IntEnum):
        API = 9090

    assert replace(config, api_port=Port.API).api_port == 9090


def test_api_port_rejects_a_numpy_integer(config: RuntimeConfig) -> None:
    """A numpy integer is not an int, and a config field is not user data.

    catalog.py coerces numpy scalar measurements at its boundary because
    those are user data mid-append and refusing would cost the whole episode.
    api_port is set once by the caller, rendered into a .env that is never
    rewritten, and interpolated into api_base_url, so refusing it costs one
    line at the call site and nothing downstream.
    """
    from dataclasses import replace

    import numpy as np

    with pytest.raises(ValueError, match="api_port must be an int, not int64"):
        replace(config, api_port=np.int64(9090))


@pytest.mark.parametrize(
    ("bad_port", "type_name"),
    [("8080", "str"), (8080.0, "float"), (None, "NoneType"), (True, "bool"), (False, "bool")],
)
def test_api_port_of_the_wrong_type_is_rejected_at_construction(
    config: RuntimeConfig, bad_port: object, type_name: str
) -> None:
    """The field is annotated int, so anything else is refused where it is set.

    Two different prior behaviors end up here. A float passed the range check
    and rendered API_PORT=8080.0, which Compose will not take. A str or None
    failed the range check instead, but as a TypeError from the comparison,
    which nothing catches: `_command_up` handles ValueError only, so it left
    a traceback. Reporting the type first turns both into one ValueError.
    """
    from dataclasses import replace

    with pytest.raises(ValueError, match=f"api_port must be an int, not {type_name}"):
        replace(config, api_port=bad_port)


def test_pipeline_directory_says_is_a_directory(config: RuntimeConfig, tmp_path: Path) -> None:
    """A directory exists, so ENOENT was the wrong reason (#102)."""
    from dataclasses import replace

    a_directory = tmp_path / "pipelines"
    a_directory.mkdir()
    broken = replace(config, pipeline_file=a_directory)
    with pytest.raises(FileNotFoundError) as excinfo:
        render_bundle(broken, tmp_path / "bundle")
    assert excinfo.value.errno == errno.EISDIR
    assert excinfo.value.filename == str(a_directory)
    assert "Is a directory" in str(excinfo.value)


def test_requirements_directory_says_is_a_directory(config: RuntimeConfig, tmp_path: Path) -> None:
    from dataclasses import replace

    a_directory = tmp_path / "reqs"
    a_directory.mkdir()
    broken = replace(config, requirements_file=a_directory)
    with pytest.raises(FileNotFoundError) as excinfo:
        render_bundle(broken, tmp_path / "bundle")
    assert excinfo.value.errno == errno.EISDIR
    assert excinfo.value.filename == str(a_directory)
    assert "Is a directory" in str(excinfo.value)
