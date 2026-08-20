"""CLI plumbing for up/down/ingest/status and App.run -- no Docker involved.

``hflow.runtime._compose.run_compose_command`` is the single subprocess
seam; monkeypatching it captures every compose invocation. Airflow API calls
are stubbed at the AirflowClient method level.
"""

import sys
import types
from pathlib import Path

import pytest

import hflow
from hflow.cli import _parse_pipeline_spec, main
from hflow.runtime import AirflowHealth, RuntimeConfig, render_bundle
from hflow.runtime._client import AirflowClient

HEALTHY = AirflowHealth(
    components={
        "metadatabase": "healthy",
        "scheduler": "healthy",
        "dag_processor": "healthy",
        "triggerer": "healthy",
    }
)

PIPELINE_SOURCE = "import hflow\n\napp = hflow.App('demo', data_root='/opt/airflow/data')\n"


@pytest.fixture
def compose_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture compose subcommands ([subcommand, *args]) instead of running docker."""
    calls: list[list[str]] = []

    def fake_run_compose_command(
        compose_file: Path,
        *arguments: str,
        project_name: str | None = None,
        timeout_s: float = 0.0,
    ) -> str:
        calls.append([str(compose_file), *arguments])
        return "SERVICE   STATUS\nfake      running\n"

    monkeypatch.setattr("hflow.runtime._compose.run_compose_command", fake_run_compose_command)
    return calls


@pytest.fixture
def healthy_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AirflowClient, "wait_until_healthy", lambda self, **_kwargs: HEALTHY)
    monkeypatch.setattr(AirflowClient, "health", lambda self: HEALTHY)
    # start_runtime also verifies the ingest DAG registered before declaring
    # victory; stub the poll's endpoint so no real HTTP happens in unit tests.
    monkeypatch.setattr(AirflowClient, "dag", lambda self, dag_id: {"dag_id": dag_id})


@pytest.fixture
def pipeline_file(tmp_path: Path) -> Path:
    file = tmp_path / "demo_pipeline.py"
    file.write_text(PIPELINE_SOURCE)
    return file


def _rendered_bundle(tmp_path: Path, pipeline_file: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    render_bundle(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=tmp_path / "data"), bundle_dir
    )
    return bundle_dir


def test_parse_pipeline_spec_variants() -> None:
    assert _parse_pipeline_spec("pipe.py") == (Path("pipe.py"), "app")
    assert _parse_pipeline_spec("dir/pipe.py:my_app") == (Path("dir/pipe.py"), "my_app")
    # A trailing non-identifier is part of the path, not a variable name.
    assert _parse_pipeline_spec("dir/pipe.py:not-an-identifier") == (
        Path("dir/pipe.py:not-an-identifier"),
        "app",
    )


def test_up_renders_starts_and_prints(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data"
    exit_code = main(
        [
            "up",
            "--pipeline",
            f"{pipeline_file}:my_app",
            "--data-root",
            str(data_root),
        ]
    )
    assert exit_code == 0
    bundle_dir = data_root / "runtime"
    assert (bundle_dir / "docker-compose.yaml").is_file()
    # The master DAG triggers the sub-DAGs; the sub-DAGs load the user's app.
    assert "TriggerDagRunOperator" in (bundle_dir / "dags" / "ingest.py").read_text()
    dag_source = (bundle_dir / "dags" / "ingest_sync.py").read_text()
    assert 'getattr(pipeline_module, "my_app")' in dag_source
    assert compose_calls == [
        [str(bundle_dir / "docker-compose.yaml"), "up", "--detach"],
    ]
    output, errors = capsys.readouterr()
    assert "http://127.0.0.1:8080" in output
    assert "demo_pipeline_ingest" in output
    assert "credentials: airflow /" in output
    # Narration goes to stderr so stdout remains machine-readable.
    assert "rendering the runtime bundle" in errors
    assert "~2 GB" in errors
    assert "app.test() needs none of this" in errors
    assert "waiting for the ingest DAGs to register" in errors
    assert "rendering" not in output


def test_start_runtime_emits_progress_events_in_phase_order(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
) -> None:
    from hflow.runtime import start_runtime

    progress_events: list[str] = []
    start_runtime(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=tmp_path / "data"),
        tmp_path / "bundle",
        on_progress=progress_events.append,
    )
    assert len(progress_events) == 3
    rendering, starting, dag_wait = progress_events
    assert "rendering the runtime bundle" in rendering
    assert str(tmp_path / "bundle") in rendering
    assert "starting containers" in starting
    assert "one-time" in starting
    assert "waiting for the ingest DAGs to register" in dag_wait


def test_start_runtime_ticks_health_summaries_while_waiting(
    compose_calls: list[list[str]],
    pipeline_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The health wait narrates its latest summary on a throttled heartbeat."""
    from hflow.runtime import start_runtime

    unhealthy = AirflowHealth(
        components={
            "metadatabase": "healthy",
            "scheduler": "unhealthy",
            "dag_processor": "healthy",
        }
    )
    health_responses = iter([unhealthy, unhealthy, HEALTHY])
    monkeypatch.setattr(AirflowClient, "health", lambda self: next(health_responses))
    monkeypatch.setattr(AirflowClient, "dag", lambda self, dag_id: {"dag_id": dag_id})
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    # Zero throttle interval: every unhealthy poll becomes a visible tick.
    monkeypatch.setattr("hflow.runtime._lifecycle._HEALTH_PROGRESS_INTERVAL_S", 0.0)

    progress_events: list[str] = []
    start_runtime(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=tmp_path / "data"),
        tmp_path / "bundle",
        on_progress=progress_events.append,
    )
    health_ticks = [event for event in progress_events if "still waiting for Airflow" in event]
    assert len(health_ticks) == 2
    assert all("scheduler=unhealthy" in tick for tick in health_ticks)


def test_start_runtime_stays_silent_without_a_callback(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """on_progress defaults to None: no narration anywhere, contract unchanged."""
    from hflow.runtime import start_runtime

    paths, health = start_runtime(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=tmp_path / "data"),
        tmp_path / "bundle",
    )
    assert health is HEALTHY
    assert paths.bundle_dir == tmp_path / "bundle"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_up_honors_bundle_dir_and_hflow_source(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "elsewhere"
    source_dir = tmp_path / "hflow-src"
    source_dir.mkdir()
    (source_dir / "pyproject.toml").write_text('[project]\nname = "hflow"\n')
    exit_code = main(
        [
            "up",
            "--pipeline",
            str(pipeline_file),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(bundle_dir),
            "--hflow-source",
            str(source_dir),
        ]
    )
    assert exit_code == 0
    compose_text = (bundle_dir / "docker-compose.yaml").read_text()
    assert f"{source_dir.resolve()}:/opt/hflow-src:ro" in compose_text
    assert compose_calls[0][0] == str(bundle_dir / "docker-compose.yaml")


def test_up_from_published_install_uses_matching_distribution(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hflow.runtime.infer_hflow_source", lambda: None)
    bundle_dir = tmp_path / "published-bundle"
    exit_code = main(
        [
            "up",
            "--pipeline",
            str(pipeline_file),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(bundle_dir),
        ]
    )
    assert exit_code == 0
    compose_text = (bundle_dir / "docker-compose.yaml").read_text()
    assert f"hflow_install_target='hflow=={hflow.__version__}'" in compose_text
    assert ":/opt/hflow-src:ro" not in compose_text
    assert compose_calls == [[str(bundle_dir / "docker-compose.yaml"), "up", "--detach"]]


def test_down_invokes_compose_down(
    compose_calls: list[list[str]], pipeline_file: Path, tmp_path: Path
) -> None:
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    assert main(["down", "--bundle-dir", str(bundle_dir)]) == 0
    assert compose_calls == [[str(bundle_dir / "docker-compose.yaml"), "down"]]
    compose_calls.clear()
    assert main(["down", "--bundle-dir", str(bundle_dir), "--volumes"]) == 0
    assert compose_calls == [[str(bundle_dir / "docker-compose.yaml"), "down", "--volumes"]]


def test_ingest_uses_env_credentials_and_dag_id(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    env_values = dict(
        line.split("=", 1)
        for line in (bundle_dir / ".env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    captured: dict[str, object] = {}

    def fake_ingest(
        self: AirflowClient,
        dag_id: str,
        uris: list[str],
        *,
        profile: str = "full",
        online: bool = False,
        dag_run_id: str | None = None,
    ) -> dict[str, str]:
        captured["credentials"] = (self._username, self._password)
        captured["dag_id"] = dag_id
        captured["uris"] = uris
        captured["profile"] = profile
        captured["online"] = online
        return {"dag_run_id": "manual__test"}

    monkeypatch.setattr(AirflowClient, "ingest", fake_ingest)
    exit_code = main(["ingest", "a.mcap", "sub/b.mcap", "--bundle-dir", str(bundle_dir)])
    assert exit_code == 0
    assert captured["credentials"] == ("airflow", env_values["AIRFLOW_ADMIN_PASSWORD"])
    assert captured["dag_id"] == "demo_pipeline_ingest"
    assert captured["uris"] == ["a.mcap", "sub/b.mcap"]
    # Defaults: the full profile over the batch lane.
    assert captured["profile"] == "full"
    assert captured["online"] is False
    output = capsys.readouterr().out
    assert "manual__test" in output
    assert "profile full, batch lane" in output


def test_ingest_plumbs_profile_and_online_into_conf(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--profile/--online land in the trigger conf keys "profile"/"mode"."""
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    captured: dict[str, object] = {}

    def fake_trigger(
        self: AirflowClient,
        dag_id: str,
        conf: dict[str, object] | None = None,
        *,
        dag_run_id: str | None = None,
    ) -> dict[str, str]:
        captured["dag_id"] = dag_id
        captured["conf"] = conf
        return {"dag_run_id": "manual__relabel"}

    monkeypatch.setattr(AirflowClient, "trigger_dag_run", fake_trigger)
    exit_code = main(
        ["ingest", "a.mcap", "--bundle-dir", str(bundle_dir), "--profile", "relabel", "--online"]
    )
    assert exit_code == 0
    assert captured["dag_id"] == "demo_pipeline_ingest"
    assert captured["conf"] == {"uris": ["a.mcap"], "profile": "relabel", "mode": "online"}
    assert "profile relabel, online lane" in capsys.readouterr().out


def test_ingest_rejects_unknown_profile(pipeline_file: Path, tmp_path: Path) -> None:
    """argparse enforces the RUN_PROFILES vocabulary before any HTTP happens."""
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    with pytest.raises(SystemExit) as exit_info:
        main(["ingest", "a.mcap", "--bundle-dir", str(bundle_dir), "--profile", "everything"])
    assert exit_info.value.code == 2


def test_start_runtime_waits_for_all_five_dags(
    compose_calls: list[list[str]],
    pipeline_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`up` declares victory only once the master AND the sub-DAGs registered."""
    from hflow.runtime import start_runtime

    monkeypatch.setattr(AirflowClient, "wait_until_healthy", lambda self, **_kwargs: HEALTHY)
    polled_dag_ids: list[str] = []
    monkeypatch.setattr(
        AirflowClient,
        "dag",
        lambda self, dag_id: polled_dag_ids.append(dag_id) or {"dag_id": dag_id},
    )
    start_runtime(
        RuntimeConfig(pipeline_file=pipeline_file, data_root=tmp_path / "data"),
        tmp_path / "bundle",
    )
    assert polled_dag_ids == [
        "demo_pipeline_ingest",
        "demo_pipeline_sync",
        "demo_pipeline_meta",
        "demo_pipeline_labels",
        "demo_pipeline_media",
    ]


def test_status_reports_health_hints_and_services(
    compose_calls: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    unhealthy = AirflowHealth(
        components={
            "metadatabase": "healthy",
            "scheduler": "unhealthy",
            "dag_processor": "healthy",
            "triggerer": None,
        }
    )
    monkeypatch.setattr(AirflowClient, "health", lambda self: unhealthy)
    assert main(["status", "--bundle-dir", str(bundle_dir)]) == 0
    output = capsys.readouterr().out
    assert "UNHEALTHY" in output
    assert "tasks will never start" in output  # the plain-language scheduler hint
    assert "fake      running" in output  # docker compose ps passthrough
    assert compose_calls == [[str(bundle_dir / "docker-compose.yaml"), "ps"]]


def test_app_run_resolves_main_module_and_variable(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = hflow.App("demo", data_root=tmp_path / "data")
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(pipeline_file)
    setattr(fake_main, "my_pipeline_app", app)  # noqa: B010 -- dynamic module attribute
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    paths = app.run()
    assert paths.bundle_dir == tmp_path / "data" / "runtime"
    # The app variable lands in the sub-DAGs (the master never loads the app).
    for sub_dag_file in paths.sub_dag_files:
        assert 'getattr(pipeline_module, "my_pipeline_app")' in sub_dag_file.read_text()
    assert compose_calls == [[str(paths.compose_file), "up", "--detach"]]


def test_app_run_errors_helpfully_without_a_script_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = hflow.App("demo", data_root=tmp_path / "data")
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    with pytest.raises(RuntimeError, match="notebook"):
        app.run()
