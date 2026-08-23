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
from hflow.app import parse_pipeline_spec
from hflow.cli import main
from hflow.runtime import AirflowHealth, RuntimeConfig, render_bundle
from hflow.runtime._client import AirflowClient, AirflowClientError, PasswordCredentials

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
    assert parse_pipeline_spec("pipe.py") == (Path("pipe.py"), "app")
    assert parse_pipeline_spec("dir/pipe.py:my_app") == (Path("dir/pipe.py"), "my_app")
    # A trailing non-identifier is part of the path, not a variable name.
    assert parse_pipeline_spec("dir/pipe.py:not-an-identifier") == (
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
    assert 'load_pipeline_application(pipeline_path, "my_app")' in dag_source
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


def test_up_api_port_reaches_the_rendered_env(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
) -> None:
    """--api-port is the whole point: it has to land in the bundle's .env."""
    bundle_dir = tmp_path / "runtime"
    exit_code = main(
        [
            "up",
            "--pipeline",
            str(pipeline_file),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(bundle_dir),
            "--api-port",
            "9090",
        ]
    )
    assert exit_code == 0
    assert "API_PORT=9090" in (bundle_dir / ".env").read_text()


def test_up_without_api_port_keeps_8080(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
) -> None:
    """The flag is additive: omitting it has to leave existing bundles where they were."""
    bundle_dir = tmp_path / "runtime"
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
    assert "API_PORT=8080" in (bundle_dir / ".env").read_text()


def test_up_api_port_does_not_rewrite_a_preserved_env(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
) -> None:
    """A second up with a different port leaves the first .env alone.

    Rendering is create-if-absent so an existing .env keeps its secrets and any
    user edits, which means --api-port only reaches a bundle that has none yet.
    docs/RUNTIME.md says so under "Port 8080 is taken"; this is what makes that
    true rather than a claim.
    """
    bundle_dir = tmp_path / "runtime"
    common = [
        "up",
        "--pipeline",
        str(pipeline_file),
        "--data-root",
        str(tmp_path / "data"),
        "--bundle-dir",
        str(bundle_dir),
    ]
    assert main([*common, "--api-port", "9090"]) == 0
    assert main([*common, "--api-port", "9091"]) == 0
    assert "API_PORT=9090" in (bundle_dir / ".env").read_text()


def test_up_rejects_an_out_of_range_api_port_before_rendering(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An argument error is reported as one, and nothing is left behind.

    The bundle directory staying absent is the point: validation happens
    before render_bundle, so there is no half-made bundle and no compose call
    to explain.
    """
    bundle_dir = tmp_path / "runtime"
    exit_code = main(
        [
            "up",
            "--pipeline",
            str(pipeline_file),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(bundle_dir),
            "--api-port",
            "70000",
        ]
    )
    assert exit_code == 2
    assert not bundle_dir.exists()
    assert compose_calls == []
    streams = capsys.readouterr()
    assert streams.out == ""
    error_output = streams.err
    assert "up: api_port 70000 is not in 1-65535" in error_output
    assert "Traceback" not in error_output
    assert "containers may still be running" not in error_output


def test_up_rejects_an_out_of_range_api_port_on_an_existing_bundle(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
) -> None:
    """The port is checked even where it could not have taken effect anyway.

    An existing .env is preserved, so --api-port is inert on a bundle that
    already has one. Validating before that is still the useful answer: the
    alternative is accepting a port, ignoring it, and reporting success. The
    preserved .env is left exactly as it was.
    """
    bundle_dir = tmp_path / "runtime"
    common = [
        "up",
        "--pipeline",
        str(pipeline_file),
        "--data-root",
        str(tmp_path / "data"),
        "--bundle-dir",
        str(bundle_dir),
    ]
    assert main([*common, "--api-port", "9090"]) == 0
    calls_after_first_up = len(compose_calls)

    assert main([*common, "--api-port", "70000"]) == 2
    assert "API_PORT=9090" in (bundle_dir / ".env").read_text()
    assert len(compose_calls) == calls_after_first_up


def test_up_reports_a_missing_pipeline_file_as_bad_input(
    compose_calls: list[list[str]],
    healthy_client: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A path that does not exist is an argument error, not a runtime failure.

    `up` used to let FileNotFoundError escape to the interpreter: a traceback
    and exit 1. 1 means the runtime started and then failed, so the caller is
    told containers may still be running and to tear them down. Nothing has
    started here -- render_bundle refuses before it makes the directory -- so
    the honest answer is 2, the same code every sibling command already returns
    for the same input.
    """
    missing = tmp_path / "no-such-pipeline.py"
    bundle_dir = tmp_path / "runtime"

    exit_code = main(
        [
            "up",
            "--pipeline",
            str(missing),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(bundle_dir),
        ]
    )

    assert exit_code == 2
    assert not bundle_dir.exists()
    assert compose_calls == []
    streams = capsys.readouterr()
    assert streams.out == ""
    # The reason, not just the path: a bare path was the complaint in #26.
    assert f"up: [Errno 2] No such file or directory: '{missing}'" in streams.err
    assert "Traceback" not in streams.err
    assert "containers may still be running" not in streams.err


def test_up_reports_a_missing_requirements_file_the_same_way(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--requirements takes the same path through render_bundle as --pipeline."""
    missing = tmp_path / "no-such-requirements.txt"

    exit_code = main(
        [
            "up",
            "--pipeline",
            str(pipeline_file),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(tmp_path / "runtime"),
            "--requirements",
            str(missing),
        ]
    )

    assert exit_code == 2
    assert compose_calls == []
    streams = capsys.readouterr()
    assert f"up: [Errno 2] No such file or directory: '{missing}'" in streams.err
    assert "Traceback" not in streams.err
    assert "containers may still be running" not in streams.err


def test_deploy_missing_pipeline_names_the_reason_not_just_the_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`deploy` exited 2 already, but printed a bare path.

    That bare form is the #26 complaint, which was only ever fixed in
    storage.py. The two render functions now raise the three-argument
    FileNotFoundError, so both commands print why the path failed. Other
    raise sites still use a message string (load_bundle) or a bare path
    (testing.py, storage.py:344); this pins the two that `up` and `deploy`
    reach, not a repo-wide convention.
    """
    missing = tmp_path / "no-such-pipeline.py"

    exit_code = main(
        [
            "deploy",
            "--pipeline",
            str(missing),
            "--data-root-uri",
            "s3://bucket/data",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 2
    streams = capsys.readouterr()
    assert f"deploy: [Errno 2] No such file or directory: '{missing}'" in streams.err


def test_up_reports_a_pipeline_directory_as_a_directory(
    compose_calls: list[list[str]],
    healthy_client: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command from the #102 report, end to end.

    The unit test on render_bundle pins the raise, and the ENOENT test above
    pins the handler, but nothing ran the reported command itself. Anything
    that later branched on errno between render_bundle and here would regress
    this to a traceback with the whole suite still green.
    """
    a_directory = tmp_path / "pipelines"
    a_directory.mkdir()
    bundle_dir = tmp_path / "runtime"

    exit_code = main(
        [
            "up",
            "--pipeline",
            str(a_directory),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(bundle_dir),
        ]
    )

    assert exit_code == 2
    assert not bundle_dir.exists()
    assert compose_calls == []
    streams = capsys.readouterr()
    assert f"up: [Errno 21] Is a directory: '{a_directory}'" in streams.err
    assert "No such file or directory" not in streams.err
    assert "Traceback" not in streams.err
    assert "containers may still be running" not in streams.err


def test_up_reports_a_requirements_directory_as_a_directory(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other `up` raise site, so reverting it alone cannot stay green."""
    a_directory = tmp_path / "reqs"
    a_directory.mkdir()
    bundle_dir = tmp_path / "runtime"

    exit_code = main(
        [
            "up",
            "--pipeline",
            str(pipeline_file),
            "--data-root",
            str(tmp_path / "data"),
            "--bundle-dir",
            str(bundle_dir),
            "--requirements",
            str(a_directory),
        ]
    )

    assert exit_code == 2
    assert compose_calls == []
    streams = capsys.readouterr()
    assert f"up: [Errno 21] Is a directory: '{a_directory}'" in streams.err
    assert "Traceback" not in streams.err


def test_deploy_reports_a_requirements_directory_as_a_directory(
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The fourth raise site, pinned at the CLI like its three siblings."""
    a_directory = tmp_path / "reqs"
    a_directory.mkdir()

    exit_code = main(
        [
            "deploy",
            "--pipeline",
            str(pipeline_file),
            "--data-root-uri",
            "s3://bucket/data",
            "--output-dir",
            str(tmp_path / "out"),
            "--requirements",
            str(a_directory),
        ]
    )

    assert exit_code == 2
    streams = capsys.readouterr()
    assert f"deploy: [Errno 21] Is a directory: '{a_directory}'" in streams.err
    assert "Traceback" not in streams.err


def test_deploy_pipeline_directory_says_is_a_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The #102 repro end to end: the path exists, it is just the wrong kind.

    The exit code and the absent traceback were already right (#90, #95); this
    pins the message only. The class stays FileNotFoundError on purpose, so the
    handler in cli.py that turns this into exit 2 keeps catching it.
    """
    a_directory = tmp_path / "pipelines"
    a_directory.mkdir()

    exit_code = main(
        [
            "deploy",
            "--pipeline",
            str(a_directory),
            "--data-root-uri",
            "s3://bucket/data",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "out").exists()
    streams = capsys.readouterr()
    assert f"deploy: [Errno 21] Is a directory: '{a_directory}'" in streams.err
    assert "No such file or directory" not in streams.err
    assert "Traceback" not in streams.err


def test_deploy_missing_requirements_names_the_reason_too(
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other raise site render_deploy_bundle owns.

    Without this, reverting the requirements raise in _deploy.py alone leaves
    the whole suite green while `deploy --requirements <missing>` goes back to
    printing a bare path.
    """
    missing = tmp_path / "no-such-requirements.txt"

    exit_code = main(
        [
            "deploy",
            "--pipeline",
            str(pipeline_file),
            "--data-root-uri",
            "s3://bucket/data",
            "--output-dir",
            str(tmp_path / "out"),
            "--requirements",
            str(missing),
        ]
    )

    assert exit_code == 2
    streams = capsys.readouterr()
    assert f"deploy: [Errno 2] No such file or directory: '{missing}'" in streams.err


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
        client_auth = self._auth
        assert isinstance(client_auth, PasswordCredentials)
        captured["credentials"] = (client_auth.username, client_auth.password)
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


def test_ingest_targets_a_remote_endpoint_without_any_bundle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hosted addressing path: --airflow-url + --dag-id + an environment
    credential drive a remote workspace with no local bundle directory at
    all. Credentials never travel via argv."""
    from hflow.runtime import BearerToken

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
        captured["base_url"] = self.base_url
        captured["auth"] = self._auth
        captured["dag_id"] = dag_id
        captured["uris"] = uris
        return {"dag_run_id": "manual__remote"}

    monkeypatch.setattr(AirflowClient, "ingest", fake_ingest)
    monkeypatch.setenv("HFLOW_AIRFLOW_TOKEN", "minted-token")
    exit_code = main(
        [
            "ingest",
            "episodes-in/a.mcap",
            "--airflow-url",
            "https://workspace.example.com",
            "--dag-id",
            "kitchen_ingest",
        ]
    )
    assert exit_code == 0
    assert captured["base_url"] == "https://workspace.example.com"
    assert captured["auth"] == BearerToken("minted-token")
    assert captured["dag_id"] == "kitchen_ingest"
    assert captured["uris"] == ["episodes-in/a.mcap"]
    output = capsys.readouterr().out
    assert "manual__remote" in output
    assert "watch it at https://workspace.example.com" in output


def test_ingest_remote_without_credentials_names_the_environment_fix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for variable in ("HFLOW_AIRFLOW_TOKEN", "HFLOW_AIRFLOW_USERNAME", "HFLOW_AIRFLOW_PASSWORD"):
        monkeypatch.delenv(variable, raising=False)
    exit_code = main(
        [
            "ingest",
            "a.mcap",
            "--airflow-url",
            "https://workspace.example.com",
            "--dag-id",
            "kitchen_ingest",
        ]
    )
    assert exit_code == 2
    assert "HFLOW_AIRFLOW_TOKEN" in capsys.readouterr().err


def test_explicit_bundle_dir_stays_local_even_with_remote_environment(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An exported HFLOW_AIRFLOW_URL must not hijack a command whose user
    explicitly addressed a local bundle."""
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://workspace.example.com")
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
        captured["base_url"] = self.base_url
        return {"dag_run_id": "manual__local"}

    monkeypatch.setattr(AirflowClient, "ingest", fake_ingest)
    assert main(["ingest", "a.mcap", "--bundle-dir", str(bundle_dir)]) == 0
    assert str(captured["base_url"]).startswith("http://127.0.0.1")
    assert "manual__local" in capsys.readouterr().out


def test_status_remote_reports_health_and_runs_without_a_bundle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Remote status needs no local bundle: environment addressing drives the
    REST-only path, and the output carries the endpoint, health, and runs."""
    healthy = AirflowHealth(
        components={"metadatabase": "healthy", "scheduler": "healthy", "dag_processor": "healthy"}
    )
    monkeypatch.setattr(AirflowClient, "health", lambda self: healthy)
    monkeypatch.setattr(AirflowClient, "dag", lambda self, dag_id: {"dag_id": dag_id})
    monkeypatch.setattr(
        AirflowClient,
        "dag_runs",
        lambda self, dag_id, **_kwargs: [{"dag_run_id": "manual__1", "state": "success"}],
    )
    monkeypatch.setenv("HFLOW_AIRFLOW_URL", "https://workspace.example.com")
    monkeypatch.setenv("HFLOW_AIRFLOW_DAG_ID", "kitchen_ingest")
    monkeypatch.setenv("HFLOW_AIRFLOW_TOKEN", "minted-token")
    assert main(["status"]) == 0
    output = capsys.readouterr().out
    assert "endpoint: https://workspace.example.com" in output
    assert "kitchen_ingest" in output
    assert "healthy" in output
    assert "manual__1 [success]" in output


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


def test_dag_registration_poll_does_not_sleep_past_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hflow.runtime._lifecycle import _wait_until_dag_registered

    class InstantlyAdvancingClock:
        def __init__(self) -> None:
            self.current_time_s = 0.0

        def monotonic(self) -> float:
            return self.current_time_s

        def sleep(self, duration_s: float) -> None:
            self.current_time_s += duration_s

    def unavailable_dag(_self: AirflowClient, dag_id: str) -> dict[str, object]:
        raise AirflowClientError(f"{dag_id} unavailable", status=404)

    clock = InstantlyAdvancingClock()
    monkeypatch.setattr("hflow.runtime._lifecycle.time", clock)
    monkeypatch.setattr(AirflowClient, "dag", unavailable_dag)
    client = AirflowClient("http://127.0.0.1:8080", "airflow", "password")

    with pytest.raises(TimeoutError, match="never registered"):
        _wait_until_dag_registered(
            client,
            ["demo_pipeline_ingest"],
            timeout_s=0.3,
            poll_interval_s=10.0,
        )
    assert clock.current_time_s == pytest.approx(0.3)


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
        assert (
            'load_pipeline_application(pipeline_path, "my_pipeline_app")'
            in sub_dag_file.read_text()
        )
    assert compose_calls == [[str(paths.compose_file), "up", "--detach"]]


def test_app_run_errors_helpfully_without_a_script_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = hflow.App("demo", data_root=tmp_path / "data")
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    with pytest.raises(RuntimeError, match="notebook"):
        app.run()
