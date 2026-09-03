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
from hflow.app import parse_pipeline_spec, resolve_pipeline_spec_for_rendering
from hflow.cli import main
from hflow.runtime import AirflowDagRun, AirflowHealth, RuntimeConfig, render_bundle
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


class TestNamingTheAppInARenderedBundle:
    """The renderers bake the App's variable name into DAG source that runs
    somewhere else, days later, without importing the pipeline. Defaulting it
    to `app` made a pipeline binding `robot_app` render and exit 0, then fail
    every stage task inside a container -- while every other command discovered
    the name and worked."""

    @staticmethod
    def _write(tmp_path: Path, body: str) -> Path:
        pipeline_file = tmp_path / "pipeline.py"
        pipeline_file.write_text(body)
        return pipeline_file

    def test_the_name_the_pipeline_actually_binds_is_used(self, tmp_path: Path) -> None:
        pipeline_file = self._write(tmp_path, "import hflow\n\nrobot_app = hflow.App('fleet')\n")

        assert resolve_pipeline_spec_for_rendering(str(pipeline_file)) == (
            pipeline_file,
            "robot_app",
        )

    def test_an_explicit_name_in_the_address_still_wins(self, tmp_path: Path) -> None:
        pipeline_file = self._write(tmp_path, "import hflow\n\nrobot_app = hflow.App('fleet')\n")

        assert resolve_pipeline_spec_for_rendering(f"{pipeline_file}:something_else") == (
            pipeline_file,
            "something_else",
        )

    def test_two_apps_are_refused_rather_than_guessed(self, tmp_path: Path) -> None:
        pipeline_file = self._write(
            tmp_path,
            "import hflow\n\nkitchen = hflow.App('a')\ngarage = hflow.App('b')\n",
        )

        with pytest.raises(ValueError, match=r"more than one hflow\.App"):
            resolve_pipeline_spec_for_rendering(str(pipeline_file))

    def test_an_app_a_static_read_cannot_see_falls_back_rather_than_refusing(
        self, tmp_path: Path
    ) -> None:
        """A factory-built App is invisible to a source scan. Refusing it would
        break a setup that works today, so the historical default stands."""
        pipeline_file = self._write(
            tmp_path,
            "import hflow\n\n\ndef build():\n    return hflow.App('made')\n\n\napp = build()\n",
        )

        assert resolve_pipeline_spec_for_rendering(str(pipeline_file)) == (pipeline_file, "app")

    def test_up_renders_the_discovered_name_into_the_dag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        compose_calls: list[list[str]],
        healthy_client: None,
    ) -> None:
        """End to end, because the defect was that rendering SUCCEEDED: the
        bundle only reveals the wrong name once a task runs in a container."""
        self._write(
            tmp_path,
            "import hflow\n\nrobot_app = hflow.App('fleet', data_root='/opt/airflow/data')\n",
        )
        (tmp_path / "hflow.toml").write_text('data_root = "./data"\n')
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)

        assert main(["up"]) == 0

        rendered = (tmp_path / "data" / "runtime" / "dags" / "ingest_sync.py").read_text()
        assert 'load_pipeline_application(pipeline_path, "robot_app")' in rendered


def test_addressing_a_pipeline_by_spec_string_imports_its_siblings(tmp_path: Path) -> None:
    # The same multi-file guarantee the runtime path has: `hflow manifest`,
    # `hflow stale --pipeline`, and `hflow serve --pipeline` all arrive here.
    from hflow import import_pipeline_application

    (tmp_path / "rig_constants.py").write_text("FLEET_NAME = 'kitchen'\n")
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(
        "import hflow\n"
        "from rig_constants import FLEET_NAME\n\n"
        "fleet = hflow.App(FLEET_NAME, data_root='./data')\n"
    )
    application = import_pipeline_application(f"{pipeline_file}:fleet")
    assert application.name == "kitchen"


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


def test_up_reports_a_data_root_that_is_a_file(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#145. Every mkdir under <data-root>/runtime raised NotADirectoryError."""
    a_file = tmp_path / "afile"
    a_file.write_text("not a directory")

    exit_code = main(["up", "--pipeline", str(pipeline_file), "--data-root", str(a_file)])

    assert exit_code == 2
    assert compose_calls == []
    streams = capsys.readouterr()
    assert f"up: Not a directory: {a_file}" in streams.err
    assert "Traceback" not in streams.err
    assert "containers may still be running" not in streams.err


def test_up_still_accepts_a_data_root_that_is_not_there_yet(
    compose_calls: list[list[str]],
    healthy_client: None,
    pipeline_file: Path,
    tmp_path: Path,
) -> None:
    """The carve-out in #145: only a root that exists and is not a directory is
    refused. Whether a missing root should be refused is open on #143, and until
    that lands `up` must not answer it differently from how it always has.
    """
    missing_root = tmp_path / "not_there_yet"

    exit_code = main(["up", "--pipeline", str(pipeline_file), "--data-root", str(missing_root)])

    assert exit_code == 0
    assert compose_calls != []


def test_curate_reports_a_sql_file_that_is_a_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#145. read_text on a directory raises IsADirectoryError, which subclasses
    OSError and not FileNotFoundError, so it walked past the handler.
    """
    a_directory = tmp_path / "sql"
    a_directory.mkdir()

    exit_code = main(["curate", "--catalog", str(tmp_path), "--sql-file", str(a_directory)])

    assert exit_code == 2
    streams = capsys.readouterr()
    assert f"curate: [Errno 21] Is a directory: '{a_directory}'" in streams.err
    assert "No such file or directory" not in streams.err
    assert "Traceback" not in streams.err


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
    ) -> AirflowDagRun:
        client_auth = self._auth
        assert isinstance(client_auth, PasswordCredentials)
        captured["credentials"] = (client_auth.username, client_auth.password)
        captured["dag_id"] = dag_id
        captured["uris"] = uris
        captured["profile"] = profile
        captured["online"] = online
        return AirflowDagRun(
            dag_run_id="manual__test",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

    monkeypatch.setattr(AirflowClient, "ingest", fake_ingest)
    exit_code = main(["ingest", "  a.mcap  ", "sub/b.mcap", "--bundle-dir", str(bundle_dir)])
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
    ) -> AirflowDagRun:
        captured["base_url"] = self.base_url
        captured["auth"] = self._auth
        captured["dag_id"] = dag_id
        captured["uris"] = uris
        return AirflowDagRun(
            dag_run_id="manual__remote",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

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


def test_ingest_remote_with_hostless_url_names_the_fix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HFLOW_AIRFLOW_TOKEN", "minted-token")
    exit_code = main(
        [
            "ingest",
            "a.mcap",
            "--airflow-url",
            "http://",
            "--dag-id",
            "kitchen_ingest",
        ]
    )
    assert exit_code == 2
    error_output = capsys.readouterr().err
    assert "needs a host" in error_output
    assert "--airflow-url" in error_output
    assert "HFLOW_AIRFLOW_URL" in error_output


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
    ) -> AirflowDagRun:
        captured["base_url"] = self.base_url
        return AirflowDagRun(
            dag_run_id="manual__local",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

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
        lambda self, dag_id, **_kwargs: [
            AirflowDagRun(
                dag_run_id="manual__1",
                state="success",
                logical_date=None,
                start_date=None,
                end_date=None,
                conf={},
            )
        ],
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


def test_ingest_plumbs_profile_lane_and_steps_into_conf(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI selection reaches the SDK-owned trigger configuration."""
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    captured: dict[str, object] = {}

    def fake_trigger(
        self: AirflowClient,
        dag_id: str,
        conf: dict[str, object] | None = None,
        *,
        dag_run_id: str | None = None,
    ) -> AirflowDagRun:
        captured["dag_id"] = dag_id
        captured["conf"] = conf
        return AirflowDagRun(
            dag_run_id="manual__relabel",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

    monkeypatch.setattr(AirflowClient, "trigger_dag_run", fake_trigger)
    exit_code = main(
        [
            "ingest",
            "a.mcap",
            "--bundle-dir",
            str(bundle_dir),
            "--profile",
            "relabel",
            "--online",
            "--step",
            "caption",
            "--step",
            "embedding",
        ]
    )
    assert exit_code == 0
    assert captured["dag_id"] == "demo_pipeline_ingest"
    assert captured["conf"] == {
        "uris": ["a.mcap"],
        "profile": "relabel",
        "mode": "online",
        "step_names": ["caption", "embedding"],
    }
    assert "profile relabel, online lane" in capsys.readouterr().out


def test_ingest_rejects_unknown_profile(pipeline_file: Path, tmp_path: Path) -> None:
    """argparse enforces the RUN_PROFILES vocabulary before any HTTP happens."""
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    with pytest.raises(SystemExit) as exit_info:
        main(["ingest", "a.mcap", "--bundle-dir", str(bundle_dir), "--profile", "everything"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize("uri", [".", "./", "a/..", "a/b/../..", "/abs/x.mcap", "../x.mcap"])
def test_ingest_rejects_uris_outside_data_root(
    uri: str,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime URIs cannot be absolute or escape the configured data root."""
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)

    exit_code = main(["ingest", uri, "--bundle-dir", str(bundle_dir)])

    assert exit_code == 2
    assert "is not relative to the data root" in capsys.readouterr().err


def test_ingest_rejects_blank_uri_before_triggering(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = _rendered_bundle(tmp_path, pipeline_file)
    called = False

    def fake_ingest(
        self: AirflowClient,
        dag_id: str,
        uris: list[str],
        *,
        profile: str = "full",
        online: bool = False,
        dag_run_id: str | None = None,
    ) -> AirflowDagRun:
        nonlocal called
        called = True
        return AirflowDagRun(
            dag_run_id="manual__unexpected",
            state="queued",
            logical_date=None,
            start_date=None,
            end_date=None,
            conf={},
        )

    monkeypatch.setattr(AirflowClient, "ingest", fake_ingest)

    assert main(["ingest", "   ", "--bundle-dir", str(bundle_dir)]) == 2
    assert not called
    assert "non-empty" in capsys.readouterr().err


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


def test_serve_refuses_a_port_it_cannot_serve(capsys: pytest.CaptureFixture[str]) -> None:
    """Bad launch input is exit 2 and one line, the answer every other command gives.

    Exit 1 would say the server started and then failed. Nothing started: the
    port never got as far as the free-port probe.
    """
    for unusable_port in ("99999", "0"):
        exit_code = main(["serve", "--data-root", "/tmp", "--port", unusable_port, "--no-browser"])
        assert exit_code == 2
        stderr = capsys.readouterr().err
        assert stderr.startswith("serve: ")
        assert "1-65535" in stderr
        assert "Traceback" not in stderr


def test_serve_refuses_a_data_root_that_is_not_a_directory(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A file used to serve an empty workspace and say nothing about why.

    It answers with the stock errno sentence the rest of the CLI uses, so the
    caller is not left guessing why their workspace looks empty.
    """
    data_root_file = tmp_path / "not-a-directory"
    data_root_file.write_text("")

    exit_code = main(["serve", "--data-root", str(data_root_file), "--no-browser"])

    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("serve: ")
    assert "Not a directory" in stderr
    assert str(data_root_file) in stderr


def test_serve_refuses_a_host_it_cannot_bind(capsys: pytest.CaptureFixture[str]) -> None:
    """The probe's failure is a launch failure, so it exits 2 like the rest.

    This one arrives as ServerStartupError rather than ValueError, which is why
    it gets its own handler around ``serve`` instead of being folded into the
    construction handler above.
    """
    exit_code = main(
        ["serve", "--data-root", "/tmp", "--host", "not-a-host", "--port", "4512", "--no-browser"]
    )
    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("serve: ")
    assert "no free port" in stderr
    assert "Traceback" not in stderr


def test_serve_does_not_turn_a_running_server_crash_into_bad_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler catches the startup failure only, not RuntimeError at large.

    A RuntimeError out of a server that is already up means it started and then
    died, which is exit 1. Widening the handler to RuntimeError would report
    that as bad launch input and exit 2, which is the bug this issue is about
    in reverse.
    """
    import hflow_server

    def crash_once_running(_settings: object) -> None:
        raise RuntimeError("uvicorn fell over mid-run")

    monkeypatch.setattr(hflow_server, "serve", crash_once_running)
    with pytest.raises(RuntimeError, match="mid-run"):
        main(["serve", "--data-root", "/tmp", "--no-browser"])


def test_an_address_without_a_variable_discovers_the_sole_app(tmp_path: Path) -> None:
    from hflow import import_pipeline_application

    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text("import hflow\n\nkitchen = hflow.App('kitchen', data_root='./data')\n")
    assert import_pipeline_application(str(pipeline_file)).name == "kitchen"


def test_the_conventional_name_still_wins_over_discovery(tmp_path: Path) -> None:
    # Two Apps is normally ambiguous, but a file that binds `app` has already
    # said which one it means, and always resolved that way.
    from hflow import import_pipeline_application

    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(
        "import hflow\n\n"
        "staging = hflow.App('staging', data_root='./data')\n"
        "app = hflow.App('production', data_root='./data')\n"
    )
    assert import_pipeline_application(str(pipeline_file)).name == "production"


def test_several_apps_refuse_and_name_the_candidates(tmp_path: Path) -> None:
    from hflow import import_pipeline_application

    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(
        "import hflow\n\n"
        "kitchen = hflow.App('kitchen', data_root='./data')\n"
        "garage = hflow.App('garage', data_root='./data')\n"
    )
    with pytest.raises(ValueError, match=r"'kitchen', 'garage'"):
        import_pipeline_application(str(pipeline_file))
    # The suggestion in the message has to be an address that works.
    assert import_pipeline_application(f"{pipeline_file}:garage").name == "garage"


def test_a_file_with_no_app_says_so_rather_than_naming_a_missing_variable(
    tmp_path: Path,
) -> None:
    from hflow import import_pipeline_application

    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text("value = 42\n")
    with pytest.raises(ValueError, match=r"defines no hflow\.App"):
        import_pipeline_application(str(pipeline_file))


def test_an_app_imported_from_a_sibling_does_not_create_ambiguity(tmp_path: Path) -> None:
    # Only names bound in the pipeline file count, so a shared helper module
    # that builds an App cannot make every pipeline importing it ambiguous.
    from hflow import import_pipeline_application

    (tmp_path / "shared_rig.py").write_text(
        "import hflow\n\nshared = hflow.App('shared', data_root='./data')\n"
    )
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(
        "import hflow\nimport shared_rig  # noqa: F401\n\n"
        "kitchen = hflow.App('kitchen', data_root='./data')\n"
    )
    assert import_pipeline_application(str(pipeline_file)).name == "kitchen"


def test_naming_a_missing_variable_is_still_refused(tmp_path: Path) -> None:
    # Discovery is for addresses that did not say; one that did must not
    # quietly resolve to something else.
    from hflow import import_pipeline_application

    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text("import hflow\n\nkitchen = hflow.App('kitchen', data_root='./data')\n")
    with pytest.raises(ValueError, match=r"no hflow\.App named 'garage'"):
        import_pipeline_application(f"{pipeline_file}:garage")
