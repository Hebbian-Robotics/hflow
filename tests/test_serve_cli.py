"""``hflow serve`` flags to :class:`hflow_server.ServerSettings` -- the launch contract.

``_command_ui`` is the only place that turns the CLI's flags into a launch, so
a flag that stops reaching ``ServerSettings`` (or a default that drifts) is
invisible to every hflow-server test, which builds its settings by hand.
``--host`` is the one flag with a posture consequence: the server
authenticates nobody, so what it binds is the whole access-control story.

``serve`` is the process boundary and is monkeypatched here: these tests
assert the settings it was handed, never a running server.
"""

import sys
from pathlib import Path

import hflow_server
import pytest
from hflow_server import ServerSettings

from hflow.cli import DEFAULT_SERVER_PORT, main


@pytest.fixture
def served_settings(monkeypatch: pytest.MonkeyPatch) -> list[ServerSettings]:
    """Capture what ``hflow serve`` would launch, instead of launching it."""
    launches: list[ServerSettings] = []
    monkeypatch.setattr("hflow_server.serve", launches.append)
    return launches


def test_ui_flags_land_in_the_launch_settings(
    served_settings: list[ServerSettings], tmp_path: Path
) -> None:
    # A real directory: `serve` refuses a root that exists and is not one
    # before it builds the launch, and this test is about the flags reaching it.
    workspace_directory = tmp_path / "workspace"
    workspace_directory.mkdir()
    exit_code = main(
        [
            "serve",
            "--data-root",
            str(workspace_directory),
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--no-browser",
            "--read-only",
            "--pipeline",
            "kitchen.py:my_app",
        ]
    )
    assert exit_code == 0
    (settings,) = served_settings
    assert settings.data_root == str(workspace_directory)
    assert settings.host == "0.0.0.0"
    assert settings.port == 9999
    assert settings.open_browser is False
    assert settings.read_only is True
    assert settings.pipeline == "kitchen.py:my_app"


def test_serve_still_launches_over_a_data_root_that_is_not_there_yet(
    served_settings: list[ServerSettings], tmp_path: Path
) -> None:
    """A missing root is "nothing ingested yet", not bad input.

    Nothing creates the data root eagerly -- the catalog makes it on first
    append -- so on a fresh install ./data does not exist until something
    ingests, and `serve` is a reasonable first command. The server already
    renders that state instead of refusing it: /api/v1/config reports the
    missing catalog as a capability the frontend hides affordances behind.
    This is the other half of `up`'s carve-out in #145, so the two commands
    answer a missing root the same way.
    """
    missing_data_root = tmp_path / "not_there_yet"

    exit_code = main(["serve", "--data-root", str(missing_data_root), "--no-browser"])

    assert exit_code == 0
    (settings,) = served_settings
    assert settings.data_root == str(missing_data_root)


def test_a_bare_ui_launch_uses_the_documented_defaults(
    served_settings: list[ServerSettings], tmp_path: Path
) -> None:
    """What `hflow serve` with no flags promises -- loopback above all."""
    assert main(["serve", "--data-root", str(tmp_path)]) == 0
    (settings,) = served_settings
    assert (settings.host, settings.port) == ("127.0.0.1", DEFAULT_SERVER_PORT)
    assert settings.open_browser is True
    assert settings.read_only is False
    assert settings.pipeline is None


def test_ui_without_the_package_exits_with_the_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """hflow-server is optional, so its absence is an instruction, not a traceback."""
    monkeypatch.setitem(sys.modules, "hflow_server", None)
    assert main(["serve", "--data-root", str(tmp_path)]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "uv add hflow-server" in streams.err
    assert "Traceback" not in streams.err


@pytest.mark.parametrize("unusable_port", ["99999", "0"])
def test_serve_refuses_a_port_it_cannot_serve(
    capsys: pytest.CaptureFixture[str], unusable_port: str
) -> None:
    """Bad launch input is exit 2 and one line, the answer every other command gives.

    Exit 1 would say the server started and then failed. Nothing started: the
    port never got as far as the free-port probe.
    """
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

    def crash_once_running(_settings: object) -> None:
        raise RuntimeError("uvicorn fell over mid-run")

    monkeypatch.setattr(hflow_server, "serve", crash_once_running)
    with pytest.raises(RuntimeError, match="mid-run"):
        main(["serve", "--data-root", "/tmp", "--no-browser"])
