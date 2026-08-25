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
