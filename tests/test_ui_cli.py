"""``hflow ui`` flags to :class:`hflow_ui.UiSettings` -- the launch contract.

``_command_ui`` is the only place that turns the CLI's flags into a launch,
and one expression in it decides whether that launch is authenticated at all:
``UiSettings`` defaults to ``token=None`` (no session middleware), so a
regression there serves the corpus with no credential while every hflow-ui
auth test, which builds its settings by hand, still passes.

``serve`` is the process boundary and is monkeypatched here: these tests
assert the settings it was handed, never a running server.
"""

import sys
from pathlib import Path

import pytest
from hflow_ui import UiSettings

from hflow.cli import DEFAULT_UI_PORT, main


@pytest.fixture
def served_settings(monkeypatch: pytest.MonkeyPatch) -> list[UiSettings]:
    """Capture what ``hflow ui`` would launch, instead of launching it."""
    launches: list[UiSettings] = []
    monkeypatch.setattr("hflow_ui.serve", launches.append)
    return launches


def test_ui_flags_land_in_the_launch_settings(
    served_settings: list[UiSettings], tmp_path: Path
) -> None:
    exit_code = main(
        [
            "ui",
            "--data-root",
            str(tmp_path / "workspace"),
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
    assert settings.data_root == str(tmp_path / "workspace")
    assert settings.host == "0.0.0.0"
    assert settings.port == 9999
    assert settings.open_browser is False
    assert settings.read_only is True
    assert settings.pipeline == "kitchen.py:my_app"


def test_ui_mints_a_fresh_session_token_per_launch(
    served_settings: list[UiSettings], tmp_path: Path
) -> None:
    """Authentication is the default, and no two launches share a credential."""
    assert main(["ui", "--data-root", str(tmp_path)]) == 0
    assert main(["ui", "--data-root", str(tmp_path)]) == 0
    first, second = served_settings
    assert first.token and second.token
    assert first.token != second.token
    # The rest of the defaults a bare `hflow ui` promises.
    assert (first.host, first.port) == ("127.0.0.1", DEFAULT_UI_PORT)
    assert first.open_browser is True
    assert first.read_only is False
    assert first.pipeline is None


def test_ui_no_token_serves_without_authentication(
    served_settings: list[UiSettings], tmp_path: Path
) -> None:
    """The opt-out has to be explicit and total: no token, no middleware."""
    assert main(["ui", "--data-root", str(tmp_path), "--no-token"]) == 0
    (settings,) = served_settings
    assert settings.token is None


def test_ui_without_the_package_exits_with_the_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """hflow-ui is optional, so its absence is an instruction, not a traceback."""
    monkeypatch.setitem(sys.modules, "hflow_ui", None)
    assert main(["ui", "--data-root", str(tmp_path)]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "uv add hflow-ui" in streams.err
    assert "Traceback" not in streams.err
