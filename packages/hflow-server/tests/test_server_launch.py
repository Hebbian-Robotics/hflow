"""Observable startup output from ``hflow serve``."""

from pathlib import Path

import pytest
from hflow_server import ServerSettings
from hflow_server import server as server_module


def _launch_without_starting_server(
    host: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    def fake_create_app(_settings: ServerSettings) -> object:
        return object()

    def choose_requested_port(_host: str, requested_port: int) -> int:
        return requested_port

    def do_not_start_uvicorn(*_args: object, **_kwargs: object) -> None:
        pass

    monkeypatch.setattr(server_module, "create_app", fake_create_app)
    monkeypatch.setattr(server_module, "_first_free_port", choose_requested_port)
    monkeypatch.setattr(server_module.uvicorn, "run", do_not_start_uvicorn)

    data_root = tmp_path / "workspace"
    server_module.serve(
        ServerSettings(
            data_root=str(data_root),
            host=host,
            port=4356,
            open_browser=False,
        )
    )
    return data_root


def test_loopback_launch_output_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = _launch_without_starting_server("127.0.0.1", tmp_path, monkeypatch)

    captured = capsys.readouterr()
    assert captured.out == f"hflow serve: serving {data_root} at http://127.0.0.1:4356/\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    ("host", "warning_expected"),
    [
        ("127.0.0.42", False),
        ("::1", False),
        ("localhost", False),
        ("LOCALHOST", False),
        ("::", True),
        ("192.168.1.42", True),
        ("workspace.example.com", True),
    ],
)
def test_warning_is_only_printed_for_non_loopback_hosts(
    host: str,
    warning_expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _launch_without_starting_server(host, tmp_path, monkeypatch)

    captured = capsys.readouterr()
    assert bool(captured.err) is warning_expected


def test_wildcard_launch_warns_without_changing_the_dialable_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = _launch_without_starting_server("0.0.0.0", tmp_path, monkeypatch)

    captured = capsys.readouterr()
    assert captured.out == f"hflow serve: serving {data_root} at http://127.0.0.1:4356/\n"
    assert captured.err == (
        "hflow serve: warning: binding to 0.0.0.0 exposes this unauthenticated workspace "
        "to reachable networks; use a loopback host and SSH port forwarding for remote access\n"
    )
