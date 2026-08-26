import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from hflow import __version__
from hflow.cli import _build_parser, main


@pytest.mark.parametrize(
    ("command", "expected_description"),
    [
        ("down", "Stop the local Docker Compose runtime rendered by `hflow up`"),
        ("ingest", "Submit one or more episode URIs to the master ingest DAG"),
        ("status", "Inspect the health of the local or remote Airflow runtime"),
    ],
)
def test_runtime_command_help_has_a_description(
    command: str, expected_description: str, capsys: CaptureFixture
) -> None:
    with pytest.raises(SystemExit) as exception:
        _build_parser().parse_args([command, "--help"])

    assert exception.value.code == 0
    assert expected_description in capsys.readouterr().out


def test_cli_version(capsys: CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exception:
        _build_parser().parse_args(["--version"])

    assert exception.value.code == 0
    assert f"hflow {__version__}" in capsys.readouterr().out


def test_cli_manifest_prints_the_pipeline_manifest_json(
    tmp_path: Path, capsys: CaptureFixture
) -> None:
    pipeline_file = tmp_path / "kitchen.py"
    pipeline_file.write_text(
        "import hflow\n\n"
        "my_app = hflow.App('kitchen', data_root='./data', default_checks=())\n\n"
        '@my_app.check(version="1", critical=True)\n'
        "def blackout(ep: hflow.Episode) -> hflow.CheckResult:\n"
        "    return hflow.CheckResult()\n"
    )
    exit_code = main(["manifest", "--pipeline", f"{pipeline_file}:my_app"])
    assert exit_code == 0
    manifest_payload = json.loads(capsys.readouterr().out)
    assert manifest_payload["pipeline_name"] == "kitchen"
    assert manifest_payload["hflow_version"] == __version__
    (check_entry,) = manifest_payload["checks"]
    assert check_entry["name"] == "blackout"
    assert check_entry["version"] == "1"
    assert check_entry["critical"] is True


def test_cli_defaults_follow_the_environment_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A shell configured with HFLOW_DATA_ROOT must not have curate/stale/up
    silently address a hardcoded ./data beside the workspace the App uses."""
    monkeypatch.setenv("HFLOW_DATA_ROOT", str(tmp_path / "workspace"))
    parser = _build_parser()
    stale_arguments = parser.parse_args(["stale", "--pipeline-version", "abc"])
    assert stale_arguments.catalog == f"{tmp_path / 'workspace'}/catalog"
    up_arguments = parser.parse_args(["up", "--pipeline", "p.py"])
    assert up_arguments.data_root == str(tmp_path / "workspace")

    monkeypatch.delenv("HFLOW_DATA_ROOT")
    default_arguments = _build_parser().parse_args(["stale", "--pipeline-version", "abc"])
    assert default_arguments.catalog == "./data/catalog"


def test_cli_manifest_reports_a_broken_pipeline_instead_of_crashing(
    tmp_path: Path, capsys: CaptureFixture
) -> None:
    pipeline_file = tmp_path / "broken.py"
    pipeline_file.write_text("raise RuntimeError('boom at import')\n")
    exit_code = main(["manifest", "--pipeline", str(pipeline_file)])
    assert exit_code == 2
    assert "boom at import" in capsys.readouterr().err


def test_cli_manifest_does_not_inherit_a_pipeline_that_exits(
    tmp_path: Path, capsys: CaptureFixture
) -> None:
    """A config guard at import time is a boundary failure, not our exit code.

    ``sys.exit`` raises SystemExit -- a BaseException -- so importing user code
    can walk straight past an ``except Exception`` and take the calling program
    with it. Here that would mean the pipeline's own status instead of 2; in the
    workspace UI it would kill a long-lived server at startup.
    """
    pipeline_file = tmp_path / "guarded.py"
    pipeline_file.write_text("import sys\n\nsys.exit('set ROBOT_FLEET')\n")
    exit_code = main(["manifest", "--pipeline", str(pipeline_file)])
    assert exit_code == 2
    error_output = capsys.readouterr().err
    assert str(pipeline_file) in error_output
    assert "set ROBOT_FLEET" in error_output
