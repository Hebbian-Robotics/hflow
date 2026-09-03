import json
import subprocess
import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture

from hflow import __version__
from hflow.cli import _build_parser, main
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode
from hflow.transform import write_canonical_episode


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


@pytest.mark.parametrize(
    ("command", "expected_description"),
    [
        ("catalog", "Group commands for inspecting and exploring the append-only"),
        ("dataset", "Group commands that turn the pipeline's policy into version-pinned"),
        ("export", "Group commands for exporting catalog selections in portable downstream"),
        ("serve", "Serve this workspace over HTTP with REST endpoints over the catalog"),
    ],
)
def test_top_level_command_help_has_a_description(
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
    catalog_ui_arguments = parser.parse_args(["catalog", "ui", "--no-browser"])
    assert catalog_ui_arguments.catalog == f"{tmp_path / 'workspace'}/catalog"
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


def _run_module(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-m", "hflow", *args], capture_output=True, text=True, check=False
    )


def _run_script(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(["hflow", *args], capture_output=True, text=True, check=False)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--help"], id="help"),
        pytest.param(["doctor", "no-such-file.mcap"], id="missing-file-exit-2"),
    ],
)
def test_the_module_form_matches_the_console_script(args: list[str]) -> None:
    """``python -m hflow`` is a shim, so it has to agree with the script.

    Asserting the module merely starts would pass on a shim that swallowed the
    exit code, which is the way this can be wrong without looking wrong. The
    missing-file case is here for exactly that: it exits 2, so a shim returning
    ``main()`` instead of raising ``SystemExit(main())`` would report 0.
    """
    module = _run_module(*args)
    script = _run_script(*args)

    assert module.returncode == script.returncode
    assert module.stdout == script.stdout


def test_the_module_form_reports_a_conforming_file(tmp_path: Path) -> None:
    """The exit-0 path, over a real episode rather than --help."""
    source = synthesize_episode(
        tmp_path / "source.mcap", SyntheticEpisodeSpec(duration_s=1.0, cameras=())
    )
    canonical = tmp_path / "canonical.mcap"
    write_canonical_episode(source, canonical)

    module = _run_module("doctor", str(canonical))
    script = _run_script("doctor", str(canonical))

    assert module.returncode == 0
    assert module.returncode == script.returncode
    assert module.stdout == script.stdout


@pytest.mark.parametrize("scheme", ["s3", "gs", "az"])
def test_catalog_ui_cli_accepts_bucket_catalog_urls(
    scheme: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hflow.catalog_ui import CatalogUiSettings
    from hflow.cli import _command_catalog_ui

    served_catalog_roots: list[str] = []

    def record_serve(settings: CatalogUiSettings) -> None:
        served_catalog_roots.append(str(settings.catalog_root))

    monkeypatch.setattr("hflow.catalog_ui.serve_catalog_ui", record_serve)
    parser = _build_parser()
    catalog_url = f"{scheme}://robot-data/production/catalog"
    arguments = parser.parse_args(["catalog", "ui", "--no-browser", "--catalog", catalog_url])

    assert _command_catalog_ui(arguments) == 0
    assert served_catalog_roots == [catalog_url]


def test_catalog_ui_cli_reports_a_missing_remote_catalog_marker(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture
) -> None:
    def raise_missing_marker(_settings: object) -> None:
        raise FileNotFoundError(
            "gs://robot-data/production/catalog is not a catalog root "
            "(no format_version marker); expected the location a Catalog was "
            "created with, e.g. <data_root>/catalog"
        )

    monkeypatch.setattr("hflow.catalog_ui.serve_catalog_ui", raise_missing_marker)

    exit_code = main(
        [
            "catalog",
            "ui",
            "--no-browser",
            "--catalog",
            "gs://robot-data/production/catalog",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("catalog ui:")
    assert "format_version" in captured.err


def test_catalog_ui_cli_reports_a_missing_bucket_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture
) -> None:
    def raise_missing_obstore() -> object:
        raise ModuleNotFoundError(
            "bucket storage roots need the optional obstore backend -- install the "
            'extra: pip install "hflow[bucket]" (uv: uv add "hflow[bucket]")'
        )

    monkeypatch.setattr("hflow.storage._load_obstore", raise_missing_obstore)

    exit_code = main(
        ["catalog", "ui", "--no-browser", "--catalog", "s3://robot-data/production/catalog"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("catalog ui:")
    assert "hflow[bucket]" in captured.err


def test_catalog_ui_cli_reports_a_bucket_credentials_error(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture
) -> None:
    from obstore.exceptions import UnauthenticatedError

    def raise_credentials_error(_settings: object) -> None:
        raise UnauthenticatedError("provider credentials are unavailable")

    monkeypatch.setattr("hflow.catalog_ui.serve_catalog_ui", raise_credentials_error)

    exit_code = main(
        ["catalog", "ui", "--no-browser", "--catalog", "s3://robot-data/production/catalog"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == "catalog ui: provider credentials are unavailable\n"


def test_catalog_ui_cli_does_not_hide_programming_errors_for_bucket_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_programming_error(_settings: object) -> None:
        raise RuntimeError("unexpected implementation bug")

    monkeypatch.setattr("hflow.catalog_ui.serve_catalog_ui", raise_programming_error)

    with pytest.raises(RuntimeError, match="unexpected implementation bug"):
        main(["catalog", "ui", "--no-browser", "--catalog", "gs://robot-data/catalog"])


def test_import_lerobot_cli_reports_a_missing_bucket_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture
) -> None:
    def raise_missing_obstore(**_kwargs: object) -> list[str]:
        raise ModuleNotFoundError(
            "bucket storage roots need the optional obstore backend -- install the "
            'extra: pip install "hflow[bucket]" (uv: uv add "hflow[bucket]")'
        )

    monkeypatch.setattr("hflow.importers.lerobot.import_lerobot_dataset", raise_missing_obstore)

    exit_code = main(
        [
            "import",
            "lerobot",
            "--repo",
            "lerobot/pusht",
            "--output-dir",
            "s3://robot-data/production",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("import lerobot:")
    assert "hflow[bucket]" in captured.err


def test_import_lerobot_cli_reports_a_bucket_credentials_error(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture
) -> None:
    from obstore.exceptions import UnauthenticatedError

    def raise_credentials_error(**_kwargs: object) -> list[str]:
        raise UnauthenticatedError("provider credentials are unavailable")

    monkeypatch.setattr("hflow.importers.lerobot.import_lerobot_dataset", raise_credentials_error)

    exit_code = main(
        [
            "import",
            "lerobot",
            "--repo",
            "lerobot/pusht",
            "--output-dir",
            "s3://robot-data/production",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == "import lerobot: provider credentials are unavailable\n"


def test_import_lerobot_cli_accepts_bucket_output_dir_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str] = []

    def record_import(**kwargs: object) -> list[str]:
        received.append(str(kwargs["output_dir"]))
        return ["s3://robot-data/production/landing/lerobot_episode_0001.mcap"]

    monkeypatch.setattr("hflow.importers.lerobot.import_lerobot_dataset", record_import)

    exit_code = main(
        [
            "import",
            "lerobot",
            "--repo",
            "lerobot/pusht",
            "--output-dir",
            "s3://robot-data/production",
        ]
    )

    assert exit_code == 0
    assert received == ["s3://robot-data/production"]
