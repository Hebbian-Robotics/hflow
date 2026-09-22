"""Public API compatibility and dependency isolation for utility consumers."""

import subprocess
import sys
from pathlib import Path


def _run_isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1])\n" + source,
            str(repository_root / "src"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_import_does_not_configure_application_logging() -> None:
    _run_isolated_python(
        "import logging\n"
        "root_logger = logging.getLogger()\n"
        "application_handler = logging.StreamHandler()\n"
        "root_logger.addHandler(application_handler)\n"
        "root_logger.setLevel(logging.ERROR)\n"
        "handlers_before = tuple(root_logger.handlers)\n"
        "level_before = root_logger.level\n"
        "import hflow\n"
        "assert tuple(root_logger.handlers) == handlers_before\n"
        "assert root_logger.level == level_before\n"
    )


def test_unconfigured_embedding_application_gets_no_fallback_output() -> None:
    completed = _run_isolated_python(
        "import logging\n"
        "import hflow\n"
        "logging.getLogger('hflow.embedding').warning('not application output')\n"
    )

    assert completed.stderr == ""


def test_application_handler_receives_propagated_hflow_records() -> None:
    _run_isolated_python(
        "import logging\n"
        "import hflow\n"
        "records = []\n"
        "class RecordingHandler(logging.Handler):\n"
        "    def emit(self, record):\n"
        "        records.append(record)\n"
        "root_logger = logging.getLogger()\n"
        "root_logger.setLevel(logging.INFO)\n"
        "root_logger.addHandler(RecordingHandler())\n"
        "logging.getLogger('hflow.embedding').warning('host-visible warning')\n"
        "assert len(records) == 1\n"
        "assert records[0].name == 'hflow.embedding'\n"
        "assert records[0].levelno == logging.WARNING\n"
    )


def test_cli_owns_default_and_verbose_log_levels() -> None:
    command = (
        "import logging\n"
        "import hflow.cli as cli\n"
        "def log_from_command(_arguments):\n"
        "    logger = logging.getLogger('hflow.cli_contract')\n"
        "    logger.warning('visible warning')\n"
        "    logger.info('verbose detail')\n"
        "    return 0\n"
        "cli._command_doctor = log_from_command\n"
    )
    default = _run_isolated_python(command + "raise SystemExit(cli.main(['doctor', 'unused']))\n")
    verbose = _run_isolated_python(
        command + "raise SystemExit(cli.main(['--verbose', 'doctor', 'unused']))\n"
    )

    assert "visible warning" in default.stderr
    assert "verbose detail" not in default.stderr
    assert "visible warning" in verbose.stderr
    assert "verbose detail" in verbose.stderr


def test_utilities_work_without_site_packages() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import runpy, sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "runpy.run_path(sys.argv[2], run_name='__main__')",
            str(repository_root / "src"),
            str(repository_root / "scripts" / "smoke_test_lazy_imports.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_public_exports_remain_available_to_full_pipeline_consumers() -> None:
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import hflow\n"
            "exported_names = set(hflow.__all__)\n"
            "assert exported_names <= set(dir(hflow))\n"
            "from hflow import *\n"
            "assert exported_names <= set(globals())\n"
            "from hflow.app import App as DirectApp\n"
            "assert App is hflow.App is DirectApp\n"
            "assert checks is hflow.checks\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
