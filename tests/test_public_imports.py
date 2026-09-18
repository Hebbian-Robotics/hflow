"""Public API compatibility and dependency isolation for utility consumers."""

import subprocess
import sys
from pathlib import Path


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
