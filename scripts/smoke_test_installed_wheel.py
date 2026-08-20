"""Verify release-only behavior from an installed wheel, without Docker."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hflow
from hflow.runtime import RuntimeConfig, infer_hflow_source, render_bundle


def smoke_test_installed_wheel() -> None:
    """Check package metadata, CLI-adjacent imports, and runtime rendering."""
    if hflow.__version__ == "0.0.0":
        raise RuntimeError("the installed distribution did not expose its package version")
    if infer_hflow_source() is not None:
        raise RuntimeError(
            "the smoke test imported a source checkout instead of the installed wheel"
        )

    with TemporaryDirectory() as temporary_directory:
        runtime_root = Path(temporary_directory)
        pipeline_file = runtime_root / "pipeline.py"
        pipeline_file.write_text('import hflow\n\napp = hflow.App("smoke")\n')
        bundle_paths = render_bundle(
            RuntimeConfig(pipeline_file=pipeline_file, data_root=runtime_root / "data"),
            runtime_root / "runtime",
        )
        compose_text = bundle_paths.compose_file.read_text()
        expected_install_target = f"hflow_install_target='hflow=={hflow.__version__}'"
        if expected_install_target not in compose_text:
            raise RuntimeError(f"runtime bundle omitted {expected_install_target!r}")
        if ":/opt/hflow-src:ro" in compose_text:
            raise RuntimeError("wheel runtime unexpectedly mounted an hflow source checkout")


if __name__ == "__main__":
    smoke_test_installed_wheel()
