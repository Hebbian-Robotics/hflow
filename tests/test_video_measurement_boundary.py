"""Architecture tests for the video-measurements incubation boundary."""

import ast
from pathlib import Path

import hflow._video_measurements as video_measurements
import hflow.ffmpeg as hflow_ffmpeg

_PACKAGE_MUST_STAY_STANDALONE = (
    "The video-measurements package must not import hflow or any hflow.* "
    "submodule. It is designed to be extracted into a standalone package, and "
    "one import in the other direction -- reaching for an Episode or a catalog "
    "key from a measurement module -- would break that extraction. Import the "
    "HFlow domain from core into this package, never from this package back "
    "into core."
)


def _forbidden_imports(source_path: Path) -> list[str]:
    syntax_tree = ast.parse(source_path.read_text(), filename=str(source_path))
    forbidden: list[str] = []
    for syntax_node in ast.walk(syntax_tree):
        if isinstance(syntax_node, ast.Import):
            for imported_alias in syntax_node.names:
                if imported_alias.name == "hflow" or imported_alias.name.startswith("hflow."):
                    forbidden.append(f"import {imported_alias.name}")
        elif isinstance(syntax_node, ast.ImportFrom):
            if syntax_node.level > 0:
                continue
            imported_module = syntax_node.module
            if imported_module is not None and (
                imported_module == "hflow" or imported_module.startswith("hflow.")
            ):
                forbidden.append(f"from {imported_module} import ...")
    return forbidden


def test_video_measurements_do_not_import_hflow_domain_modules() -> None:
    package_directory = Path(video_measurements.__file__).parent
    forbidden_imports_by_source: dict[str, list[str]] = {}
    for source_path in package_directory.rglob("*.py"):
        forbidden = _forbidden_imports(source_path)
        if forbidden:
            forbidden_imports_by_source[source_path.name] = forbidden

    assert forbidden_imports_by_source == {}, (
        f"{_PACKAGE_MUST_STAY_STANDALONE}\n"
        "Offending imports: "
        + ", ".join(
            f"{source}: {', '.join(imports)}"
            for source, imports in forbidden_imports_by_source.items()
        )
    )


def test_ffmpeg_namespace_no_longer_exposes_the_incubating_measurements() -> None:
    assert not hasattr(hflow_ffmpeg, "frame_stats")
    assert not hasattr(hflow_ffmpeg, "luma_frames")
    assert not hasattr(hflow_ffmpeg, "rgb_frames")
