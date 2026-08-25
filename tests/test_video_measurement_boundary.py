"""Architecture tests for the video-measurements incubation boundary."""

import ast
from pathlib import Path

import hflow._video_measurements as video_measurements
import hflow.ffmpeg as hflow_ffmpeg


def test_video_measurements_do_not_import_hflow_domain_modules() -> None:
    package_directory = Path(video_measurements.__file__).parent
    forbidden_imports_by_source: dict[str, list[str]] = {}
    for source_path in package_directory.glob("*.py"):
        syntax_tree = ast.parse(source_path.read_text(), filename=str(source_path))
        forbidden_imports = [
            imported_module
            for syntax_node in ast.walk(syntax_tree)
            if isinstance(syntax_node, ast.ImportFrom)
            and (imported_module := syntax_node.module) is not None
            and (imported_module == "hflow" or imported_module.startswith("hflow."))
        ]
        if forbidden_imports:
            forbidden_imports_by_source[source_path.name] = forbidden_imports

    assert forbidden_imports_by_source == {}


def test_ffmpeg_namespace_no_longer_exposes_the_incubating_measurements() -> None:
    assert not hasattr(hflow_ffmpeg, "frame_stats")
    assert not hasattr(hflow_ffmpeg, "luma_frames")
    assert not hasattr(hflow_ffmpeg, "rgb_frames")
