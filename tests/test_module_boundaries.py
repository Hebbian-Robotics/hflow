"""Architecture tests for documented HFlow module boundaries."""

import ast
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import hflow._video_measurements as video_measurements
import hflow.ffmpeg as hflow_ffmpeg

_HFLOW_PACKAGE_DIRECTORY = Path(video_measurements.__file__).parent.parent


class _ImportScope(Enum):
    ALL = auto()
    MODULE = auto()


@dataclass(frozen=True)
class _ImportBoundary:
    source_paths: tuple[Path, ...]
    restricted_module_prefix: str
    allowed_modules: frozenset[str]
    scope: _ImportScope
    rule: str


_DOCUMENTED_HFLOW_IMPORT_BOUNDARIES = (
    _ImportBoundary(
        source_paths=tuple(sorted(Path(video_measurements.__file__).parent.rglob("*.py"))),
        restricted_module_prefix="hflow",
        allowed_modules=frozenset(),
        scope=_ImportScope.ALL,
        rule=(
            "the video-measurements package must not import hflow or any hflow.* "
            "module because it is designed to be extracted into a standalone package"
        ),
    ),
    _ImportBoundary(
        source_paths=(_HFLOW_PACKAGE_DIRECTORY / "testing.py",),
        restricted_module_prefix="hflow",
        allowed_modules=frozenset({"hflow.ffmpeg", "hflow.format"}),
        scope=_ImportScope.ALL,
        rule=(
            "hflow.testing may import only hflow.ffmpeg and hflow.format; the fixture "
            "must not depend on the canonical writer or other HFlow domain modules"
        ),
    ),
    _ImportBoundary(
        source_paths=tuple(sorted(_HFLOW_PACKAGE_DIRECTORY.rglob("*.py"))),
        restricted_module_prefix="hflow_server",
        allowed_modules=frozenset(),
        scope=_ImportScope.MODULE,
        rule=(
            "hflow-server is optional for core installs, so hflow_server imports must stay "
            "inside the command or function that needs them"
        ),
    ),
)


class _ModuleScopeImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[ast.Import | ast.ImportFrom] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass


def _absolute_imports(source_path: Path, scope: _ImportScope) -> list[tuple[int, str, str]]:
    syntax_tree = ast.parse(source_path.read_text(), filename=str(source_path))
    if scope is _ImportScope.MODULE:
        collector = _ModuleScopeImportCollector()
        collector.visit(syntax_tree)
        import_nodes = collector.imports
    else:
        import_nodes = [
            syntax_node
            for syntax_node in ast.walk(syntax_tree)
            if isinstance(syntax_node, ast.Import | ast.ImportFrom)
        ]

    imports: list[tuple[int, str, str]] = []
    for syntax_node in import_nodes:
        if isinstance(syntax_node, ast.Import):
            for imported_alias in syntax_node.names:
                rendered_import = f"import {imported_alias.name}"
                if imported_alias.asname is not None:
                    rendered_import += f" as {imported_alias.asname}"
                imports.append((syntax_node.lineno, imported_alias.name, rendered_import))
        else:
            if syntax_node.level > 0:
                continue
            imported_module = syntax_node.module
            if imported_module is not None:
                imports.append((syntax_node.lineno, imported_module, ast.unparse(syntax_node)))
    return sorted(imports)


def test_documented_hflow_import_boundaries() -> None:
    violations: list[str] = []
    for boundary in _DOCUMENTED_HFLOW_IMPORT_BOUNDARIES:
        for source_path in boundary.source_paths:
            for line_number, imported_module, rendered_import in _absolute_imports(
                source_path, boundary.scope
            ):
                imports_restricted_module = (
                    imported_module == boundary.restricted_module_prefix
                    or imported_module.startswith(f"{boundary.restricted_module_prefix}.")
                )
                if imports_restricted_module and imported_module not in boundary.allowed_modules:
                    relative_path = source_path.relative_to(_HFLOW_PACKAGE_DIRECTORY.parent)
                    violations.append(
                        f"{relative_path}:{line_number}: {rendered_import} -- {boundary.rule}"
                    )

    assert not violations, "Documented HFlow import boundaries were crossed:\n" + "\n".join(
        violations
    )


def test_ffmpeg_namespace_no_longer_exposes_the_incubating_measurements() -> None:
    assert not hasattr(hflow_ffmpeg, "frame_stats")
    assert not hasattr(hflow_ffmpeg, "luma_frames")
    assert not hasattr(hflow_ffmpeg, "rgb_frames")
