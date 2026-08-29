"""Architecture tests for documented HFlow module boundaries."""

import ast
from pathlib import Path

import hflow._video_measurements as video_measurements
import hflow.ffmpeg as hflow_ffmpeg

_HFLOW_PACKAGE_DIRECTORY = Path(video_measurements.__file__).parent.parent
_DOCUMENTED_HFLOW_IMPORT_BOUNDARIES = (
    (
        tuple(sorted(Path(video_measurements.__file__).parent.rglob("*.py"))),
        frozenset(),
        "the video-measurements package must not import hflow or any hflow.* "
        "module because it is designed to be extracted into a standalone package",
    ),
    (
        (_HFLOW_PACKAGE_DIRECTORY / "testing.py",),
        frozenset({"hflow.ffmpeg", "hflow.format"}),
        "hflow.testing may import only hflow.ffmpeg and hflow.format; the fixture "
        "must not depend on the canonical writer or other HFlow domain modules",
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


def _absolute_imports(
    source_path: Path, *, module_scope_only: bool = False
) -> list[tuple[int, str, str]]:
    syntax_tree = ast.parse(source_path.read_text(), filename=str(source_path))
    if module_scope_only:
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
    for source_paths, allowed_hflow_modules, rule in _DOCUMENTED_HFLOW_IMPORT_BOUNDARIES:
        for source_path in source_paths:
            for line_number, imported_module, rendered_import in _absolute_imports(source_path):
                imports_hflow = imported_module == "hflow" or imported_module.startswith("hflow.")
                if imports_hflow and imported_module not in allowed_hflow_modules:
                    relative_path = source_path.relative_to(_HFLOW_PACKAGE_DIRECTORY.parent)
                    violations.append(f"{relative_path}:{line_number}: {rendered_import} -- {rule}")

    assert not violations, "Documented HFlow import boundaries were crossed:\n" + "\n".join(
        violations
    )


def test_core_does_not_import_optional_server_at_module_scope() -> None:
    violations: list[str] = []
    rule = (
        "hflow-server is optional for core installs, so hflow_server imports must stay "
        "inside the command or function that needs them"
    )
    for source_path in sorted(_HFLOW_PACKAGE_DIRECTORY.rglob("*.py")):
        for line_number, imported_module, rendered_import in _absolute_imports(
            source_path, module_scope_only=True
        ):
            imports_server = imported_module == "hflow_server" or imported_module.startswith(
                "hflow_server."
            )
            if imports_server:
                relative_path = source_path.relative_to(_HFLOW_PACKAGE_DIRECTORY.parent)
                violations.append(f"{relative_path}:{line_number}: {rendered_import} -- {rule}")

    assert not violations, "Optional-server import boundary was crossed:\n" + "\n".join(violations)


def test_ffmpeg_namespace_no_longer_exposes_the_incubating_measurements() -> None:
    assert not hasattr(hflow_ffmpeg, "frame_stats")
    assert not hasattr(hflow_ffmpeg, "luma_frames")
    assert not hasattr(hflow_ffmpeg, "rgb_frames")
