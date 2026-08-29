"""Architecture test for the hflow_server module-level import boundary."""

import ast
from pathlib import Path

import hflow

_MUST_NOT_IMPORT_SERVER_AT_MODULE_LEVEL = (
    "hflow_server ships as a separate package (packages/hflow-server) so pipeline "
    "workers never carry it. A module-level import anywhere under src/hflow/ would "
    "make `import hflow.cli` -- and therefore every hflow command, not just "
    "`serve` -- fail for anyone who installed only the core package. The existing "
    "function-level import inside `_command_serve`, guarded by an `except "
    "ImportError`, is the correct pattern and must stay function-scoped."
)


def _module_level_server_imports(source_path: Path) -> list[str]:
    syntax_tree = ast.parse(source_path.read_text(), filename=str(source_path))
    forbidden: list[str] = []
    # Only the module's top-level statements matter here: a function-scoped
    # import (like the one in cli.py's _command_serve) is deliberately allowed,
    # so this does not walk into function or class bodies.
    for syntax_node in syntax_tree.body:
        if isinstance(syntax_node, ast.Import):
            for imported_alias in syntax_node.names:
                if imported_alias.name == "hflow_server" or imported_alias.name.startswith(
                    "hflow_server."
                ):
                    forbidden.append(f"import {imported_alias.name}")
        elif isinstance(syntax_node, ast.ImportFrom):
            if syntax_node.level > 0:
                continue
            imported_module = syntax_node.module
            if imported_module is not None and (
                imported_module == "hflow_server" or imported_module.startswith("hflow_server.")
            ):
                forbidden.append(f"from {imported_module} import ...")
    return forbidden


def test_no_module_under_hflow_imports_server_at_module_level() -> None:
    package_directory = Path(hflow.__file__).parent
    forbidden_imports_by_source: dict[str, list[str]] = {}
    for source_path in package_directory.rglob("*.py"):
        forbidden = _module_level_server_imports(source_path)
        if forbidden:
            forbidden_imports_by_source[str(source_path.relative_to(package_directory))] = forbidden

    assert forbidden_imports_by_source == {}, (
        f"{_MUST_NOT_IMPORT_SERVER_AT_MODULE_LEVEL}\n"
        "Offending imports: "
        + ", ".join(
            f"{source}: {', '.join(imports)}"
            for source, imports in forbidden_imports_by_source.items()
        )
    )
