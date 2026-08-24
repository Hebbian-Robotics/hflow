"""``hflow.toml``: the project file, its parsing boundary, and precedence."""

from pathlib import Path

import pytest

from hflow.cli import main as cli_main
from hflow.project import (
    PROJECT_CONFIG_FILE_NAME,
    NoProjectConfig,
    ProjectConfig,
    find_project_config,
)
from hflow.storage import BucketStorageRoot, LocalStorageRoot


def _write_config(directory: Path, body: str) -> Path:
    config_file = directory / PROJECT_CONFIG_FILE_NAME
    config_file.write_text(body)
    return config_file


class TestFindingTheFile:
    def test_no_file_anywhere_is_an_outcome_not_a_failure(self, tmp_path: Path) -> None:
        assert isinstance(find_project_config(tmp_path), NoProjectConfig)

    def test_found_from_a_subdirectory(self, tmp_path: Path) -> None:
        _write_config(tmp_path, 'data_root = "./data"\n')
        working_directory = tmp_path / "notebooks" / "scratch"
        working_directory.mkdir(parents=True)

        found = find_project_config(working_directory)

        assert isinstance(found, ProjectConfig)
        assert found.config_file == tmp_path / PROJECT_CONFIG_FILE_NAME

    def test_the_nearest_file_wins(self, tmp_path: Path) -> None:
        _write_config(tmp_path, 'data_root = "./outer"\n')
        inner = tmp_path / "inner"
        inner.mkdir()
        _write_config(inner, 'data_root = "./nested"\n')

        found = find_project_config(inner)

        assert isinstance(found, ProjectConfig)
        assert found.storage_root == LocalStorageRoot(inner / "nested")


class TestParsing:
    def test_relative_paths_resolve_against_the_file_not_the_caller(self, tmp_path: Path) -> None:
        """The whole point of the ancestor walk: one project, one answer.

        Resolved against the working directory instead, running a command
        from a subdirectory would address a different workspace than running
        it from the project root.
        """
        _write_config(tmp_path, 'data_root = "./data"\npipeline = "src/pipeline.py"\n')
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)

        found = find_project_config(deep)

        assert isinstance(found, ProjectConfig)
        assert found.storage_root == LocalStorageRoot(tmp_path / "data")
        assert found.pipeline_file == tmp_path / "src" / "pipeline.py"

    def test_a_bucket_data_root_stays_a_url(self, tmp_path: Path) -> None:
        _write_config(tmp_path, 'data_root = "gs://fleet-corpus/kitchen"\n')

        found = find_project_config(tmp_path)

        assert isinstance(found, ProjectConfig)
        assert isinstance(found.storage_root, BucketStorageRoot)
        assert str(found.storage_root) == "gs://fleet-corpus/kitchen"

    def test_every_setting_is_optional(self, tmp_path: Path) -> None:
        _write_config(tmp_path, "")

        found = find_project_config(tmp_path)

        assert found == ProjectConfig(config_file=tmp_path / PROJECT_CONFIG_FILE_NAME)

    def test_malformed_toml_is_refused_naming_the_file(self, tmp_path: Path) -> None:
        _write_config(tmp_path, "data_root = [unclosed\n")
        with pytest.raises(ValueError, match=PROJECT_CONFIG_FILE_NAME):
            find_project_config(tmp_path)

    def test_an_unknown_key_is_refused_rather_than_ignored(self, tmp_path: Path) -> None:
        """A typo in a settings file is silent by nature: the setting never
        takes effect and the user concludes the feature does not work."""
        _write_config(tmp_path, 'data_roots = "./data"\n')
        with pytest.raises(ValueError, match=r"unknown key 'data_roots'"):
            find_project_config(tmp_path)

    def test_a_future_config_version_is_refused(self, tmp_path: Path) -> None:
        _write_config(tmp_path, "config_version = 99\n")
        with pytest.raises(ValueError, match="config_version 99"):
            find_project_config(tmp_path)

    def test_a_non_string_setting_is_refused(self, tmp_path: Path) -> None:
        _write_config(tmp_path, "data_root = 3\n")
        with pytest.raises(ValueError, match="'data_root' must be a non-empty string"):
            find_project_config(tmp_path)


class TestCliPrecedence:
    def test_the_file_supplies_the_catalog_when_no_flag_or_environment_does(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        _write_config(tmp_path, 'data_root = "./workspace"\n')
        monkeypatch.chdir(tmp_path)

        # No --catalog: the default has to come from hflow.toml, and the
        # command fails on a catalog that is not there rather than on ./data.
        assert cli_main(["curate", "SELECT 1"]) == 2
        assert str(tmp_path / "workspace" / "catalog") in capsys.readouterr().err

    def test_the_environment_outranks_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file committed beside the pipeline must never pin a shell or a
        control plane to the workspace its author happened to use."""
        _write_config(tmp_path, 'data_root = "./workspace"\n')
        monkeypatch.setenv("HFLOW_DATA_ROOT", str(tmp_path / "from-environment"))
        monkeypatch.chdir(tmp_path)

        assert cli_main(["curate", "SELECT 1"]) == 2
        assert str(tmp_path / "from-environment" / "catalog") in capsys.readouterr().err

    def test_an_unreadable_file_exits_two_rather_than_crashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        _write_config(tmp_path, "data_root = [unclosed\n")
        monkeypatch.chdir(tmp_path)

        assert cli_main(["--help"]) == 2
        assert PROJECT_CONFIG_FILE_NAME in capsys.readouterr().err
