"""``hflow.toml``: the project file, its parsing boundary, and precedence."""

from pathlib import Path

import pytest

import hflow
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

    def test_the_file_supplies_the_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "ingest.py").write_text(
            "import hflow\n\nkitchen = hflow.App('kitchen', data_root='./data')\n"
        )
        _write_config(tmp_path, 'pipeline = "src/ingest.py"\n')
        monkeypatch.chdir(tmp_path)

        assert cli_main(["manifest"]) == 0
        assert '"pipeline_name": "kitchen"' in capsys.readouterr().out

    def test_a_conventional_pipeline_py_needs_no_configuration_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        (tmp_path / "pipeline.py").write_text(
            "import hflow\n\nkitchen = hflow.App('kitchen', data_root='./data')\n"
        )
        monkeypatch.chdir(tmp_path)

        assert cli_main(["manifest"]) == 0
        assert '"pipeline_name": "kitchen"' in capsys.readouterr().out

    def test_the_conventional_pipeline_is_found_from_a_subdirectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Otherwise a command run from notebooks/ would take the data root
        from the project and the pipeline from the shell's directory, which is
        the one combination guaranteed to address two different things."""
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        (tmp_path / "pipeline.py").write_text(
            "import hflow\n\nkitchen = hflow.App('kitchen', data_root='./data')\n"
        )
        _write_config(tmp_path, 'data_root = "./data"\n')
        working_directory = tmp_path / "notebooks"
        working_directory.mkdir()
        monkeypatch.chdir(working_directory)

        assert cli_main(["manifest"]) == 0
        assert '"pipeline_name": "kitchen"' in capsys.readouterr().out

    def test_no_pipeline_anywhere_says_all_three_ways_to_supply_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)

        assert cli_main(["manifest"]) == 2
        errors = capsys.readouterr().err
        assert "--pipeline" in errors
        assert PROJECT_CONFIG_FILE_NAME in errors
        assert "pipeline.py" in errors

    def test_an_explicit_flag_outranks_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        (tmp_path / "configured.py").write_text(
            "import hflow\n\nconfigured = hflow.App('configured', data_root='./data')\n"
        )
        (tmp_path / "asked_for.py").write_text(
            "import hflow\n\nasked_for = hflow.App('asked-for', data_root='./data')\n"
        )
        _write_config(tmp_path, 'pipeline = "configured.py"\n')
        monkeypatch.chdir(tmp_path)

        assert cli_main(["manifest", "--pipeline", "asked_for.py"]) == 0
        assert '"pipeline_name": "asked-for"' in capsys.readouterr().out

    def test_the_pipeline_and_the_cli_address_one_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the file, and the case every other test here
        misses: they all configure `./data`, which is byte-identical to the
        built-in default, so they pass whether or not anything reads the file.

        A pipeline written the way the docs now teach -- `hflow.App("name")`,
        no `data_root=` -- has to land in the workspace `hflow.toml` names, or
        `hflow ingest` writes one corpus while `hflow curate` reads another and
        both report success against paths the user never configured.
        """
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        _write_config(tmp_path, 'data_root = "./corpus"\n')
        (tmp_path / "pipeline.py").write_text("import hflow\n\napp = hflow.App('demo')\n")
        monkeypatch.chdir(tmp_path)

        application = hflow.import_pipeline_application(str(tmp_path / "pipeline.py"))

        assert Path(str(application.data_root)).resolve() == (tmp_path / "corpus").resolve()

    def test_the_environment_still_outranks_the_file_for_the_pipeline_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_config(tmp_path, 'data_root = "./corpus"\n')
        (tmp_path / "pipeline.py").write_text("import hflow\n\napp = hflow.App('demo')\n")
        monkeypatch.setenv("HFLOW_DATA_ROOT", str(tmp_path / "from-environment"))
        monkeypatch.chdir(tmp_path)

        application = hflow.import_pipeline_application(str(tmp_path / "pipeline.py"))

        assert str(application.data_root) == str(tmp_path / "from-environment")

    def test_an_explicit_data_root_still_outranks_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        _write_config(tmp_path, 'data_root = "./corpus"\n')
        (tmp_path / "pipeline.py").write_text(
            "import hflow\n\napp = hflow.App('demo', data_root='./pinned')\n"
        )
        monkeypatch.chdir(tmp_path)

        application = hflow.import_pipeline_application(str(tmp_path / "pipeline.py"))

        assert Path(str(application.data_root)).name == "pinned"

    def test_an_unreadable_file_exits_two_rather_than_crashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
        _write_config(tmp_path, "data_root = [unclosed\n")
        monkeypatch.chdir(tmp_path)

        assert cli_main(["--help"]) == 2
        assert PROJECT_CONFIG_FILE_NAME in capsys.readouterr().err
