"""`hflow dataset create`: the pipeline's own policy as an immutable artifact."""

import json
from pathlib import Path

import duckdb
import pytest

import hflow
from hflow.cli import main as cli_main
from hflow.dataset import create_dataset, dataset_slug, default_dataset_sql
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

PIPELINE_SOURCE = """
import hflow

app = hflow.App("dataset-demo", default_checks=())


@app.check()
def duration(ep: hflow.Episode) -> hflow.CheckResult:
    return hflow.CheckResult(measurements={"seconds": 1.0})
"""


@pytest.fixture(scope="module")
def source_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(
        tmp_path_factory.mktemp("dataset-source") / "episode_0001.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
    )


@pytest.fixture
def ingested_project(tmp_path: Path, source_episode: Path) -> Path:
    """A project whose one episode has been ingested by its own pipeline."""
    data_root = tmp_path / "data"
    episodes_in = data_root / "episodes-in"
    episodes_in.mkdir(parents=True)
    (episodes_in / "episode_0001.mcap").write_bytes(source_episode.read_bytes())
    (tmp_path / "pipeline.py").write_text(PIPELINE_SOURCE)
    (tmp_path / "hflow.toml").write_text('data_root = "./data"\n')
    return tmp_path


def _ingest(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    monkeypatch.chdir(project)
    assert cli_main(["ingest", "episodes-in/episode_0001.mcap"]) == 0


class TestDefaultPolicy:
    def test_it_selects_what_the_pipeline_currently_stands_behind(
        self, ingested_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ingest(ingested_project, monkeypatch)
        app = hflow.import_pipeline_application(str(ingested_project / "pipeline.py"))

        dataset = create_dataset(app, "clean")

        assert dataset.row_count == 1
        assert dataset.total_episodes == 1

    def test_a_step_that_never_ran_excludes_its_episodes(
        self, ingested_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A check added after the corpus was ingested leaves a hole; the
        dataset reports zero rows rather than a corpus with a hole in it."""
        _ingest(ingested_project, monkeypatch)
        app = hflow.import_pipeline_application(str(ingested_project / "pipeline.py"))

        @app.check()
        def added_later(ep: hflow.Episode) -> hflow.CheckResult:
            return hflow.CheckResult()

        assert create_dataset(app, "with-a-hole").row_count == 0

    def test_evidence_only_checks_are_selected_not_excluded(
        self, ingested_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trap this policy exists to avoid. Evidence-only checks record
        `measured`, never `passed`, so a `status = 'passed'` reading would
        select nothing at all -- and HFlow's whole built-in library is
        evidence-only."""
        _ingest(ingested_project, monkeypatch)
        app = hflow.import_pipeline_application(str(ingested_project / "pipeline.py"))

        sql = default_dataset_sql(app)

        assert "'measured'" in sql
        assert create_dataset(app, "evidence-only").row_count == 1

    def test_a_quarantined_episode_is_left_out(
        self, ingested_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ingest(ingested_project, monkeypatch)
        app = hflow.import_pipeline_application(str(ingested_project / "pipeline.py"))
        sql = default_dataset_sql(app)
        assert "status != 'quarantined'" in sql

    def test_a_default_check_the_pipeline_supersedes_is_not_a_hole(
        self, tmp_path: Path, source_episode: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other empty-dataset trap, and the one the docs walk users into.

        Wrapping a built-in under a name of your own is how
        docs/how-to/enable-built-in-checks.md says to configure one, and the
        auto-registered copy then stands down and records `skipped` -- on every
        episode, forever, by design. Read as an unfilled hole it excluded every
        episode, so `hflow dataset create` returned an empty dataset that
        looked like a policy decision.
        """
        data_root = tmp_path / "data"
        (data_root / "episodes-in").mkdir(parents=True)
        (data_root / "episodes-in" / "episode_0001.mcap").write_bytes(source_episode.read_bytes())
        (tmp_path / "hflow.toml").write_text('data_root = "./data"\n')
        (tmp_path / "pipeline.py").write_text(
            """
import hflow
from hflow.checks import episode_duration

app = hflow.App("wrapper-demo")


@app.check()
def my_duration(ep: hflow.Episode) -> hflow.CheckResult:
    return episode_duration(ep)
"""
        )
        _ingest(tmp_path, monkeypatch)
        app = hflow.import_pipeline_application(str(tmp_path / "pipeline.py"))

        report = app.test(data_root / "episodes-in" / "episode_0001.mcap")
        superseded = next(run for run in report.checks if run.check.name == "episode_duration")
        assert superseded.status is hflow.CheckStatus.SKIPPED

        assert create_dataset(app, "with-a-wrapper").row_count == 1


class TestArtifacts:
    def test_the_manifest_and_its_provenance_land_together(
        self, ingested_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ingest(ingested_project, monkeypatch)
        app = hflow.import_pipeline_application(str(ingested_project / "pipeline.py"))

        dataset = create_dataset(app, "Clean Corpus!")

        manifest_file = Path(dataset.manifest_path)
        sidecar_file = Path(dataset.sidecar_path)
        assert manifest_file.parent.name == "manifests"
        assert manifest_file.name.startswith("clean-corpus-")
        assert sidecar_file.is_file()

        # The manifest is what `hflow export snapshot --manifest` consumes.
        episode_ids = duckdb.execute(
            "SELECT episode_id FROM read_parquet(?)", [str(manifest_file)]
        ).fetchall()
        assert len(episode_ids) == 1

        provenance = json.loads(sidecar_file.read_text())
        assert provenance["name"] == "Clean Corpus!"
        assert provenance["sql"] == dataset.sql
        assert provenance["row_count"] == 1
        assert provenance["pipeline"]["pipeline_name"] == "dataset-demo"
        assert provenance["pipeline"]["pipeline_version"] == app.pipeline_version
        assert [check["name"] for check in provenance["pipeline"]["checks"]] == ["duration"]

    def test_two_datasets_of_one_name_never_overwrite(
        self, ingested_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ingest(ingested_project, monkeypatch)
        app = hflow.import_pipeline_application(str(ingested_project / "pipeline.py"))

        first = create_dataset(app, "clean")
        second = create_dataset(app, "clean")

        assert first.manifest_path != second.manifest_path
        assert Path(first.manifest_path).is_file()
        assert Path(second.manifest_path).is_file()


class TestCli:
    def test_print_sql_writes_nothing_and_shows_the_policy(
        self,
        ingested_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _ingest(ingested_project, monkeypatch)
        capsys.readouterr()

        assert cli_main(["dataset", "create", "clean", "--print-sql"]) == 0

        printed = capsys.readouterr().out
        assert "FROM episodes" in printed
        assert not (ingested_project / "data" / "manifests").exists()

    def test_create_writes_the_pair_and_reports(
        self,
        ingested_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _ingest(ingested_project, monkeypatch)
        capsys.readouterr()

        assert cli_main(["dataset", "create", "clean"]) == 0

        printed = capsys.readouterr().out
        assert "1 of 1 episodes" in printed
        manifests = sorted((ingested_project / "data" / "manifests").iterdir())
        assert [path.suffix for path in manifests] == [".json", ".parquet"]

    def test_own_sql_keeps_the_artifact_and_the_provenance(
        self,
        ingested_project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _ingest(ingested_project, monkeypatch)
        capsys.readouterr()

        assert (
            cli_main(["dataset", "create", "mine", "--sql", "SELECT episode_id FROM episodes"]) == 0
        )

        sidecar = next((ingested_project / "data" / "manifests").glob("mine-*.json"))
        assert json.loads(sidecar.read_text())["sql"] == "SELECT episode_id FROM episodes"


def test_slugs_fall_back_rather_than_being_refused() -> None:
    assert dataset_slug("Clean Corpus!") == "clean-corpus"
    assert dataset_slug("!!!") == "dataset"


def test_a_bucket_backed_workspace_can_write_a_manifest(tmp_path: Path) -> None:
    """The hosted case, and the reason this moved out of the server: hosted
    workspaces are bucket data roots, and pinning used to refuse them with a
    501 because it did local path arithmetic."""
    pytest.importorskip("obstore", reason="bucket tests need the hflow[bucket] extra")
    from hflow.dataset import write_dataset_manifest
    from hflow.storage import BucketStorageRoot
    from hflow.workspace import Workspace

    remote_dir = tmp_path / "bucket"
    remote_dir.mkdir()
    storage_root = BucketStorageRoot(f"file://{remote_dir}", mirror=tmp_path / "mirror")
    workspace = Workspace(storage_root)
    app = hflow.App("bucket-demo", data_root=storage_root, default_checks=())
    source = synthesize_episode(
        tmp_path / "episode_0001.mcap", SyntheticEpisodeSpec(duration_s=1.0, cameras=())
    )
    app.process(source, record=True, verbose=False)

    written = write_dataset_manifest(workspace, name="clean", sql="SELECT episode_id FROM episodes")

    assert written.relative_key.startswith("manifests/clean-")
    assert written.report.row_count == 1
    # The object really landed in the store, not only in the local mirror.
    assert (remote_dir / written.relative_key).is_file()


def test_a_manifest_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two writers racing one key: the store arbitrates, not a check-then-write."""
    from hflow.dataset import ManifestAlreadyExistsError, write_dataset_manifest
    from hflow.workspace import Workspace

    data_root = tmp_path / "data"
    app = hflow.App("collide", data_root=data_root, default_checks=())
    source = synthesize_episode(
        tmp_path / "episode_0001.mcap", SyntheticEpisodeSpec(duration_s=1.0, cameras=())
    )
    app.process(source, record=True, verbose=False)
    workspace = Workspace.parse(data_root)

    write_dataset_manifest(
        workspace, name="clean", sql="SELECT episode_id FROM episodes", file_stem="pinned"
    )
    with pytest.raises(ManifestAlreadyExistsError):
        write_dataset_manifest(
            workspace, name="clean", sql="SELECT episode_id FROM episodes", file_stem="pinned"
        )
