"""`hflow ingest` with no runtime addressed: the third executor.

Asserts the OUTCOME -- episodes processed and catalog rows written -- rather
than which branch the resolver took, so the executor can be restructured
without rewriting these.
"""

from pathlib import Path

import pytest

import hflow
from hflow.cli import main as cli_main
from hflow.curation import open_catalog_connection
from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

PIPELINE_SOURCE = """
import hflow
from hflow.checks import episode_duration

app = hflow.App("in-process")
app.check()(episode_duration)
"""


@pytest.fixture(scope="module")
def source_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthesize_episode(
        tmp_path_factory.mktemp("in-process-source") / "episode_0001.mcap",
        SyntheticEpisodeSpec(duration_s=1.0, cameras=()),
    )


@pytest.fixture
def project(tmp_path: Path, source_episode: Path) -> Path:
    """A project with a pipeline, a data root, and one episode to ingest."""
    data_root = tmp_path / "data"
    episodes_in = data_root / "episodes-in"
    episodes_in.mkdir(parents=True)
    (episodes_in / "episode_0001.mcap").write_bytes(source_episode.read_bytes())
    (tmp_path / "pipeline.py").write_text(PIPELINE_SOURCE)
    (tmp_path / "hflow.toml").write_text('data_root = "./data"\n')
    return tmp_path


def test_ingest_without_a_runtime_processes_and_records(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    monkeypatch.chdir(project)

    assert cli_main(["ingest", "episodes-in/episode_0001.mcap"]) == 0

    printed = capsys.readouterr()
    assert "no runtime addressed" in printed.err
    assert "sync: 1 processed" in printed.out

    # The point of the executor: a queryable corpus, not just a green exit.
    connection = open_catalog_connection(project / "data" / "catalog")
    try:
        rows = connection.execute('SELECT source_uri, "duration_s" FROM episodes').fetchall()
    finally:
        connection.close()
    assert len(rows) == 1
    source_uri, duration_s = rows[0]
    assert source_uri == "episodes-in/episode_0001.mcap"
    assert duration_s == pytest.approx(1.0, abs=0.2)


def test_a_pipeline_that_cannot_be_found_says_so_before_processing(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    (project / "pipeline.py").unlink()
    monkeypatch.chdir(project)

    assert cli_main(["ingest", "episodes-in/episode_0001.mcap"]) == 2
    assert "no pipeline found" in capsys.readouterr().err


def test_a_rendered_bundle_still_wins_over_running_in_process(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """In-process is the fallback, not a new default: a workspace that HAS a
    runtime keeps using it, or `hflow up` would silently stop mattering."""
    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    bundle_dir = project / "data" / "runtime"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "docker-compose.yaml").write_text("services: {}\n")
    monkeypatch.chdir(project)

    # The bundle is addressed but unloadable, so this fails at load_bundle
    # rather than falling through to the in-process path.
    assert cli_main(["ingest", "episodes-in/episode_0001.mcap"]) == 2
    printed = capsys.readouterr()
    assert "no runtime addressed" not in printed.err


def test_the_mass_failure_budget_still_applies_in_process(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand-rolled loop over app.process would report success here."""
    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")
    monkeypatch.chdir(project)

    assert cli_main(["ingest", "episodes-in/corrupt.mcap"]) == 1
    assert "processing errors" in capsys.readouterr().err


def test_an_absolute_uri_is_refused_before_anything_runs(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    monkeypatch.chdir(project)

    assert cli_main(["ingest", str(project / "data" / "episodes-in" / "episode_0001.mcap")]) == 2
    assert "not relative to the data root" in capsys.readouterr().err


def test_stages_run_in_graph_order_whatever_order_they_are_asked_for(project: Path) -> None:
    """Order is the stage graph's, not the caller's.

    Nothing can canonicalize after its checks have run, and labels read the
    quarantine state meta writes, so a set that happens to iterate
    labels-first must not run labels first.
    """
    from hflow.stage_execution import run_stages_directly

    app = hflow.App("in-process", data_root=project / "data")
    outcomes = run_stages_directly(
        app,
        ["episodes-in/episode_0001.mcap"],
        {hflow.Stage.LABELS, hflow.Stage.META, hflow.Stage.SYNC},
    )

    assert [outcome.stage for outcome in outcomes] == [
        hflow.Stage.SYNC,
        hflow.Stage.META,
        hflow.Stage.LABELS,
    ]
    assert all(outcome.counts["errors"] == 0 for outcome in outcomes)


def test_a_failed_source_is_recorded_where_it_can_be_found(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source that never canonicalized has no catalog row to be, and this
    executor has no task log behind it, so without the ledger the only trace
    of a failure is a traceback nobody kept."""
    from hflow.curation import open_catalog_connection

    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")
    monkeypatch.chdir(project)

    assert cli_main(["ingest", "episodes-in/corrupt.mcap"]) == 1
    assert "ingest_failures" in capsys.readouterr().err

    connection = open_catalog_connection(project / "data" / "catalog")
    try:
        rows = connection.execute(
            "SELECT source_uri, stage, failure_kind, error_type FROM ingest_failures"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [("episodes-in/corrupt.mcap", "sync", "source-unreadable", "InvalidMagic")]


def test_a_source_that_is_not_there_is_not_blamed_on_the_data(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Your file is bad" and "your file is missing" send people to different
    places, so they are different kinds."""
    from hflow.curation import open_catalog_connection

    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    monkeypatch.chdir(project)

    assert cli_main(["ingest", "episodes-in/never-uploaded.mcap"]) == 1

    connection = open_catalog_connection(project / "data" / "catalog")
    try:
        kinds = connection.execute("SELECT failure_kind FROM ingest_failures").fetchall()
    finally:
        connection.close()
    assert kinds == [("source-missing",)]


def test_replaying_the_same_failure_records_one_row(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hflow.curation import open_catalog_connection

    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")
    monkeypatch.chdir(project)

    cli_main(["ingest", "episodes-in/corrupt.mcap"])
    cli_main(["ingest", "episodes-in/corrupt.mcap"])

    connection = open_catalog_connection(project / "data" / "catalog")
    try:
        assert connection.execute("SELECT count(*) FROM ingest_failures").fetchone() == (1,)
    finally:
        connection.close()


def test_a_workspace_where_everything_failed_still_opens(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ledger writes the catalog's format marker first: otherwise a corpus
    whose every episode failed would hold a table no reader would open."""
    from hflow.curation import open_catalog_connection

    monkeypatch.delenv("HFLOW_DATA_ROOT", raising=False)
    monkeypatch.delenv("HFLOW_AIRFLOW_URL", raising=False)
    (project / "data" / "episodes-in" / "corrupt.mcap").write_bytes(b"not an mcap file")
    monkeypatch.chdir(project)

    cli_main(["ingest", "episodes-in/corrupt.mcap"])

    connection = open_catalog_connection(project / "data" / "catalog")
    try:
        assert connection.execute("SELECT count(*) FROM episodes").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM ingest_failures").fetchone() == (1,)
    finally:
        connection.close()
