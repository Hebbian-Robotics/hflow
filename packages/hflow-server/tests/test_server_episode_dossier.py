"""GET /api/v1/episodes/{episode_id}: the dossier shape."""

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from hflow_server import ServerSettings, create_app
from ui_test_fixtures import STAMPS, PopulatedWorkspace

import hflow
from hflow.catalog import Catalog, CheckRunRow


def _dossier(api: TestClient, episode_id: str) -> dict:
    response = api.get(f"/api/v1/episodes/{episode_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_dossier_has_every_contract_section(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    dossier = _dossier(api, populated_workspace.ok_episode_id)
    assert set(dossier) == {
        "episode",
        "measurements",
        "check_runs",
        "intervals",
        "tags",
        "history",
        "media",
        "canonical_url",
    }


def test_ok_episode_status_and_identity(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    episode = _dossier(api, populated_workspace.ok_episode_id)["episode"]
    assert episode["episode_id"] == populated_workspace.ok_episode_id
    assert episode["status"] == "ok"
    assert episode["quarantine_tags"] == []
    assert episode["task"] == "fold_napkin"
    assert episode["operator"] == "alice"
    assert episode["embodiment"] == "arm-1"
    assert datetime.fromisoformat(episode["recorded_at"]).tzinfo is not None
    # Full "+00:00" offset, not DuckDB's bare "+00" (see _catalog).
    assert episode["recorded_at"].endswith("+00:00")


def test_quarantined_episode_carries_parsed_tags(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    dossier = _dossier(api, populated_workspace.quarantined_episode_id)
    assert dossier["episode"]["status"] == "quarantined"
    assert dossier["episode"]["quarantine_tags"] == ["failed:camera_blackout"]
    check_statuses = {run["check_name"]: run["status"] for run in dossier["check_runs"]}
    assert check_statuses["camera_blackout"] == "failed"


def test_measurements_are_the_latest_per_key(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    measurements = _dossier(api, populated_workspace.ok_episode_id)["measurements"]
    by_key = {entry["key"]: entry for entry in measurements}
    assert by_key["max_velocity"]["value_double"] == 2.0  # the second run's value
    assert by_key["max_velocity"]["check_name"] == "joint_check"
    assert by_key["max_velocity"]["check_version"] == "v1"
    assert by_key["nan_metric"]["value_double"] is None  # NaN is not JSON
    assert by_key["artifact/wrist_cam"]["value_text"] == str(populated_workspace.contact_sheet_file)


def test_check_runs_cover_every_recorded_run(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    check_runs = _dossier(api, populated_workspace.ok_episode_id)["check_runs"]
    assert len(check_runs) == 4  # two runs x two checks
    assert all(run["run_fingerprint"] for run in check_runs)
    assert {run["status"] for run in check_runs} == {"measured"}
    newest_first = [run["recorded_at"] for run in check_runs]
    assert newest_first == sorted(newest_first, reverse=True)


def test_intervals_come_from_the_latest_run_with_check_version(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    intervals = _dossier(api, populated_workspace.ok_episode_id)["intervals"]
    assert intervals == [
        {
            "label": "span",
            "start_ns": 0,
            "end_ns": 100,
            "check_name": "joint_check",
            "check_version": "v1",
        }
    ]


def test_tags_come_from_the_latest_run(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    tags = _dossier(api, populated_workspace.ok_episode_id)["tags"]
    assert len(tags) == 1
    assert tags[0]["tag"] == "seen"
    assert tags[0]["check_name"] == "joint_check"


def test_history_lists_every_append_newest_first(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    history = _dossier(api, populated_workspace.ok_episode_id)["history"]
    assert len(history) == 2
    assert history[0]["recorded_at"] >= history[1]["recorded_at"]
    assert history[0]["run_fingerprint"] != history[1]["run_fingerprint"]


def test_media_entries_carry_serving_urls(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    episode_id = populated_workspace.ok_episode_id
    media = _dossier(api, episode_id)["media"]
    assert media == [
        {
            "name": "wrist_cam",
            "uri": str(populated_workspace.contact_sheet_file),
            "url": f"/api/v1/episodes/{episode_id}/media/wrist_cam",
        }
    ]


def test_unservable_media_urls_are_null(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    dossier = _dossier(api, populated_workspace.escaping_episode_id)
    urls_by_name = {entry["name"]: entry["url"] for entry in dossier["media"]}
    assert urls_by_name == {"outside": None, "missing": None}
    assert dossier["canonical_url"] is None  # the canonical file escapes the root too


def test_canonical_url_present_for_contained_files(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    episode_id = populated_workspace.ok_episode_id
    dossier = _dossier(api, episode_id)
    assert dossier["canonical_url"] == f"/api/v1/episodes/{episode_id}/canonical"


def test_unknown_episode_is_a_404_with_detail(api: TestClient) -> None:
    response = api.get("/api/v1/episodes/definitely-not-an-id")
    assert response.status_code == 404
    assert "definitely-not-an-id" in response.json()["detail"]


def test_dossier_reports_unverified_when_a_critical_check_crashed(
    tmp_path: Path, unbuilt_assets_dir: Path
) -> None:
    """#164 item 7: the server renders the third status value.

    Built on its own root rather than the shared populated workspace, so the
    episode counts other tests pin stay as they are.
    """
    data_root = tmp_path / "data"
    catalog_root = data_root / "catalog"
    episodes_directory = data_root / "episodes"
    episodes_directory.mkdir(parents=True)
    canonical = episodes_directory / "crashed.canonical.mcap"
    canonical.write_bytes(b"canonical for a crashed critical check")

    catalog = Catalog(catalog_root)
    append = catalog.append_episode(
        canonical_path=canonical,
        stamps=STAMPS,
        episode_metadata={"task": "fold_napkin"},
        check_rows=[
            CheckRunRow(
                check_name="camera_blackout",
                check_version="v1",
                critical=True,
                status=hflow.CheckStatus.ERROR,
                duration_s=0.01,
                error="ffmpeg exited 1",
            )
        ],
    )

    api = TestClient(
        create_app(ServerSettings(data_root=str(data_root), assets_dir=unbuilt_assets_dir))
    )
    episode = _dossier(api, append.episode_id)["episode"]
    assert episode["status"] == "unverified"
    assert episode["quarantine_tags"] == []
