"""Media and canonical byte-serving: content, Range tolerance, containment."""

from fastapi.testclient import TestClient
from ui_test_fixtures import PopulatedWorkspace


def test_media_bytes_are_served_with_content_type(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.ok_episode_id}/media/wrist_cam")
    assert response.status_code == 200
    assert response.content == populated_workspace.contact_sheet_file.read_bytes()
    assert response.headers["content-type"] == "image/jpeg"


def test_a_range_request_does_not_500(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(
        f"/api/v1/episodes/{populated_workspace.ok_episode_id}/media/wrist_cam",
        headers={"Range": "bytes=0-3"},
    )
    assert response.status_code in (200, 206)
    if response.status_code == 206:
        assert response.content == populated_workspace.contact_sheet_file.read_bytes()[:4]


def test_media_outside_the_data_root_is_403_without_the_path(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.escaping_episode_id}/media/outside")
    assert response.status_code == 403
    assert str(populated_workspace.outside_media_file) not in response.text
    assert str(populated_workspace.outside_media_file.parent) not in response.text


def test_media_whose_file_vanished_is_404(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.escaping_episode_id}/media/missing")
    assert response.status_code == 404


def test_unknown_artifact_name_is_404(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(
        f"/api/v1/episodes/{populated_workspace.ok_episode_id}/media/no_such_artifact"
    )
    assert response.status_code == 404


def test_media_of_an_unknown_episode_is_404(api: TestClient) -> None:
    assert api.get("/api/v1/episodes/not-an-id/media/wrist_cam").status_code == 404


def test_canonical_bytes_are_served(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.ok_episode_id}/canonical")
    assert response.status_code == 200
    assert response.content == populated_workspace.canonical_file.read_bytes()


def test_canonical_outside_the_data_root_is_403(
    api: TestClient, populated_workspace: PopulatedWorkspace
) -> None:
    response = api.get(f"/api/v1/episodes/{populated_workspace.escaping_episode_id}/canonical")
    assert response.status_code == 403


def test_canonical_of_an_unknown_episode_is_404(api: TestClient) -> None:
    assert api.get("/api/v1/episodes/not-an-id/canonical").status_code == 404
